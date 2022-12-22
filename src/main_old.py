import supervisely as sly
from dotenv import load_dotenv
import json
import os
import cv2
import numpy as np
import torch
import infer_utils
import pandas as pd
from collections import defaultdict


def get_crops(img_np, yxyx_coords, hw_expand=(0,0)):
    # crop image to sub_images by yxyx coords
    crops = []
    h,w = hw_expand
    for yxyx in yxyx_coords:
        y1,x1,y2,x2 = yxyx
        y1,y2 = y1-h, y2+h
        x1,x2 = x1-w, x2+w
        y1,x1,y2,x2 = [max(0, coord) for coord in (y1,x1,y2,x2)]  # clip coords
        crop = img_np[y1:y2, x1:x2, :]
        crops.append(crop)
    return crops


def extract_crops(img_np, labels, input_size_hw, instance_mode='image', hw_expand=(0,0), resize_interpolation=cv2.INTER_LINEAR):
    # make all needed crops and resizes for image-annotation pair
    # return.__len__ may lay in [0, K+1], where K == n_instances.
    # return.__len__ can be 0 if mode is 'instance' and no any instances on image.
    assert instance_mode in ['image', 'instance', 'both'], f'unexpected instance_mode {instance_mode}'
    accepted_geometry = (sly.Rectangle, sly.Bitmap, sly.Polygon, sly.Polyline)
    result_crops = []
    result_obj_cls = []
    result_yxyx = []
    if instance_mode in ['image', 'both']:
        img = cv2.resize(img_np, input_size_hw[::-1], interpolation=resize_interpolation)
        result_crops.append(img)
        result_yxyx.append([0, 0, img_np.shape[0], img_np.shape[1]])
        result_obj_cls.append(None)
    if instance_mode in ['instance', 'both']:
        yxyx_croods = []
        for label in labels:
            if not isinstance(label.geometry, accepted_geometry):
                continue
            rect = label.geometry.to_bbox()
            yxyx_croods.append([rect.top, rect.left, rect.bottom, rect.right])
            result_obj_cls.append(label.obj_class.name)
        crops = get_crops(img_np, yxyx_croods, hw_expand=hw_expand)
        crops = [cv2.resize(crop, input_size_hw[::-1], interpolation=resize_interpolation) for crop in crops]
        result_yxyx += yxyx_croods
        result_crops += crops
    assert len(result_crops) == len(result_obj_cls) == len(result_yxyx)
    return result_crops, result_obj_cls, result_yxyx


def form_batches(crops, batch_size):
    idxs_split = list(range(0, len(crops), batch_size))+[None]  # e.g: [0,5,10,15,None]
    crops_batched = []
    for i in range(len(idxs_split)-1):
        batch = crops[idxs_split[i] : idxs_split[i+1]]
        batch = np.stack(batch)
        crops_batched.append(batch)
    return crops_batched


def normalize(img_batch, mean, std, np_dtype=np.float32):
    # img_batch: [B,H,W,C]
    assert img_batch.shape[3] == 3
    mean = np.array(mean, dtype=np_dtype)
    std = np.array(std, dtype=np_dtype)
    return (img_batch-mean)/std


def fetch_deltas(infos_old, infos_new):
    infos_old_list = [dict(tuple(zip(infos_old.keys(), vals))) for vals in zip(*list(infos_old.values()))]
    infos_new_list = [dict(tuple(zip(infos_new.keys(), vals))) for vals in zip(*list(infos_new.values()))]
    imgs_old = infos_old['image_id']
    imgs_new = infos_new['image_id']
    objs_old = infos_old['object_id']
    objs_new = infos_new['object_id']

    # remove imgages
    to_remove_img_ids = set(imgs_old) - set(imgs_new)
    # remove objects
    to_remove_obj_ids = set(objs_old) - set(objs_new)
    mask = (np.array(imgs_old) != to_remove_img_ids) & (np.array(objs_old) != to_remove_obj_ids)
    # add new images
    new_img_ids = list(set(imgs_new) - set(imgs_old))
    # update labels
    new_l2idx = dict(zip(objs_new, range(len(objs_new))))
    to_remove_obj_ids_2 = []
    update_labels_idxs = []
    for i,old_info in enumerate(infos_old_list):
        idx_in_new = new_l2idx.get(old_info['object_id'])
        if idx_in_new is None:
            to_remove_obj_ids_2.append(old_info['object_id'])
        else:
            new_info = infos_new_list[idx_in_new]
            if new_info['updatedAt'] != old_info['updatedAt']:
                update_labels_idxs.append(i)
    # add new labels
    old_l2idx = dict(zip(objs_old, range(len(objs_old))))
    new_labels_idxs = []
    for i,new_info in enumerate(infos_new_list):
        idx_in_old = old_l2idx.get(new_info['object_id'])
        if idx_in_old is None:
            new_labels_idxs.append(i)
    
    return mask, new_img_ids, update_labels_idxs, new_labels_idxs


if __name__ == '__main__':
    
    model_name = 'facebook/convnext-tiny-224'
    instance_mode = 'both'
    batch_size = 2
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size_api = 50
    instance_rect_expand = [0, 0]
    np_dtype = np.float32
    
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

    api = sly.Api()

    project_id = sly.env.project_id(raise_not_found=False)
    dataset_id = sly.env.dataset_id(raise_not_found=False)
    assert (project_id is not None) or (dataset_id is not None), 'either project_id or dataset_id must be set in local.env'

    if project_id is not None:
        datasets = api.dataset.get_list(project_id)
    else:
        datasets = [api.dataset.get_info_by_id(dataset_id)]
        project_id = datasets[0].project_id
    project = api.project.get_info_by_id(project_id)
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

    model_name = 'facebook/convnext-tiny-224'
    save_name = model_name.replace('/','_')

    # init model
    model, cfg, format_input = infer_utils.create_model(model_name)
    model.to(device)
    input_size_hw = cfg['input_size']
    resize_interpolation = cfg['interpolation']


    load_path = save_name+'_info.json'
    # TODO api.file.exists()
    if os.path.exists(save_name+'_info.json'):
        with open(load_path, 'r') as f:
            infos_old = json.load(f)
        embeddings = torch.load(save_name+'_embeddings.pt')
    else:
        infos_old = {
            'dataset_id': [],
            'image_id': [],
            'object_id': [],
            'crop_yxyx': [],
            'updatedAt': [],
        }
        embeddings = None

    infos_new = {
        'dataset_id': [],
        'image_id': [],
        'object_id': [],
        'crop_yxyx': [],
        'updatedAt': [],
    }
    for dataset in datasets:
        all_anns = api.annotation.get_list(dataset.id)
        # all_imgs = api.image.get_list(dataset.id)
        for ann in all_anns:
            infos_new['dataset_id'].append(dataset.id)
            infos_new['image_id'].append(ann.image_id)
            infos_new['object_id'].append(None)
            infos_new['crop_yxyx'].append([0,0,None,None])
            infos_new['updatedAt'].append('None')
            for object in ann.annotation['objects']:
                label = sly.Label.from_json(object, project_meta)
                rect = label.geometry.to_bbox()
                yxyx = [rect.top, rect.left, rect.bottom, rect.right]
                infos_new['dataset_id'].append(dataset.id)
                infos_new['image_id'].append(ann.image_id)
                infos_new['object_id'].append(object['id'])
                infos_new['crop_yxyx'].append(yxyx)
                infos_new['updatedAt'].append(object['updatedAt'])

        mask, new_img_ids, update_labels_idxs, new_labels_idxs = fetch_deltas(infos_old, infos_new)
        # embeddings = embeddings[mask]

        def infer_one(image, labels, instance_mode):
            crops, crops_obj_cls, crops_yxyx = extract_crops(image, labels, input_size_hw=input_size_hw,
                instance_mode=instance_mode, hw_expand=instance_rect_expand, resize_interpolation=resize_interpolation)
            if len(crops) == 0:
                return None
            crops_batched = form_batches(crops, batch_size=batch_size)
            print(len(crops))

            features = []
            # infer model
            for img_batch in crops_batched:
                # 1. prepare input
                img_batch = img_batch.astype(np.float32)/255
                img_batch = normalize(img_batch, cfg['mean'], cfg['std'], np_dtype=np_dtype)
                img_batch = img_batch.transpose(0,3,1,2)
                inputs = format_input(torch.tensor(img_batch))
                # 2. run infer
                features_batch = infer_utils.get_features(model, inputs, pool_mode='auto')
                features.append(features_batch.cpu().numpy())
            return np.concatenate(features)

        to_add_dataset = []
        to_add_img = []
        to_add_obj = []
        to_add_yxyx = []
        to_add_updatedAt = []

        # 1. update old
        img_id2labels = defaultdict(list)
        img_id2idxs = defaultdict(list)
        image_ids_upd = [infos_old['image_id'][i] for i in update_labels_idxs]
        image_ids = list(set(image_ids_upd))
        annotations = api.annotation.download_batch(dataset.id, image_ids)
        for img_id, ann in zip(image_ids, annotations):
            objects = ann.annotation['objects']
            id2obj = {obj['id'] : obj for obj in objects}
            for idx_upd in update_labels_idxs:
                obj_id = infos_old['object_id'][idx_upd]
                img_id2 = infos_old['image_id'][idx_upd]
                if img_id == img_id2:
                    img_id2labels[img_id].append(sly.Label.from_json(id2obj[obj_id], project_meta))
                    img_id2idxs[img_id].append(idx_upd)
        
        images = api.image.download_nps(dataset.id, list(img_id2labels))
        for image, labels, back_idxs in zip(images, img_id2labels.values(), img_id2idxs.values()):
            features = infer_one(image, labels, instance_mode='instance')
            if features is None: continue
            embeddings[back_idxs] = features

        
        # 2. add new labels in old imgs
        img_id2labels = defaultdict(list)
        img_id2idxs = defaultdict(list)
        image_ids_add = [infos_new['image_id'][i] for i in new_labels_idxs]
        image_ids = list(set(image_ids_add))
        annotations = api.annotation.download_batch(dataset.id, image_ids)
        for img_id, ann in zip(image_ids, annotations):
            objects = ann.annotation['objects']
            id2obj = {obj['id'] : obj for obj in objects}
            for idx_upd in new_labels_idxs:
                obj_id = infos_new['object_id'][idx_upd]
                img_id2 = infos_new['image_id'][idx_upd]
                if img_id == img_id2:
                    img_id2labels[img_id].append(sly.Label.from_json(id2obj[obj_id], project_meta))
                    img_id2idxs[img_id].append(idx_upd)
        
        to_add = []
        images = api.image.download_nps(dataset.id, list(img_id2labels))
        for image, labels, back_idxs in zip(images, img_id2labels.values(), img_id2idxs.values()):
            features = infer_one(image, labels, instance_mode='instance')
            if features is None: continue
            to_add.append(features)
        if to_add:
            embeddings = np.concatenate([embeddings, np.concatenate(to_add)])
        
        # 3. add absolutely new
        img_id2labels = defaultdict(list)
        annotations = api.annotation.download_batch(dataset.id, new_img_ids)
        for img_id, ann in zip(new_img_ids, annotations):
            for label in ann.annotation['objects']:
                img_id2labels[img_id].append(sly.Label.from_json(label, project_meta))

        to_add = []
        images = api.image.download_nps(dataset.id, list(img_id2labels))
        for image, labels in zip(images, img_id2labels.values()):
            features = infer_one(image, labels, instance_mode='both')
            if features is None: continue
            to_add.append(features)
        if to_add:
            if embeddings is None:
                embeddings = np.concatenate(to_add)
            else:
                embeddings = np.concatenate([embeddings, np.concatenate(to_add)])


    with open(save_name+'_info.json', 'w') as f:
        json.dump(infos_new, f)
    with open(save_name+'_cfg.json', 'w') as f:
        json.dump(cfg, f)
    torch.save(embeddings, save_name+'_embeddings.pt')

    print('done')
    print(embeddings.shape)


    raise
    for dataset in datasets:
        all_image_ids = api.image.get_list(dataset.id)
        for batch in sly.batched(all_image_ids, batch_size=batch_size_api):
            image_ids = [image.id for image in batch]
            anns_json = api.annotation.download_json_batch(dataset.id, image_ids)

            images = api.image.download_nps(dataset.id, image_ids)
            annotations = [sly.Annotation.from_json(ann, project_meta) for ann in anns_json]
            assert len(images) == len(annotations)

            for img_id, image, ann in zip(image_ids, images, annotations):
                crops, crops_obj_cls, crops_yxyx = extract_crops(image, ann.labels, input_size_hw=input_size_hw,
                    instance_mode=instance_mode, hw_expand=instance_rect_expand, resize_interpolation=resize_interpolation)
                if len(crops) == 0:
                    continue
                crops_batched = form_batches(crops, batch_size=batch_size)

                # infer model
                for img_batch in crops_batched:
                    # 1. prepare input
                    img_batch = img_batch.astype(np.float32)/255
                    img_batch = normalize(img_batch, cfg['mean'], cfg['std'], np_dtype=np_dtype)
                    img_batch = img_batch.transpose(0,3,1,2)
                    inputs = format_input(torch.tensor(img_batch))
                    # 2. run infer
                    features_batch = infer_utils.get_features(model, inputs, pool_mode='auto')
                    all_features.append(features_batch.cpu().numpy())

                # collect infos
                n = len(crops_obj_cls)
                all_info['dataset_id'] += [dataset.id]*n
                all_info['image_id'] += [img_id]*n
                all_info['obj_cls'] += crops_obj_cls
                all_info['crop_yxyx'] += crops_yxyx

    all_features = np.concatenate(all_features)

    save_name = model_name.replace('/','_')
    with open(save_name+'_info.json', 'w') as f:
        json.dump(all_info, f)
    with open(save_name+'_cfg.json', 'w') as f:
        json.dump(cfg, f)
    torch.save(all_features, save_name+'_embeddings.pt')

    print('done')
    print(all_features.shape)
