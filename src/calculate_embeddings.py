import supervisely as sly
from dotenv import load_dotenv
import json
import os
import cv2
import numpy as np
import torch
import infer_utils


# example_model_names = [
#     "maxvit_large_tf_384.in21k_ft_in1k",
#     "facebook/convnext-xlarge-224-22k",
#     "beitv2_large_patch16_224",
#     "beitv2_large_patch16_224_in22k",
#     "openai/clip-vit-base-patch32",
#     "openai/clip-vit-large-patch14",
#     "facebook/flava-full",
#     "facebook/convnext-large-384",
#     "microsoft/beit-large-patch16-224-pt22k",
#     "microsoft/beit-large-patch16-384",
# ]


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


def extract_crops(img_np, ann, input_size_hw, instance_mode='image', hw_expand=(0,0), resize_interpolation=cv2.INTER_LINEAR):
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
        for label in ann.labels:
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


if __name__ == '__main__':

    model_name = 'facebook/convnext-tiny-224'
    instance_mode = 'both'
    batch_size = 2
    device = 'cpu'
    batch_size_api = 50
    instance_rect_expand = [0, 0]
    np_dtype = np.float32
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

    api = sly.Api()

    # init sly project
    project_id = sly.env.project_id()
    project = api.project.get_info_by_id(project_id)
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

    # init model
    model, cfg, format_input = infer_utils.create_model(model_name)
    model.to(device)
    input_size_hw = cfg['input_size']
    resize_interpolation = cfg['interpolation']


    all_features = []
    all_info = {
        'dataset_id': [],
        'image_id': [],
        'obj_cls': [],
        'crop_yxyx': []
    }

    for dataset in api.dataset.get_list(project.id):
        all_image_ids = api.image.get_list(dataset.id)
        for batch in sly.batched(all_image_ids, batch_size=batch_size_api):
            image_ids = [image.id for image in batch]
            anns_json = api.annotation.download_json_batch(dataset.id, image_ids)

            images = api.image.download_nps(dataset.id, image_ids)
            annotations = [sly.Annotation.from_json(ann, project_meta) for ann in anns_json]
            assert len(images) == len(annotations)

            for img_id, image, ann in zip(image_ids, images, annotations):
                crops, crops_obj_cls, crops_yxyx = extract_crops(image, ann, input_size_hw=input_size_hw,
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
