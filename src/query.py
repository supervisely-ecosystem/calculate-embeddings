import json
import numpy as np
import torch
import sklearn.decomposition
import supervisely as sly
from dotenv import load_dotenv
import os
import cv2
import sklearn.cluster
import scipy.spatial
from sklearn.neighbors import KDTree
import torchvision
from matplotlib import pyplot as plt

from calculate_embeddings import get_crops

def show_tensor(tensor, transpose=None, normalize=None, figsize=(10,10), title=None, nrow=None, padding=2, verbose=True, **kwargs):
    '''Flexible tool for visulizing tensors of any shape. Support batch_size >= 1.'''
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(np.array(tensor))
    tensor = tensor.detach().cpu().float()
    
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        if verbose: print('processing as black&white')
        tensor = tensor.repeat(1,3,1,1)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        if verbose: print('processing as black&white')
        tensor = tensor.unsqueeze(0).repeat(3,1,1).unsqueeze(0)
        
    if normalize is None:
        if tensor.max() <= 1.0 and tensor.min() >= 0.0:
            normalize = False
        else:
            if verbose: print('tensor has been normalized to [0., 1.]')
            normalize = True
            
    if transpose is None:
        transpose = True if tensor.shape[1] != 3 else False
    if transpose:
        tensor = tensor.permute(0,3,1,2)
    
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(tensor.shape[0])))
        
    grid = torchvision.utils.make_grid(tensor, normalize=normalize, nrow=nrow, padding=padding, **kwargs)
    plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    return plt.imshow(grid.permute(1,2,0))


if __name__ == '__main__':
    # init sly project
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

    api = sly.Api()

    project_id = sly.env.project_id()
    project = api.project.get_info_by_id(project_id)
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

    model_name = 'facebook/convnext-tiny-224'
    save_name = model_name.replace('/','_')
    with open(save_name+'_info.json', 'r') as f:
        all_info = json.load(f)
    with open(save_name+'_cfg.json', 'r') as f:
        cfg = json.load(f)
    all_info = [dict(tuple(zip(all_info.keys(), vals))) for vals in zip(*list(all_info.values()))]
    features = torch.load(save_name+'_embeddings.pt')


    # pca = sklearn.decomposition.PCA(32)
    # features = pca.fit_transform(features)
    # pca = umap.UMAP(min_dist=0.01)
    # proj = pca.fit_transform(features)
    # plt.scatter(proj[:,0], proj[:,1], s=10)
    # plt.savefig(save_name+'_PCA.jpg')
    
    kdtree = KDTree(features)

    n = 6
    query_idxs = list(range(n))
    d, idxs = kdtree.query(features[query_idxs], n)
    img_ids = idxs.flatten()
    img_ids = list(set([all_info[i]['image_id'] for i in img_ids]))
    imgs = api.image.download_nps(all_info[0]['dataset_id'], img_ids)
    img_id2img_np = dict(tuple(zip(img_ids, imgs)))

    crops = []
    for idx_i in idxs:
        selected_info = [all_info[i] for i in idx_i]
        for i, info in enumerate(selected_info):
            img = img_id2img_np[info['image_id']]
            crop = get_crops(img, [info['crop_yxyx']])[0]
            crop = cv2.resize(crop, cfg['input_size'][::-1], interpolation=cfg['interpolation'])
            crop = cv2.putText(crop, str(info['obj_cls']), (15,25), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,255,0], 2)
            crops.append(crop)
    crops = np.stack(crops)

    ax = show_tensor(crops, figsize=(12,12), title=model_name, verbose=False)
    plt.savefig(save_name+'_query.jpg')
