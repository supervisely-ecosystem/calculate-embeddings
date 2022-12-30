import os
import json
from collections import defaultdict
import numpy as np
import supervisely as sly
from dotenv import load_dotenv
import torch
import sklearn.manifold
import sklearn.cluster
import sklearn.decomposition
import umap
from matplotlib.colors import rgb2hex
import re


def calculate_projections(
    embeddings, all_info_list, projection_method, metric="euclidean", umap_min_dist=0.1
):
    try:
        if projection_method == "PCA":
            decomp = sklearn.decomposition.PCA(2)
            projections = decomp.fit_transform(embeddings)
        elif projection_method == "UMAP":
            decomp = umap.UMAP(min_dist=umap_min_dist, metric=metric)
            projections = decomp.fit_transform(embeddings)
        elif projection_method == "PCA-UMAP":
            decomp = sklearn.decomposition.PCA(64)
            projections = decomp.fit_transform(embeddings)
            decomp = umap.UMAP(min_dist=umap_min_dist, metric=metric)
            projections = decomp.fit_transform(projections)
        elif projection_method == "t-SNE":
            decomp = sklearn.manifold.TSNE(
                2, perplexity=min(30, len(all_info_list) - 1), metric=metric, n_jobs=-1
            )
            projections = decomp.fit_transform(embeddings)
        elif projection_method == "PCA-t-SNE":
            decomp = sklearn.decomposition.PCA(64)
            projections = decomp.fit_transform(embeddings)
            decomp = sklearn.manifold.TSNE(
                2, perplexity=min(30, len(all_info_list) - 1), metric=metric, n_jobs=-1
            )
            projections = decomp.fit_transform(projections)
        else:
            raise ValueError(f"unexpexted projection_method {projection_method}")
    except:
        raise RuntimeError(f"Try PCA as projection_method")
    return projections


def upload_embeddings(embeddings, info_updated, cfg, api, path_prefix, save_paths, team_id):
    os.makedirs(path_prefix, exist_ok=True)
    save_paths = {k: save_paths[k] for k in ["info", "cfg", "embeddings"]}
    with open(save_paths["info"], "w") as f:
        json.dump(info_updated, f)
    with open(save_paths["cfg"], "w") as f:
        json.dump(cfg, f)
    torch.save(embeddings, save_paths["embeddings"])
    api.file.upload_bulk(team_id, list(save_paths.values()), list(save_paths.values()))


def download_embeddings(api, path_prefix, save_paths, team_id):
    os.makedirs(path_prefix, exist_ok=True)
    api.file.download(team_id, "/" + save_paths["info"], save_paths["info"])
    api.file.download(team_id, "/" + save_paths["embeddings"], save_paths["embeddings"])
    api.file.download(team_id, "/" + save_paths["cfg"], save_paths["cfg"])
    with open(save_paths["info"], "r") as f:
        all_info = json.load(f)
    with open(save_paths["cfg"], "r") as f:
        cfg = json.load(f)
    embeddings = torch.load(save_paths["embeddings"])
    return embeddings, all_info, cfg


def make_series(projections, all_info_list, project_meta):
    x = projections[:, 1].tolist()
    y = projections[:, 0].tolist()

    series = defaultdict(list)
    global_idxs_mapping = defaultdict(list)
    for i in range(len(all_info_list)):
        obj_cls = str(all_info_list[i]["object_cls"] or "Image")
        series[obj_cls].append({"x": x[i], "y": y[i]})
        global_idxs_mapping[obj_cls].append(i)

    series = [{"name": k, "data": v} for k, v in series.items()]
    obj2color = {x.name: rgb2hex(np.array(x.color) / 255) for x in project_meta.obj_classes.items()}
    obj2color["Image"] = "#222222"
    colors = [obj2color[s["name"]] for s in series]

    return series, colors, global_idxs_mapping
