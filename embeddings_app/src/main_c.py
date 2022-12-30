import os
import json
from collections import defaultdict
import numpy as np
import supervisely as sly
from supervisely.app.content import StateJson
from dotenv import load_dotenv
import torch
import sklearn.manifold
import sklearn.cluster
import sklearn.decomposition
import umap
import re
from matplotlib.colors import rgb2hex
from supervisely.app.widgets import (
    ScatterChart,
    Container,
    Card,
    LabeledImage,
    Text,
    Table,
    Select,
    InputNumber,
    Checkbox,
    Button,
    Field,
    Progress,
    Input,
    ProjectSelector,
)

import src.run_utils as run_utils
import src.calculate_embeddings as calculate_embeddings


def normalize_string(s):
    return re.sub("[^A-Z0-9_()-]", "", s, flags=re.IGNORECASE)


def list2items(values):
    return [Select.Item(x) for x in values]


### Globals init
available_projection_methods = ["UMAP", "PCA", "t-SNE", "PCA-UMAP", "PCA-t-SNE"]
model_name = "facebook/convnext-tiny-224"
umap_min_dist = 0.05

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

# if app has started from context menu, one of this has to be set:
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)

if project_id is not None:
    is_one_dataset_mode = False
    datasets = api.dataset.get_list(project_id)
elif dataset_id is not None:
    is_one_dataset_mode = True
    datasets = [api.dataset.get_info_by_id(dataset_id)]
    project_id = datasets[0].project_id

if project_id is not None:
    project = api.project.get_info_by_id(project_id)
    workspace = api.workspace.get_info_by_id(project.workspace_id)
    team_id = workspace.team_id
    workspace_id = workspace.id
else:
    team_id = None
    workspace_id = None


def run():
    global model_name, global_idxs_mapping, all_info_list, project_meta

    card_embeddings_chart.hide()

    # 1. Read fields
    team_id = int(project_selector.get_selected_team_id(StateJson()))
    project_id = int(project_selector.get_selected_project_id(StateJson()))
    datasets = project_selector.get_selected_datasets(StateJson())
    instance_mode = str(select_instance_mode._content.get_value())
    expand_hw = [int(input_expand_wh._content.value)] * 2
    projection_method = str(select_instance_mode._content.get_value())
    metric = str(select_metric._content.get_value())
    force_recalculate = False  # legacy
    device = str(select_device._content.get_value())
    batch_size = int(input_batch_size._content.value)

    project = api.project.get_info_by_id(project_id)
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    if len(datasets) == 0:
        datasets = api.dataset.get_list(project_id)
    print(f"Selected datasets: {datasets}")

    save_name = model_name.replace("/", "_")
    path_prefix = f"embeddings/{normalize_string(project.name)}_{project_id}"
    save_paths = {
        "info": f"{path_prefix}/{save_name}_info.json",
        "cfg": f"{path_prefix}/{save_name}_cfg.json",
        "embeddings": f"{path_prefix}/{save_name}_embeddings.pt",
        "projections": f"{path_prefix}/{save_name}_projections_{projection_method}_{metric}.pt",
    }

    # TODO: we also have to recaclulate if instance_mode / expand_hw was changed

    # 2. Load embeddings if exist
    if api.file.exists(team_id, "/" + save_paths["info"]):
        embeddings, all_info, cfg = run_utils.download_embeddings(
            api, path_prefix, save_paths, team_id
        )
        print("embeddings loaded. n =", len(embeddings))
    else:
        embeddings = None
        all_info = None

    # 3. Calculate or upadate embeddings
    embeddings, all_info, cfg, is_updated = calculate_embeddings.calculate_embeddings(
        api,
        model_name,
        datasets,
        device,
        batch_size,
        embeddings,
        all_info,
        instance_mode,
        expand_hw,
        project_meta,
        progress,
    )

    # 4. Save embeddings if was updated
    if is_updated:
        print("uploading to team_files...")
        run_utils.upload_embeddings(
            embeddings, all_info, cfg, api, path_prefix, save_paths, team_id
        )

    # 5. Calculate projections or load from team_files
    force_recalculate = is_updated
    all_info_list = [
        dict(tuple(zip(all_info.keys(), vals))) for vals in zip(*list(all_info.values()))
    ]
    if api.file.exists(team_id, "/" + save_paths["projections"]) and not force_recalculate:
        api.file.download(team_id, "/" + save_paths["projections"], save_paths["projections"])
        projections = torch.load(save_paths["projections"])
    else:
        print("calculating projections...")
        projections = run_utils.calculate_projections(
            embeddings, all_info_list, projection_method, metric=metric, umap_min_dist=umap_min_dist
        )
        print("uploading projections to team_files...")
        torch.save(projections, save_paths["projections"])
        api.file.upload(team_id, save_paths["projections"], save_paths["projections"])

    obj_classes = list(set(all_info["object_cls"]))
    print(f"n_classes = {len(obj_classes)}")

    # 6. Show chart
    series, colors, global_idxs_mapping = run_utils.make_series(
        projections, all_info_list, project_meta
    )
