from time import sleep
import os, sys
import json
from collections import defaultdict
import numpy as np
import supervisely as sly
from supervisely.app.content import StateJson, DataJson
from dotenv import load_dotenv
import torch
import re
from supervisely.app.widgets import Progress, NotificationBox, Container

from . import calculate_embeddings


progress = Progress()
note = NotificationBox()
content = Container([progress, note])
app = sly.Application(layout=content)

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

device = "cpu"
batch_size = 2
embeddings, all_info = None, None
instance_mode = "both"
expand_hw = [0, 0]
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))

embeddings, info_updated, cfg, is_updated = calculate_embeddings.calculate_embeddings_if_needed(
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
)
