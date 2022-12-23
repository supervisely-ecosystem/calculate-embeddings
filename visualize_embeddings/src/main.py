import os
import json
import numpy as np
import supervisely as sly
from dotenv import load_dotenv
import torch
import sklearn.cluster
import sklearn.decomposition
import umap
from supervisely.app.widgets import Apexchart, Container, Card, Button, Progress, LabeledImage, Table, Text

from .src import color_names


projection_method = 'PCA-UMAP'  # ['PCA', 'UMAP', 't-SNE', 'PCA-UMAP']
metric = 'euclidean'  # ['euclidean', 'cosine']

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

project_id = sly.env.project_id()
project = api.project.get_info_by_id(project_id)
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
team_id = sly.env.team_id(raise_not_found=False) or api.team.get_list()[0].id

model_name = 'facebook/convnext-tiny-224'
save_name = model_name.replace('/','_')

# load embeddings if exists
path_prefix = f'embeddings/{project_id}'
save_paths = {
    'info': f"{path_prefix}/{save_name}_info.json",
    'cfg': f"{path_prefix}/{save_name}_cfg.json",
    'embeddings': f"{path_prefix}/{save_name}_embeddings.pt",
    'projections': f"{path_prefix}/{save_name}_{projection_method}_projections.pt"
}
os.makedirs(path_prefix, exist_ok=True)
if api.file.exists(team_id, '/'+save_paths['info']):
    api.file.download(team_id, '/'+save_paths['info'], save_paths['info'])
    api.file.download(team_id, '/'+save_paths['embeddings'], save_paths['embeddings'])
    api.file.download(team_id, '/'+save_paths['cfg'], save_paths['cfg'])
    with open(save_paths['info'], 'r') as f:
        all_info = json.load(f)
    with open(save_paths['cfg'], 'r') as f:
        cfg = json.load(f)
    embeddings = torch.load(save_paths['embeddings'])
    print('embeddings loaded. n =', len(embeddings))
else:
    raise FileNotFoundError('/'+save_paths['info'])

all_info_list = [dict(tuple(zip(all_info.keys(), vals))) for vals in zip(*list(all_info.values()))]

if api.file.exists(team_id, '/'+save_paths['projections']):
    api.file.download(team_id, '/'+save_paths['projections'], save_paths['projections'])
    projections = torch.load(save_paths['projections'])
else:
    print('calculating projections...')
    if projection_method == 'PCA':
        decomp = sklearn.decomposition.PCA(2)
        projections = decomp.fit_transform(embeddings)
    elif projection_method == 'UMAP':
        decomp = umap.UMAP(min_dist=0.01, metric=metric)
        projections = decomp.fit_transform(embeddings)
    elif projection_method == 'PCA-UMAP':
        decomp = sklearn.decomposition.PCA(64)
        projections = decomp.fit_transform(embeddings)
        decomp = umap.UMAP(min_dist=0.01, metric=metric)
        projections = decomp.fit_transform(projections)
    torch.save(projections, save_paths['projections'])
    print('uploading projections to team_files...')
    api.file.upload(team_id, save_paths['projections'], save_paths['projections'])


colors = list(color_names.color_names.values())
to_color = dict(zip(list(set(all_info["object_cls"])), colors))
x = projections[:,1].tolist()
y = projections[:,0].tolist()
s1 = [{"x": x_i, "y": y_i, "fillColor": to_color[all_info_list[i]["object_cls"]]} for i, (x_i, y_i) in enumerate(zip(x, y))]

apexchart = Apexchart(
    series=[{"name": "y", "data": s1}],
    options={
        "chart": {
            "type": "scatter",
            "zoom": {"enabled": True, 'type': 'xy'},
            "redrawOnParentResize": False,
            },
        "title": {"text": "Data Embeddings", "align": "left"},
        "xaxis": {"type": "numeric"},
        "yaxis": {"type": "numeric", "decimalsInFloat": 2},
    },
    type="scatter",
    height="400",
)


card = Card(title="Apexchart", content=apexchart)
labeled_image = LabeledImage()
text = Text("no object selected")
preview_card = Card(title="Object preview", content=Container(widgets=[labeled_image, text]))
app = sly.Application(layout=Container(widgets=[card, preview_card], direction="horizontal", fractions=[3, 1]))

@apexchart.click
def on_click(datapoint: Apexchart.ClickedDataPoint):
    idx = datapoint.data_index
    print(idx, datapoint.x, datapoint.y)
    print(all_info_list[idx]['image_id'], all_info_list[idx]['object_cls'])
    show_image(all_info_list[idx]['image_id'], all_info_list[idx]['object_cls'])

def show_image(image_id, obj_cls):
    labeled_image.loading = True
    # image_id = datapoint.row["id"]
    image = api.image.get_info_by_id(image_id)
    ann_json = api.annotation.download_json(image_id)
    ann = sly.Annotation.from_json(ann_json, project_meta)
    labeled_image.set(title=image.name, image_url=image.preview_url, ann=ann, image_id=image_id)
    labeled_image.loading = False
    text.set('object class: '+str(obj_cls), 'text')
    print('OK')
