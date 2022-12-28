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
import matplotlib.cm
from supervisely.app.widgets import ScatterChart, Container, Card, LabeledImage, Text


def calculate_projections(projection_method, metric="euclidean", umap_min_dist=0.1):
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
            2, perplexity=min(30, len(all_info_list)), metric=metric, n_jobs=-1
        )
        projections = decomp.fit_transform(embeddings)
    elif projection_method == "PCA-t-SNE":
        decomp = sklearn.decomposition.PCA(64)
        projections = decomp.fit_transform(embeddings)
        decomp = sklearn.manifold.TSNE(
            2, perplexity=min(30, len(all_info_list)), metric=metric, n_jobs=-1
        )
        projections = decomp.fit_transform(projections)
    return projections


projection_method = "UMAP"  # ['PCA', 'UMAP', 't-SNE', 'PCA-UMAP', 'PCA-t-SNE']
umap_min_dist = 0.05
metric = "euclidean"  # ['euclidean', 'cosine']

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

project_id = sly.env.project_id()
project = api.project.get_info_by_id(project_id)
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
team_id = sly.env.team_id(raise_not_found=False) or api.team.get_list()[0].id
projection_method = os.environ.get("modal.state.projection_method")
if not projection_method:
    projection_method = "UMAP"
    print("cant't find projection_method, setting to default:", projection_method)

model_name = "facebook/convnext-tiny-224"
save_name = model_name.replace("/", "_")

# load embeddings if exists
path_prefix = f"embeddings/{project_id}"
save_paths = {
    "info": f"{path_prefix}/{save_name}_info.json",
    "cfg": f"{path_prefix}/{save_name}_cfg.json",
    "embeddings": f"{path_prefix}/{save_name}_embeddings.pt",
    "projections": f"{path_prefix}/{save_name}_{projection_method}_projections.pt",
}
os.makedirs(path_prefix, exist_ok=True)
if api.file.exists(team_id, "/" + save_paths["info"]):
    api.file.download(team_id, "/" + save_paths["info"], save_paths["info"])
    api.file.download(team_id, "/" + save_paths["embeddings"], save_paths["embeddings"])
    api.file.download(team_id, "/" + save_paths["cfg"], save_paths["cfg"])
    with open(save_paths["info"], "r") as f:
        all_info = json.load(f)
    with open(save_paths["cfg"], "r") as f:
        cfg = json.load(f)
    embeddings = torch.load(save_paths["embeddings"])
    print("embeddings loaded. n =", len(embeddings))
else:
    raise FileNotFoundError("/" + save_paths["info"])

all_info_list = [dict(tuple(zip(all_info.keys(), vals))) for vals in zip(*list(all_info.values()))]

if api.file.exists(team_id, "/" + save_paths["projections"]):
    api.file.download(team_id, "/" + save_paths["projections"], save_paths["projections"])
    projections = torch.load(save_paths["projections"])
else:
    print("calculating projections...")
    projections = calculate_projections(
        projection_method, metric=metric, umap_min_dist=umap_min_dist
    )
    torch.save(projections, save_paths["projections"])
    print("uploading projections to team_files...")
    api.file.upload(team_id, save_paths["projections"], save_paths["projections"])


obj_classes = list(set(all_info["object_cls"]))
cm = matplotlib.cm.get_cmap("gist_rainbow")
colors = [matplotlib.colors.rgb2hex(cm(x)) for x in np.linspace(0, 1, len(obj_classes))]
to_color = dict(zip(obj_classes, colors))
x = projections[:, 1].tolist()
y = projections[:, 0].tolist()
print(f"n_classes = {len(obj_classes)}")

series = defaultdict(list)
global_idxs_mapping = defaultdict(list)
for i in range(len(all_info_list)):
    obj_cls = str(all_info_list[i]["object_cls"])
    series[obj_cls].append({"x": x[i], "y": y[i]})
    global_idxs_mapping[obj_cls].append(i)

series = [{"name": k, "data": v} for k, v in series.items()]

chart = ScatterChart(title="Embeddings", series=series, xaxis_type="numeric", height=400)


card = Card(title="Embeddings Cloud", content=chart)
labeled_image = LabeledImage()
text = Text("no object selected")
preview_card = Card(title="Object preview", content=Container(widgets=[labeled_image, text]))
app = sly.Application(
    layout=Container(widgets=[card, preview_card], direction="horizontal", fractions=[3, 1])
)


@chart.click
def on_click(datapoint: ScatterChart.ClickedDataPoint):
    idx = global_idxs_mapping[datapoint.series_name][datapoint.data_index]
    info = all_info_list[idx]
    print(datapoint.data_index, idx, info["image_id"], info["object_cls"])
    print(info["image_id"], info["object_cls"])
    show_image(info)


def show_image(info):
    image_id, obj_cls, obj_id = info["image_id"], info["object_cls"], info["object_id"]
    labeled_image.loading = True

    image = api.image.get_info_by_id(image_id)
    ann_json = api.annotation.download_json(image_id)
    ann_json["objects"] = [obj for obj in ann_json["objects"] if obj["id"] == obj_id]
    ann = sly.Annotation.from_json(ann_json, project_meta)

    labeled_image.set(title=image.name, image_url=image.preview_url, ann=ann, image_id=image_id)
    text.set("object class: " + str(obj_cls), "text")
    labeled_image.loading = False
