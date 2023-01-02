import os, sys
import json
from collections import defaultdict
import numpy as np
import supervisely as sly
from supervisely.app.content import StateJson, DataJson
from dotenv import load_dotenv
import torch
import re
from supervisely.app.widgets import (
    ScatterChart,
    Container,
    Card,
    LabeledImage,
    Text,
    RadioTable,
    Select,
    InputNumber,
    Input,
    Checkbox,
    Button,
    Field,
    Progress,
    ProjectSelector,
    NotificationBox,
)

from . import run_utils
from . import calculate_embeddings


def normalize_string(s):
    return re.sub("[^A-Z0-9_()-]", "", s, flags=re.IGNORECASE)


def list2items(values):
    return [Select.Item(x) for x in values]


def get_save_paths(model_name, project, projection_method=None, metric=None):
    # TODO: separate folder for every project/model
    save_name = model_name.replace("/", "_")
    path_prefix = f"embeddings/{normalize_string(project.name)}_{project.id}/{save_name}"
    save_paths = {
        "info": f"{path_prefix}/info.json",
        "cfg": f"{path_prefix}/cfg.json",
        "embeddings": f"{path_prefix}/embeddings.pt",
        "projections": f"{path_prefix}/projections_{projection_method}_{metric}.pt",
    }
    return path_prefix, save_paths


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


### Project selection
project_selector = ProjectSelector(team_id, workspace_id, project_id)
if is_one_dataset_mode:
    state = StateJson()[project_selector.widget_id]
    state["datasetsIds"] = datasets[0].id
    state["allDatasets"] = False
card_project_settings = Card(title="Project selection", content=project_selector)


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


### Model selection
columns_names = ["Name", "Model size", "Architecture type"]
items = [
    ["openai/clip-vit-base-patch32", "605 MB", "Transformer"],
    ["openai/clip-vit-large-patch14", "1710 MB", "Transformer"],
    ["facebook/convnext-tiny-224", "114 MB", "ConvNet"],
    ["facebook/convnext-large-384", "791 MB", "ConvNet"],
    ["facebook/convnext-xlarge-224-22k", "1570 MB", "ConvNet"],
    ["facebook/flava-full", "1430 MB", "Tramsformer"],
    ["microsoft/beit-large-patch16-224-pt22k", "1250 MB", "Tramsformer"],
    ["microsoft/beit-large-patch16-384", "1280 MB", "Tramsformer"],
    ["beitv2_large_patch16_224", "1310 MB", "Tramsformer"],
    ["beitv2_large_patch16_224_in22k", "1310 MB", "Tramsformer"],
    # ["maxvit_large_tf_384.in21k_ft_in1k", "849 MB", "ConvNet+Transformer"],  # now it is at pre-release in timm lib
]
# data = {col: v for col, v in zip(columns_names, zip(*items))}
table_model_select = RadioTable(columns_names, items)
table_model_select_f = Field(table_model_select, "Click on the table to select a model:")
input_select_model = Input("", placeholder="openai/clip-vit-base-patch32")
desc_select_model = Text(
    "...or type a model_name to download the model from <a href='https://huggingface.co/models'>HuggingFace</a> or <a href='https://huggingface.co/models?sort=downloads&search=timm%2F'>timm</a>",
)
cuda_names = [
    f"cuda:{i} ({torch.cuda.get_device_name(i)})" for i in range(torch.cuda.device_count())
]
cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
labels = cuda_names + ["cpu"]
values = cuda_devices + ["cpu"]
select_device = Field(Select([Select.Item(v, l) for v, l in zip(values, labels)]), "Device")
input_batch_size = Field(
    InputNumber(2, 1, 10000),
    "Batch size",
)
content = Container(
    [table_model_select_f, desc_select_model, input_select_model, select_device, input_batch_size]
)
card_model_selection = Card(title="Model selection", content=content)


### Preprocessing settings
select_instance_mode = Field(
    Select(
        list2items(
            [
                "objects",
                "images",
                "both",
            ]
        )
    ),
    "Instance mode",
    "Whether to run for images or for cropped objects in the images or both",
)
input_expand_wh = Field(
    InputNumber(0, -10000, 10000),
    "Expand crops (px)",
    "Expand rectangles of the cropped objects by a few pixels on both XY sides. Used to give the model a little context on the boundary of an objects.",
)
content = Container([select_instance_mode, input_expand_wh])
card_preprocessing_settings = Card(
    title="Preprocessing settings", content=content, collapsable=True
)
card_preprocessing_settings.collapse()

### Visualizer settings
select_projection_method = Field(
    Select(list2items(available_projection_methods)),
    "Projection method",
    "How to project high-dimensional embeddings into 2D space for futher visualizaton in chart",
)
select_metric = Field(
    Select(list2items(["euclidean", "cosine"])), "Metric", "The parameter for projection method"
)
content = Container([select_projection_method, select_metric])
card_visualizer_settings = Card(title="Visualizer settings", content=content, collapsable=True)
card_visualizer_settings.collapse()


### Run section
btn_run = Button("Run")
check_force_recalculate = Checkbox("Force recalculate")
progress = Progress()
info_run = NotificationBox()
content = Container([btn_run, check_force_recalculate, progress, info_run])
card_run = Card(title="Run", content=content)


### Embeddings Chart
chart = ScatterChart(
    title=f"None",
    xaxis_type="numeric",
    height=600,
    # series=series,
    # colors=colors,
)
card_chart = Card(content=chart)
labeled_image = LabeledImage()
text = Text("no object selected")
btn_toggle = Button("Toggle annotations", "info")
show_all_anns = False
cur_info = None
card_preview = Card(
    title="Object preview", content=Container(widgets=[labeled_image, text, btn_toggle])
)
card_embeddings_chart = Container(
    widgets=[card_chart, card_preview], direction="horizontal", fractions=[3, 1]
)
# card_embeddings_chart = Card(title="Embeddings Chart", content=content)
card_embeddings_chart.hide()


app = sly.Application(
    layout=Container(
        widgets=[
            card_project_settings,
            card_model_selection,
            card_preprocessing_settings,
            card_visualizer_settings,
            card_run,
            card_embeddings_chart,
        ]
    )
)


# @table_model_select.click
# def table_on_click(item: Table.ClickedDataPoint):
#     global model_name
#     text_selected_model.text = f"Selected model: <b>{item.row['Name']}</b>"
#     project_id = int(project_selector.get_selected_project_id(StateJson()))
#     project = api.project.get_info_by_id(project_id)

#     model_name = item.row["Name"]

#     path_prefix, save_paths = get_save_paths(model_name, project)
#     team_id = int(project_selector.get_selected_team_id(StateJson()))
#     if api.file.exists(team_id, "/" + save_paths["cfg"]):
#         print(f"{model_name} exists!")
#         api.file.download(team_id, "/" + save_paths["cfg"], save_paths["cfg"])
#         with open(save_paths["cfg"], "r") as f:
#             cfg = json.load(f)
#         StateJson()[select_instance_mode._content.widget_id]["value"] = cfg["instance_mode"]
#         input_expand_wh._content.value = cfg["expand_hw"][0]
#         info_run.description = "found existing embeddings<br>"
#         # StateJson().send_changes()


@btn_toggle.click
def toggle_ann():
    global show_all_anns
    show_all_anns = not show_all_anns
    show_image(cur_info, project_meta)


@chart.click
def on_click(datapoint: ScatterChart.ClickedDataPoint):
    global global_idxs_mapping, all_info_list, project_meta
    idx = global_idxs_mapping[datapoint.series_name][datapoint.data_index]
    info = all_info_list[idx]
    print(datapoint.data_index, idx, info["image_id"], info["object_cls"])
    print(info["image_id"], info["object_cls"])
    show_image(info, project_meta)


def show_image(info, project_meta):
    global cur_info, show_all_anns
    cur_info = info
    image_id, obj_cls, obj_id = info["image_id"], info["object_cls"], info["object_id"]
    labeled_image.loading = True

    image = api.image.get_info_by_id(image_id)
    ann_json = api.annotation.download_json(image_id)
    if not show_all_anns:
        ann_json["objects"] = [obj for obj in ann_json["objects"] if obj["id"] == obj_id]
    ann = sly.Annotation.from_json(ann_json, project_meta)

    labeled_image.set(title=image.name, image_url=image.preview_url, ann=ann, image_id=image_id)
    text.set("object class: " + str(obj_cls), "info")
    labeled_image.loading = False


@btn_run.click
def run():
    global model_name, global_idxs_mapping, all_info_list, project_meta

    card_embeddings_chart.hide()
    info_run.description = ""

    # 1. Read fields
    team_id = int(project_selector.get_selected_team_id(StateJson()))
    project_id = int(project_selector.get_selected_project_id(StateJson()))
    datasets = project_selector.get_selected_datasets(StateJson())
    if input_select_model.get_value():
        model_name = input_select_model.get_value()
    else:
        model_name = table_model_select.get_selected_row(StateJson())[0]
    instance_mode = str(select_instance_mode._content.get_value())
    expand_hw = [int(input_expand_wh._content.value)] * 2
    projection_method = str(select_projection_method._content.get_value())
    metric = str(select_metric._content.get_value())
    device = str(select_device._content.get_value())
    batch_size = int(input_batch_size._content.value)
    force_recalculate = bool(check_force_recalculate.is_checked())

    project = api.project.get_info_by_id(project_id)
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    if len(datasets) == 0:
        datasets = api.dataset.get_list(project_id)

    path_prefix, save_paths = get_save_paths(model_name, project, projection_method, metric)

    # TODO: when to recalculate projections?

    # 2. Load embeddings if exist
    need_recalculate = force_recalculate
    if api.file.exists(team_id, "/" + save_paths["info"]) and not need_recalculate:
        info_run.description += "found existing embeddings<br>"
        embeddings, all_info, cfg = run_utils.download_embeddings(
            api, path_prefix, save_paths, team_id
        )
        print("embeddings downloaded. n =", len(embeddings))
        need_recalculate = (
            force_recalculate
            or cfg["instance_mode"] != instance_mode
            or cfg["expand_hw"] != expand_hw
        )
    else:
        embeddings = None
        all_info = None
    if need_recalculate:
        embeddings = None
        all_info = None
        info_run.description += "some parameters are force to recalculate the embeddings<br>"

    # 3. Calculate or upadate embeddings
    out = calculate_embeddings.calculate_embeddings_if_needed(
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
        info_run,
    )
    is_updated = out[-1]
    if is_updated:
        embeddings, all_info, cfg = out[:3]

    # 4. Save embeddings if was updated
    is_updated = is_updated or need_recalculate
    if is_updated:
        print("uploading embeddings to team_files...")
        run_utils.upload_embeddings(
            embeddings, all_info, cfg, api, path_prefix, save_paths, team_id
        )

    # 5. Calculate projections or load from team_files
    all_info_list = [
        dict(tuple(zip(all_info.keys(), vals))) for vals in zip(*list(all_info.values()))
    ]
    if api.file.exists(team_id, "/" + save_paths["projections"]) and not is_updated:
        info_run.description += "found existing projections<br>"
        print("downloading projections...")
        api.file.download(team_id, "/" + save_paths["projections"], save_paths["projections"])
        projections = torch.load(save_paths["projections"])
    else:
        info_run.description += "calculating projections...<br>"
        print("calculating projections...")
        projections = run_utils.calculate_projections(
            embeddings, all_info_list, projection_method, metric=metric, umap_min_dist=umap_min_dist
        )
        print("uploading projections to team_files...")
        torch.save(projections, save_paths["projections"])
        api.file.upload(team_id, save_paths["projections"], save_paths["projections"])
    file_id = str(api.file.get_info_by_path(team_id, "/" + save_paths["embeddings"]).id)
    server = os.environ.get("SERVER_ADDRESS")
    if server:
        url = f"{server}/files/{file_id}"
    else:
        url = ""
    info_run.description += (
        f"all was saved to team_files: <a href={url}>{save_paths['embeddings']}</a><br>"
    )

    obj_classes = list(set(all_info["object_cls"]))
    print(f"n_classes = {len(obj_classes)}")

    # 6. Show chart
    series, colors, global_idxs_mapping = run_utils.make_series(
        projections, all_info_list, project_meta
    )
    chart._options["title"]["text"] = f"{model_name} {project.name} {projection_method} embeddings"
    chart._options["colors"] = colors
    chart._series = series
    chart.update_data()
    # DataJson().send_changes()
    card_embeddings_chart.show()
