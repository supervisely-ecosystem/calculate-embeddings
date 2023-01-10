import os
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
    SelectString,
    InputNumber,
    Input,
    Checkbox,
    Button,
    Field,
    Progress,
    SelectDataset,
    NotificationBox,
)

from . import run_utils
from . import calculate_embeddings


def update_globals(new_dataset_ids):
    global dataset_ids, project_id, workspace_id, team_id, project_info, project_meta
    dataset_ids = new_dataset_ids
    if dataset_ids:
        project_id = api.dataset.get_info_by_id(dataset_ids[0]).project_id
        workspace_id = api.project.get_info_by_id(project_id).workspace_id
        team_id = api.workspace.get_info_by_id(workspace_id).team_id
        project_info = api.project.get_info_by_id(project_id)
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        print(f"Project is {project_info.name}, {dataset_ids}")
    elif project_id:
        workspace_id = api.project.get_info_by_id(project_id).workspace_id
        team_id = api.workspace.get_info_by_id(workspace_id).team_id
        project_info = api.project.get_info_by_id(project_id)
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    else:
        print("All globals set to None")
        dataset_ids = []
        project_id, workspace_id, team_id, project_info, project_meta = [None] * 5


### Globals init
available_projection_methods = ["UMAP", "PCA", "t-SNE", "PCA-UMAP", "PCA-t-SNE"]

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()

# if app had started from context menu, one of this has to be set:
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)
dataset_ids = [dataset_id] if dataset_id else []
update_globals(dataset_ids)


### Dataset selection
dataset_selector = SelectDataset(project_id=project_id, multiselect=True)
card_project_settings = Card(title="Dataset selection", content=dataset_selector)

### Model selection
model_items = [
    ["facebook/convnext-tiny-224", "114 MB", "ConvNet"],
    ["facebook/convnext-large-384", "791 MB", "ConvNet"],
    ["facebook/convnext-xlarge-224-22k", "1570 MB", "ConvNet"],
    ["openai/clip-vit-base-patch32", "605 MB", "Transformer"],
    ["openai/clip-vit-large-patch14", "1710 MB", "Transformer"],
    ["facebook/flava-full", "1430 MB", "Transformer"],
    ["microsoft/beit-large-patch16-224-pt22k", "1250 MB", "Transformer"],
    ["microsoft/beit-large-patch16-384", "1280 MB", "Transformer"],
    ["beitv2_large_patch16_224", "1310 MB", "Transformer"],
    ["beitv2_large_patch16_224_in22k", "1310 MB", "Transformer"],
    # ["maxvit_large_tf_384.in21k_ft_in1k", "849 MB", "ConvNet+Transformer"],  # now it is at pre-release in timm lib
]
files_list = api.file.list(team_id, "/embeddings")
rows = run_utils.get_rows(files_list, model_items, project_info)
column_names = ["Name", "Model size", "Architecture type", "Already calculated"]
table_model_select = RadioTable(column_names, rows)
table_model_select_f = Field(table_model_select, "Click on the table to select a model:")
input_select_model = Input("", placeholder="timm/vit_base_patch16_clip_224.openai")
desc_select_model = Text(
    "...or you can type a model_name from <a href='https://huggingface.co/models?sort=downloads&search=timm%2F'>timm</a>",
)
device_names, torch_devices = run_utils.get_devices()
select_device = Select([Select.Item(v, l) for v, l in zip(torch_devices, device_names)])
select_device_f = Field(select_device, "Device")
input_batch_size = InputNumber(2, 1, 10000)
input_batch_size_f = Field(
    input_batch_size,
    "Batch size",
)
content = Container(
    [
        table_model_select_f,
        desc_select_model,
        input_select_model,
        select_device_f,
        input_batch_size_f,
    ]
)
card_model_selection = Card(title="Model selection", content=content)


### Preprocessing settings
select_instance_mode = SelectString(
    [
        "objects",
        "images",
        "both",
    ]
)
select_instance_mode_f = Field(
    select_instance_mode,
    "Instance mode",
    "Whether to run for images or for cropped objects in the images or both",
)
input_expand_wh = InputNumber(0, -10000, 10000)
input_expand_wh_f = Field(
    input_expand_wh,
    "Expand crops (px)",
    "Expand rectangles of the cropped objects by a few pixels on both XY sides. Used to give the model a little context on the boundary of the objects.",
)
content = Container([select_instance_mode_f, input_expand_wh_f])
card_preprocessing_settings = Card(title="Preprocessing settings", content=content, collapsable=True)
card_preprocessing_settings.collapse()

### Visualizer settings
select_projection_method = SelectString(available_projection_methods)
select_projection_method_f = Field(
    select_projection_method,
    "Projection method",
    "A decomposition method: how to project the high-dimensional embeddings onto 2D space for further visualization.",
)
select_metric = SelectString(["euclidean", "cosine"])
select_metric_f = Field(select_metric, "Metric", "The parameter for projection method")
content = Container([select_projection_method_f, select_metric_f])
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
)
card_chart = Card(content=chart)
labeled_image = LabeledImage()
text = Text("no object selected")
show_all_anns = False
cur_info = None
btn_toggle = Button(f"Show all annotations: {show_all_anns}", "default", button_size="small")
card_preview = Card(title="Object preview", content=Container(widgets=[labeled_image, text, btn_toggle]))
card_embeddings_chart = Container(widgets=[card_chart, card_preview], direction="horizontal", fractions=[3, 1])
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


@btn_toggle.click
def toggle_ann():
    global show_all_anns
    show_all_anns = not show_all_anns
    btn_toggle.text = f"Show all annotations: {show_all_anns}"
    if cur_info:
        show_image(cur_info, project_meta)


@chart.click
def on_click(datapoint: ScatterChart.ClickedDataPoint):
    global global_idxs_mapping, all_info_list, project_meta
    idx = global_idxs_mapping[datapoint.series_name][datapoint.data_index]
    info = all_info_list[idx]
    print(datapoint.data_index, idx, info["image_id"], info["object_cls"], show_all_anns)
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
    ann = sly.Annotation.from_json(ann_json, project_meta) if len(ann_json["objects"]) else None

    labeled_image.set(title=image.name, image_url=image.preview_url, ann=ann, image_id=image_id)
    text.set("object class: " + str(obj_cls), "info")
    labeled_image.loading = False


@dataset_selector.value_changed
def on_dataset_selected(new_dataset_ids):
    update_globals(new_dataset_ids)
    update_table()


def update_table():
    files_list = api.file.list(team_id, "/embeddings")
    rows = run_utils.get_rows(files_list, model_items, project_info)
    table_model_select.rows = rows


@btn_run.click
def run():
    global model_name, global_idxs_mapping, all_info_list  # , project_meta, dataset_ids, project_id, workspace_id, team_id
    info_run.description = ""
    card_embeddings_chart.hide()

    if not dataset_ids:
        info_run.description += "Dataset is not selected"
        return

    # 1. Read fields
    datasets = [api.dataset.get_info_by_id(i) for i in dataset_ids]
    if input_select_model.get_value():
        model_name = input_select_model.get_value()
    else:
        model_name = table_model_select.get_selected_row(StateJson())[0]
    instance_mode = str(select_instance_mode.get_value())
    expand_hw = [int(input_expand_wh.value)] * 2
    projection_method = str(select_projection_method.get_value())
    metric = str(select_metric.get_value())
    device = str(select_device.get_value())
    batch_size = int(input_batch_size.value)
    force_recalculate = bool(check_force_recalculate.is_checked())
    path_prefix, save_paths = run_utils.get_save_paths(model_name, project_info, projection_method, metric)

    # 2. Load embeddings if exist
    if api.file.exists(team_id, "/" + save_paths["info"]) and not force_recalculate:
        info_run.description += "found existing embeddings<br>"
        embeddings, all_info, cfg = run_utils.download_embeddings(api, path_prefix, save_paths, team_id)
        print("embeddings downloaded. n =", len(embeddings))
    else:
        embeddings, all_info, cfg = None, None, None

    # 3. Calculate or update embeddings
    out = calculate_embeddings.calculate_embeddings_if_needed(
        api,
        model_name,
        datasets,
        device,
        batch_size,
        embeddings,
        all_info,
        cfg,
        instance_mode,
        expand_hw,
        project_meta,
        progress,
        info_run,
    )
    is_updated = out[-1]
    if is_updated:
        embeddings, all_info, cfg = out[:3]

    # 4. Save embeddings if it was updated
    is_updated = is_updated or force_recalculate
    if is_updated:
        print("uploading embeddings to team_files...")
        run_utils.upload_embeddings(embeddings, all_info, cfg, api, path_prefix, save_paths, team_id)

    # 5. Calculate projections or load from team_files
    all_info_list = [dict(tuple(zip(all_info.keys(), vals))) for vals in zip(*list(all_info.values()))]
    if api.file.exists(team_id, "/" + save_paths["projections"]) and not is_updated:
        info_run.description += "found existing projections<br>"
        print("downloading projections...")
        api.file.download(team_id, "/" + save_paths["projections"], save_paths["projections"])
        projections = torch.load(save_paths["projections"])
    else:
        info_run.description += "calculating projections...<br>"
        print("calculating projections...")
        if len(embeddings) <= 1:
            info_run.description += f"the count of embeddings (n={len(embeddings)}) must be > 1<br>"
            return
        try:
            projections = run_utils.calculate_projections(embeddings, all_info_list, projection_method, metric=metric)
        except RuntimeError:
            info_run.description += f"the count of embeddings is {len(embeddings)}, it may too small to project with UMAP, trying PCA...<br>"
            projection_method = "PCA"
            projections = run_utils.calculate_projections(embeddings, all_info_list, projection_method, metric=metric)
        print("uploading projections to team_files...")
        torch.save(projections, save_paths["projections"])
        api.file.upload(team_id, save_paths["projections"], save_paths["projections"])
    file_id = str(api.file.get_info_by_path(team_id, "/" + save_paths["embeddings"]).id)
    server_address = os.environ.get("SERVER_ADDRESS")
    if server_address:
        url = f"{server_address}/files/{file_id}"
        info_run.description += f"all was saved to team_files: <a href={url}>{save_paths['embeddings']}</a><br>"

    # 6. Show chart
    obj_classes = list(set(all_info["object_cls"]))
    print(f"n_classes = {len(obj_classes)}")
    series, colors, global_idxs_mapping = run_utils.make_series(projections, all_info_list, project_meta)
    chart.set_title(f"{model_name} {project_info.name} {projection_method} embeddings", send_changes=False)
    chart.set_colors(colors, send_changes=False)
    chart.set_series(series, send_changes=True)
    card_embeddings_chart.show()
    update_table()
    info_run.description += "done!<br>"
