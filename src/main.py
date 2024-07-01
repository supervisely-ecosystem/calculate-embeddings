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
    global dataset_ids, project_id, workspace_id, team_id, project_info, project_meta, is_marked, tag_meta
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
    if dataset_ids or project_id:
        is_marked = False
        tag_meta = project_meta.get_tag_meta(tag_name)
        print("tag_meta is exists:", bool(tag_meta))


### Globals init
available_projection_methods = ["UMAP", "PCA", "t-SNE", "PCA-UMAP", "PCA-t-SNE"]
tag_name = "MARKED"

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
    "Expand bounding boxes by the given number of pixels on all sides (both X and Y axes). This helps provide the model with some context around the boundaries of the objects.",
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
btn_mark = Button(f"Assign tag 'MARKED'", button_size="small")
card_preview = Card(title="Object preview", content=Container(widgets=[labeled_image, text, btn_toggle, btn_mark]))
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
    global global_idxs_mapping, all_info_list, project_meta, is_marked, tag_meta
    idx = global_idxs_mapping[datapoint.series_name][datapoint.data_index]
    info = all_info_list[idx]
    if tag_meta is not None:
        tag = read_tag(info["image_id"], info["object_id"])
        is_marked = bool(tag)
        update_marked()
    show_image(info, project_meta)
    if btn_mark.is_hidden():
        btn_mark.show()


def update_marked():
    global is_marked
    if is_marked:
        btn_mark.text = "Remove tag 'MARKED'"
    else:
        btn_mark.text = "Assign tag 'MARKED'"


@btn_mark.click
def on_mark():
    global project_info, project_meta, tag_meta, cur_info, is_marked
    if tag_meta is None:
        print("first marking, creating tag_meta")
        tag_meta = sly.TagMeta(tag_name, sly.TagValueType.NONE)
        project_meta, tag_meta = get_or_create_tag_meta(project_id, tag_meta)
        is_marked = False
    img_id, obj_id = cur_info["image_id"], cur_info["object_id"]
    if is_marked:
        resp = remove_tag(img_id, obj_id)
    else:
        resp = add_tag(img_id, obj_id)
    tag = read_tag(img_id, obj_id)
    is_marked = bool(tag)
    update_marked()


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
    update_marked()


def update_table():
    files_list = api.file.list(team_id, "/embeddings")
    rows = run_utils.get_rows(files_list, model_items, project_info)
    table_model_select.rows = rows


@btn_run.click
def run():
    global model_name, global_idxs_mapping, all_info_list  # , project_meta, dataset_ids, project_id, workspace_id, team_id
    info_run.description = ""
    card_embeddings_chart.hide()
    btn_mark.hide()

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
        info_run.description += "Calculating projections...<br>"
        print("calculating projections...")
        if len(embeddings) <= 1:
            info_run.description += f"the count of embeddings (n={len(embeddings)}) must be > 1<br>"
            return
        try:
            projections = run_utils.calculate_projections(embeddings, all_info_list, projection_method, metric=metric)
        except RuntimeError:
            info_run.description += (
                f"the count of embeddings is {len(embeddings)}, not enough to use UMAP. Trying PCA instead...<br>"
            )
            projection_method = "PCA"
            projections = run_utils.calculate_projections(embeddings, all_info_list, projection_method, metric=metric)
        print("uploading projections to team_files...")
        torch.save(projections, save_paths["projections"])
        remote_path = f"/{save_paths['projections']}"
        api.file.upload(team_id, save_paths["projections"], remote_path)
    file_id = str(api.file.get_info_by_path(team_id, "/" + save_paths["embeddings"]).id)
    server_address = os.environ.get("SERVER_ADDRESS")
    if server_address:
        if sly.is_development():
            url = sly.utils.abs_url(f"files/{file_id}")
        else:
            url = f"/files/{file_id}"
        info_run.description += f"Embeddings were saved to Team Files: <a href={url}>{save_paths['embeddings']}</a><br>"

    # 6. Show chart
    obj_classes = list(set(all_info["object_cls"]))
    print(f"n_classes = {len(obj_classes)}")
    series, colors, global_idxs_mapping = run_utils.make_series(projections, all_info_list, project_meta)
    chart.set_title(f"{model_name} {project_info.name} {projection_method} embeddings", send_changes=False)
    chart.set_colors(colors, send_changes=False)
    chart.set_series(series, send_changes=True)
    card_embeddings_chart.show()
    update_table()
    info_run.description += "Done!<br>"


def get_or_create_tag_meta(project_id, tag_meta):
    # params: project_id
    # updates: global project_meta, tag_meta
    project_meta_json = api.project.get_meta(id=project_id)
    project_meta = sly.ProjectMeta.from_json(data=project_meta_json)
    tag_names = [tag_meta.name for tag_meta in project_meta.tag_metas]
    if tag_meta.name not in tag_names:
        project_meta = project_meta.add_tag_meta(new_tag_meta=tag_meta)
        api.project.update_meta(id=project_id, meta=project_meta)
    tag_meta = get_tag_meta(project_id, name=tag_meta.name)  # we need to re-assign tag_meta
    return project_meta, tag_meta


def get_tag_meta(project_id, name) -> sly.TagMeta:
    project_meta = api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(project_meta)
    return project_meta.get_tag_meta(name)


def read_img_tag(image_id, tag_meta):
    image_info = api.image.get_info_by_id(image_id)
    tags = [tag for tag in image_info.tags if tag["tagId"] == tag_meta.sly_id]
    if len(tags) == 1:
        return tags[0]


def read_label_tag(object_id, tag_meta):
    tags = api.advanced.get_object_tags(object_id)
    tags_filtered = [tag for tag in tags if tag["tagId"] == tag_meta.sly_id]
    if len(tags_filtered) == 1:
        return tags_filtered[0]


def read_tag(image_id, object_id):
    if object_id is None:
        # it is an image
        return read_img_tag(image_id, tag_meta)
    else:
        # it is an object
        return read_label_tag(object_id, tag_meta)


def add_img_tag(image_id, tag_meta, value=None):
    return api.image.add_tag(image_id=image_id, tag_id=tag_meta.sly_id, value=value)


def add_label_tag(object_id, tag_meta, value=None):
    return api.advanced.add_tag_to_object(tag_meta_id=tag_meta.sly_id, figure_id=object_id, value=value)


def add_tag(image_id, object_id):
    if object_id is None:
        # it is an image
        return add_img_tag(image_id, tag_meta)
    else:
        # it is an object
        return add_label_tag(object_id, tag_meta)


def remove_img_tag(image_id, tag_meta):
    tag = read_img_tag(image_id, tag_meta)
    if tag:
        tag_id = tag["id"]
        return api.advanced.remove_tag_from_image(tag_meta_id=tag_meta.sly_id, image_id=image_id, tag_id=tag_id)
    else:
        return False


def remove_label_tag(object_id, tag_meta):
    tag = read_label_tag(object_id, tag_meta)
    if tag:
        tag_id = tag["id"]
        return api.advanced.remove_tag_from_object(tag_meta_id=tag_meta.sly_id, figure_id=object_id, tag_id=tag_id)
    else:
        return False


def remove_tag(image_id, object_id):
    if object_id is None:
        # it is an image
        return remove_img_tag(image_id, tag_meta)
    else:
        # it is an object
        return remove_label_tag(object_id, tag_meta)
