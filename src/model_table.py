from typing import Optional

import supervisely as sly
from supervisely.app.content import StateJson
from supervisely.app.widgets import RadioTable, SelectString, Field, Container

from src import run_utils
from src.globals import params


other_model_items = [
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

clip_model_items = [
    ["ViT-L-14:openai", "0.933 GB", "openai"],
    ["coca_ViT-L-14:mscoco", "2.55 GB", "mscoco_finetuned_laion2B-s13B-b90k"],
    ["coca_ViT-L-14", "2.55 GB", "laion2B-s13B-b90k"],
    ["ViT-L-14", "0.933 GB", "laion2b_s32b_b82k"],
    ["ViT-L-14-336", "0.933 GB", "openai"],
    ["ViT-g-14", "5.47 GB", "laion2b_s34b_b88k"],
    ["ViT-bigG-14", "10.2 GB", "laion2b_s39b_b160k"],
    ["convnext_base_w", "0.718 GB", "laion2b_s13b_b82k_augreg"],
    ["convnext_large_d_320", "1.41 GB", "laion2b_s29b_b131k_ft_soup"]
]

clip_models_info = {
    "ViT-L-14:openai": {
        "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "path": "clip/ViT-L-14.pt",
    },
    "coca_ViT-L-14:mscoco": {
        "url": "https://huggingface.co/laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/resolve/main/open_clip_pytorch_model.bin",
        "path": "huggingface/hub/models--laion--mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/blobs/f22c34acef2b7a5d1ed28982a21077de651363eaaebcf34a3f10676e17837cb8",
    },
    "coca_ViT-L-14": {
        "url": "https://huggingface.co/laion/CoCa-ViT-L-14-laion2B-s13B-b90k/resolve/main/open_clip_pytorch_model.bin",
        "path": "huggingface/hub/models--laion--CoCa-ViT-L-14-laion2B-s13B-b90k/blobs/73725652298ad76ed2162caffdae96d8653a05d7a29b6281103e4df81d0ff8ea",
    },
    "ViT-L-14": {
        "url": "https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/main/open_clip_pytorch_model.bin",
        "path": "huggingface/hub/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/blobs/5ddb47339f44e4fd9cace3d3960d38af1b51a25857440cfae90afc44706d7e2b",
    },
    "ViT-L-14-336": {
        "url": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
        "path": "clip/ViT-L-14-336px.pt",
    },
    "ViT-g-14": {
        "url": "https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s34B-b88K/resolve/main/open_clip_pytorch_model.bin",
        "path": "huggingface/hub/models--laion--CLIP-ViT-g-14-laion2B-s34B-b88K/blobs/9ef136f407986fb607cd37a823eba38a3b6f95e8ec702b3d1687252985d84750",
    },
    "ViT-bigG-14": {
        "url": "https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/open_clip_pytorch_model.bin",
        "path": "huggingface/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/blobs/0d5318839ad03607c48055c45897c655a14c0276a79f6b867934ddd073760e39",
    },
    "convnext_base_w": {
        "url": "https://huggingface.co/laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg/resolve/main/open_clip_pytorch_model.bin",
        "path": "huggingface/hub/models--laion--CLIP-convnext_base_w-laion2B-s13B-b82K-augreg/blobs/249e2302c1670bb04476792196f788ff046fedef61191a24983e61b6eca56987",
    },
    "convnext_large_d_320": {
        "url": "https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/resolve/main/open_clip_pytorch_model.bin",
        "path": "huggingface/hub/models--laion--CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/blobs/4572137af44b2e26f01f638337a59688ec289e9363e15c08dde16640afb86988",
    },
}

files_list = params.api.file.list(params.team_id, "/embeddings")
rows = run_utils.get_rows(files_list, other_model_items, params.project_info)
column_names = ["Name", "Model size", "Architecture type", "Already calculated"]
column_clip_names = ["Name", "Model size", "Pretrained model", "Already calculated"]

model_table = RadioTable(column_names, rows)

model_type_selector = SelectString(["Clip models", "Other models"], filterable=True)
# TODO: Is this default?
model_type_selector.set_value("Other models")

model_table_container = Container([model_type_selector, model_table])
table_model_select_f = Field(model_table_container, "Click on the table to select a model:")

def update_table(selected_type: Optional[str] = None):
    if selected_type is None:
        selected_type = model_type_selector.get_value()
        print("Found type: ", selected_type)

    if selected_type == "Clip models":
        model_items = clip_model_items
        columns = column_clip_names
    else:
        model_items = other_model_items
        columns = column_names

    rows = run_utils.get_rows(files_list, model_items, params.project_info)
    # TODO: fix set_data and set_{smth} for RadioTable -- subtitles can be unnecessary(?)
    # model_table.set_data(columns, rows, subtitles=None)
    model_table.rows = rows


@model_type_selector.value_changed
def update_models_on_type_change(selected_type):
    update_table(selected_type)


def get_selected_model():
    model_name_info = model_table.get_selected_row(StateJson())
    name = model_name_info[0]
    if model_type_selector.get_value() == "Clip models":
        return name, clip_models_info[name]
    return name, None