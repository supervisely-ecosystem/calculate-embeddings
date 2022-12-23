import os
import json
import numpy as np
import supervisely as sly
from dotenv import load_dotenv
import cv2
import torch
from matplotlib import pyplot as plt
import sklearn.cluster
from sklearn.neighbors import KDTree
import scipy.spatial
import sklearn.decomposition
import umap
from supervisely.app.widgets import Apexchart, Container, Card, Button, Progress, LabeledImage, Table, Text


s1 = [{"x": 12., "y": 66., "label":"aba1"}, {"x":35., "y":24., "label":"aba2", "fillColor":"#050205"}, {"x": 22., "y": 46.3, "label":"aba1"}]


# apexchart = Apexchart(
#     series=[{"name": "y", "data": s1}],
#     options={
#         "chart": {
#             "type": "scatter",
#             "zoom": {"enabled": True, 'type': 'xy'},
#             # "width": "50%"
#             },
#         "dataLabels": {"enabled": False},
#         # "markers": {
#         #     "colors": colors
#         # },
#         # "stroke": {"curve": "smooth", "width": 2},
#         "title": {"text": "Data Embeddings", "align": "left"},
#         # "grid": {"row": {"colors": ["#f3f3f3", "transparent"], "opacity": 0.5}},
#         "xaxis": {"type": "numeric"},
#         "yaxis": {"type": "numeric", "decimalsInFloat": 2},
#         "labels": all_info['object_cls']
#     },
#     type="scatter",
#     height="400",
# )
apexchart = Apexchart(
    series=[{'name':'ab', 'data': s1}],
    options = {
        "chart": {
            "type": 'scatter',
            "redrawOnParentResize": False
        },
        "dataLabels": {"enabled": False},
        "labels": ['Apple', 'Mango'],
        "xaxis": {"type": "numeric"},
        "yaxis": {"type": "numeric", "decimalsInFloat": 2},
        "fill": {
            "colors": ['#1A73E8', '#B32824', "#aaaaaa"]
        }
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