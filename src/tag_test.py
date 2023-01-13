import os
from dotenv import load_dotenv
import supervisely as sly


def create_tag_meta(project_id, tag_meta):
    # params: project_id
    # updates: global project_meta, tag_meta
    project_meta_json = api.project.get_meta(id=project_id)
    project_meta = sly.ProjectMeta.from_json(data=project_meta_json)
    project_meta = project_meta.add_tag_meta(new_tag_meta=tag_meta)
    api.project.update_meta(id=project_id, meta=project_meta)
    tag_meta = get_tag_meta(project_id, name=tag_meta.name)  # we need to re-assign tag_meta
    return project_meta, tag_meta


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


def read_tag():
    pass


def add_img_tag(image_id, tag_meta, value=None):
    return api.image.add_tag(image_id=image_id, tag_id=tag_meta.sly_id, value=value)


def add_label_tag(object_id, tag_meta, value=None):
    return api.advanced.add_tag_to_object(tag_meta_id=tag_meta.sly_id, figure_id=object_id, value=value)


def add_tag():
    pass


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


def remove_tag():
    pass


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

project_id = 16080
dataset_id = 54254

tag_meta = sly.TagMeta("marked", sly.TagValueType.NONE)
project_meta, tag_meta = get_or_create_tag_meta(project_id, tag_meta)
img_id = api.image.get_list(dataset_id)[0].id
remove_img_tag(img_id, tag_meta)
add_img_tag(img_id, tag_meta)
read_img_tag(img_id, tag_meta)
remove_img_tag(img_id, tag_meta)

ann = api.annotation.download_json(img_id)
obj_id = ann["objects"][0]["id"]
remove_label_tag(obj_id, tag_meta)
add_label_tag(obj_id, tag_meta)
read_label_tag(obj_id, tag_meta)
remove_label_tag(obj_id, tag_meta)
