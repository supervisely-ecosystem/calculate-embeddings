import os
from typing import List, Optional
from dotenv import load_dotenv
from dataclasses import dataclass, field

import supervisely as sly


@dataclass
class GlobalParams:
    api: sly.Api
    dataset_ids: List[int] = field(default_factory=list)
    project_id: Optional[int] = None
    workspace_id: Optional[int] = None
    team_id: Optional[int] = None
    project_info: Optional[sly.ProjectInfo] = None
    project_meta: Optional[sly.ProjectMeta] = None
    is_marked: bool = False
    tag_meta: Optional[sly.TagMeta] = None


tag_name = "MARKED"


def update_globals(new_dataset_ids: List[int], params: GlobalParams):
    params.dataset_ids = new_dataset_ids
    if len(params.dataset_ids) > 0:
        params.project_id = api.dataset.get_info_by_id(params.dataset_ids[0]).project_id
        params.workspace_id = api.project.get_info_by_id(params.project_id).workspace_id
        params.team_id = api.workspace.get_info_by_id(params.workspace_id).team_id
        params.project_info = api.project.get_info_by_id(params.project_id)
        params.project_meta = sly.ProjectMeta.from_json(api.project.get_meta(params.project_id))
        print(f"Project is {params.project_info.name}, {params.dataset_ids}")
    elif params.project_id:
        params.workspace_id = api.project.get_info_by_id(params.project_id).workspace_id
        params.team_id = api.workspace.get_info_by_id(params.workspace_id).team_id
        params.project_info = api.project.get_info_by_id(params.project_id)
        params.project_meta = sly.ProjectMeta.from_json(api.project.get_meta(params.project_id))
    else:
        print("All globals set to None")
        params.dataset_ids = []
        params.project_id, params.workspace_id, params.team_id, params.project_info, params.project_meta = [None] * 5
    if params.dataset_ids or params.project_id:
        params.is_marked = False
        params.tag_meta = params.project_meta.get_tag_meta(tag_name)
        print("tag_meta is exists:", bool(params.tag_meta))


### Globals init
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()
params: GlobalParams = GlobalParams(api)

# if app had started from context menu, one of this has to be set:
params.project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)
params.dataset_ids = [dataset_id] if dataset_id else []

if len(params.dataset_ids) == 0 and params.project_id is None:
    raise KeyError(
            (
                f"PROJECT_ID and DATASET_ID are not defined as environment variable. One of the envs has to be defined."
                "Learn more in developer portal: https://developer.supervise.ly/getting-started/environment-variables"
            )
        )

update_globals(params.dataset_ids, params)