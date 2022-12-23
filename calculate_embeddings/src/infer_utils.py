import cv2
import torch
import timm
import transformers
from PIL.Image import Resampling


to_cv2_interpolation = {
    Resampling.NEAREST: cv2.INTER_NEAREST,
    Resampling.BILINEAR: cv2.INTER_LINEAR,
    Resampling.BICUBIC: cv2.INTER_CUBIC,
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
}


# class SlyDatasetForEmbeddings(torch.utils.data.IterableDataset):
#     def __init__(self, api, project_id, instance_mode, batch_size, batch_size_api, instance_rect_expand, img_size_hw):
#         super().__init__()
#         self.api = api
#         self.project = self.api.project.get_info_by_id(project_id)
#         self.project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
#         # self.

#     def __iter__(self):
#         for dataset in self.api.dataset.get_list(self.project.id):
#             all_image_ids = self.api.image.get_list(dataset.id)
#             for batch in sly.batched(all_image_ids, batch_size=batch_size_api):
#                 image_ids = [image.id for image in batch]
#                 anns_json = self.api.annotation.download_json_batch(dataset.id, image_ids)

#                 images = self.api.image.download_nps(dataset.id, image_ids)
#                 annotations = [sly.Annotation.from_json(ann, self.project_meta) for ann in anns_json]
#                 assert len(images) == len(annotations)

#                 for image, ann in zip(images, annotations):
#                     crops = extract_crops(image, ann, target_img_size_hw=img_size_hw,
#                         instance_mode=instance_mode, hw_expand=instance_rect_expand)
#                     if len(crops) == 0:
#                         continue
#                     crops_batched = form_batches(crops, batch_size=batch_size)
#                     crops_batched = [batch.transpose(0,3,1,2) for batch in crops_batched]



def create_model(model_name):
    # simple detect if '/' in model_name
    if '/' in model_name:
        return create_hf_model(model_name)
    else:
        return create_timm_model(model_name)


def create_hf_model(model_name):
    model = transformers.AutoModel.from_pretrained(model_name).eval()
    processor = transformers.AutoFeatureExtractor.from_pretrained(model_name)
    print(model_name, sum(map(torch.numel, model.parameters()))/1e6, 'M parameters')
    cfg = {}
    size = processor.size
    if isinstance(size, int):
        size = [size, size]
    elif isinstance(size, dict):
        size = list(size.values())
        if len(size) == 1:
            size = [size[0]]*2
    assert len(size) == 2
    cfg['input_size'] = size  # H,W
    cfg['interpolation'] = to_cv2_interpolation[processor.resample]
    cfg['mean'] = processor.image_mean
    cfg['std'] = processor.image_std
    format_input = lambda img_tensor: {'pixel_values': img_tensor}
    return model, cfg, format_input


def create_timm_model(model_name):
    model = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
    print(model_name, sum(map(torch.numel, model.parameters()))/1e6, 'M parameters')
    cfg = {}
    cfg['input_size'] = model.default_cfg['input_size'][1:]  # H,W
    cfg['interpolation'] = to_cv2_interpolation[model.default_cfg['interpolation']]
    cfg['mean'] = list(model.default_cfg['mean'])
    cfg['std'] = list(model.default_cfg['std'])
    format_input = lambda img_tensor: {'x': img_tensor}
    return model, cfg, format_input


def get_features(model, inputs, pool_mode='auto'):
    device = next(model.parameters()).device
    # inputs = processor(images=images, return_tensors="pt", padding=True)
    # inputs = {k:x.to(device) for k,x in inputs.items()}
    
    pool_mode_actually = 'none'

    with torch.no_grad():
        if inputs.get('x') is not None:
            # timm models
            if pool_mode == 'auto':
                out = model(**inputs)
                pool_mode_actually = 'pooler'
            elif pool_mode == 'mean':
                out = model.forward_features(**inputs)
                assert out.ndim == 3, f'wrong shape {out.shape}'
                out = out.mean(1)
                pool_mode_actually = 'mean'
        else:
            # HuggingFace
            if isinstance(model, transformers.CLIPModel):
                # CLIP
                out = model.get_image_features(**inputs)
            elif isinstance(model, transformers.FlavaModel):
                # FLAVA
                out = model(**inputs).image_output[1]  # pooled
            elif isinstance(model, transformers.ConvNextModel):
                # ConvNeXt
                out = model(**inputs).pooler_output
            elif isinstance(model, transformers.BeitModel):
                # BEiT
                out = model(**inputs).pooler_output

    assert out.ndim == 2

    return out
