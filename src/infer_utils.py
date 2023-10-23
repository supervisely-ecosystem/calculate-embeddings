from typing import Optional
import cv2
import torch
import timm
import transformers
import numpy as np
from PIL.Image import Resampling

import src.clip_api as capi


to_cv2_interpolation = {
    Resampling.NEAREST: cv2.INTER_NEAREST,
    Resampling.BILINEAR: cv2.INTER_LINEAR,
    Resampling.BICUBIC: cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


def create_model(model_name, device, model_data: Optional[str] = None):
    if model_data is None:
        return create_hf_model(model_name, device)
    else:
        return create_clip_model(model_name, model_data)


def create_hf_model(model_name, device):
    model = transformers.AutoModel.from_pretrained(model_name).eval()
    processor = transformers.AutoFeatureExtractor.from_pretrained(model_name)
    print(model_name, sum(map(torch.numel, model.parameters())) / 1e6, "M parameters")
    cfg = {}
    size = processor.size
    if isinstance(size, int):
        size = [size, size]
    elif isinstance(size, dict):
        size = list(size.values())
        if len(size) == 1:
            size = [size[0]] * 2
    assert len(size) == 2
    cfg["input_size"] = size  # H,W
    cfg["interpolation"] = to_cv2_interpolation[processor.resample]
    cfg["mean"] = processor.image_mean
    cfg["std"] = processor.image_std

    def format_input(img_batch: np.ndarray):
        img_batch = img_batch.astype(np.float32) / 255
        img_batch = normalize(img_batch, cfg["mean"], cfg["std"], np_dtype=np.float32)
        img_batch = img_batch.transpose(0, 3, 1, 2)
        input_batch = torch.tensor(img_batch, device=device)
        return {"pixel_values": input_batch}

    return model, cfg, format_input


def create_clip_model(model_name, device, model_data):
    model, processor, tockenizer = capi.build_model(model_name, device, model_data)
    model.eval()
    print(model_name, sum(map(torch.numel, model.parameters())) / 1e6, "M parameters")
    
    def format_input(img_batch: np.ndarray):
        return processor(img_batch)
    return model, {}, format_input


def get_features(model, inputs, pool_mode="auto"):
    device = next(model.parameters()).device
    # inputs = processor(images=images, return_tensors="pt", padding=True)
    # inputs = {k:x.to(device) for k,x in inputs.items()}

    pool_mode_actually = "none"

    with torch.no_grad():
        if inputs.get("x") is not None:
            # timm models
            if pool_mode == "auto":
                out = model(**inputs)
                pool_mode_actually = "pooler"
            elif pool_mode == "mean":
                out = model.forward_features(**inputs)
                assert out.ndim == 3, f"wrong shape {out.shape}"
                out = out.mean(1)
                pool_mode_actually = "mean"
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


def normalize(img_batch, mean, std, np_dtype=np.float32):
    # img_batch: [B,H,W,C]
    assert img_batch.shape[3] == 3
    mean = np.array(mean, dtype=np_dtype)
    std = np.array(std, dtype=np_dtype)
    return (img_batch - mean) / std