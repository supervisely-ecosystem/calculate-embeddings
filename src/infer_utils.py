import cv2
import torch
import timm
import transformers
from PIL.Image import Resampling


to_cv2_interpolation = {
    Resampling.NEAREST: cv2.INTER_NEAREST,
    Resampling.BILINEAR: cv2.INTER_LINEAR,
    Resampling.BICUBIC: cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


def create_model(model_name):
    # simple detect if '/' in model_name
    if "/" in model_name:
        return create_hf_model(model_name)
    else:
        return create_timm_model(model_name)


def create_hf_model(model_name):
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
    format_input = lambda img_tensor: {"pixel_values": img_tensor}
    return model, cfg, format_input


def create_timm_model(model_name):
    model = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
    print(model_name, sum(map(torch.numel, model.parameters())) / 1e6, "M parameters")
    cfg = {}
    cfg["input_size"] = model.default_cfg["input_size"][1:]  # H,W
    cfg["interpolation"] = to_cv2_interpolation[model.default_cfg["interpolation"]]
    cfg["mean"] = list(model.default_cfg["mean"])
    cfg["std"] = list(model.default_cfg["std"])
    format_input = lambda img_tensor: {"x": img_tensor}
    return model, cfg, format_input


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
