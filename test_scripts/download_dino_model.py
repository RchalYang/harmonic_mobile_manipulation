import torch
model_type = ["dinov2_vitb14", "dinov2_vits14"]
for model in model_type:
    torch.hub.load("facebookresearch/dinov2", model)