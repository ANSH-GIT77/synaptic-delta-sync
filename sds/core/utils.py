import torch

def get_model_stats(model):
    """Returns memory usage or weight distribution stats."""
    return {k: v.norm().item() for k, v in model.named_parameters()}