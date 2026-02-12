import torch

def apply_random_attack(model, scale=5.0):
    for p in model.parameters():
        p.data.add_(torch.randn_like(p)*scale)
