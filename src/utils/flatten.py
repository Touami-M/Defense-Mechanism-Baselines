import torch

def get_update(before, after):
    diff = []
    for k in before:
        diff.append((after[k] - before[k]).view(-1))
    return torch.cat(diff)

def apply_update(model, update):
    new_state = {}
    pointer = 0
    for k,v in model.state_dict().items():
        numel = v.numel()
        delta = update[pointer:pointer+numel].view(v.size())
        new_state[k] = v + delta
        pointer += numel
    model.load_state_dict(new_state)
