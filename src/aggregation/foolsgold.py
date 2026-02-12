import torch

def aggregate(updates):

    K = len(updates)
    sims = torch.zeros((K,K))

    for i in range(K):
        for j in range(K):
            if i != j:
                sims[i,j] = torch.dot(updates[i], updates[j]) / (
                    torch.norm(updates[i]) * torch.norm(updates[j]) + 1e-10)

    max_sims = torch.max(sims, dim=1)[0]
    weights = 1.0 - max_sims
    weights = torch.clamp(weights, min=0.0)

    if torch.sum(weights) == 0:
        weights = torch.ones_like(weights)/K
    else:
        weights = weights/torch.sum(weights)

    return sum(updates[i]*weights[i] for i in range(K))
