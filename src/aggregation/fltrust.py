import torch

def aggregate(updates, server_update):

    server_norm = torch.norm(server_update)
    server_dir = server_update / (server_norm + 1e-10)

    trust_scores = []
    norm_updates = []

    for upd in updates:
        upd_norm = torch.norm(upd)
        upd_dir = upd / (upd_norm + 1e-10)

        trust = max(torch.dot(upd_dir, server_dir).item(), 0.0)
        trust_scores.append(trust)

        norm_updates.append(upd * (server_norm / (upd_norm + 1e-10)))

    total_trust = sum(trust_scores)

    if total_trust == 0:
        return sum(norm_updates) / len(norm_updates)

    return sum(norm_updates[i] * (trust_scores[i]/total_trust)
               for i in range(len(norm_updates)))
