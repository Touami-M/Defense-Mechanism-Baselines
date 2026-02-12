# experiments/run_foolsgold_vs_fedavg.py
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.simplenet_mlp import SimpleNet
from src.data.loaders import load_fashionmnist
from src.data.partition import iid_partition
from src.utils.training import train_one_epoch, evaluate
from src.utils.flatten import get_update, apply_update
from src.aggregation.fedavg import aggregate as fedavg_aggregate
from src.aggregation.foolsgold import aggregate as foolsgold_aggregate
from src.attacks.random_update import apply_random_attack

# ----------------------------
# Config
# ----------------------------
NUM_CLIENTS = 5
NUM_ROUNDS = 15
MALICIOUS_FRAC = 0.3
BATCH_SIZE = 32
LR = 0.1
ATTACK_SCALE = 5.0  # used by random_update attack

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ----------------------------
# Data
# ----------------------------
train_ds, test_ds = load_fashionmnist()
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# Client loaders (IID for baseline)
client_loaders = iid_partition(train_ds, num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE)

criterion = nn.CrossEntropyLoss()

# ----------------------------
# Two independent global models
# ----------------------------
global_fedavg = SimpleNet().to(device)
global_fg = SimpleNet().to(device)

acc_fedavg, acc_fg = [], []

# ----------------------------
# Training loop
# ----------------------------
for rnd in range(NUM_ROUNDS):
    print(f"\n=== Round {rnd} ===")

    # ---- Collect client updates relative to FedAvg global ----
    updates_fedavg = []
    for i, loader in enumerate(client_loaders):
        local = SimpleNet().to(device)
        local.load_state_dict(global_fedavg.state_dict())
        opt = optim.SGD(local.parameters(), lr=LR)

        # Malicious?
        if random.random() < MALICIOUS_FRAC:
            apply_random_attack(local, scale=ATTACK_SCALE)
        else:
            train_one_epoch(local, loader, device, criterion, opt)

        upd = get_update(global_fedavg.state_dict(), local.state_dict())
        updates_fedavg.append(upd)

    # ---- FedAvg aggregation & update ----
    fedavg_update = fedavg_aggregate(updates_fedavg)
    tmp = copy.deepcopy(global_fedavg)
    apply_update(tmp, fedavg_update)
    global_fedavg.load_state_dict(tmp.state_dict())
    a1 = evaluate(global_fedavg, test_loader, device)
    acc_fedavg.append(a1)

    # ---- Collect client updates relative to FoolsGold global ----
    updates_fg = []
    for i, loader in enumerate(client_loaders):
        local = SimpleNet().to(device)
        local.load_state_dict(global_fg.state_dict())
        opt = optim.SGD(local.parameters(), lr=LR)

        # Same malicious sampling style (independent per client per round)
        if random.random() < MALICIOUS_FRAC:
            apply_random_attack(local, scale=ATTACK_SCALE)
        else:
            train_one_epoch(local, loader, device, criterion, opt)

        upd = get_update(global_fg.state_dict(), local.state_dict())
        updates_fg.append(upd)

    # ---- FoolsGold aggregation & update ----
    fg_update = foolsgold_aggregate(updates_fg)
    tmp2 = copy.deepcopy(global_fg)
    apply_update(tmp2, fg_update)
    global_fg.load_state_dict(tmp2.state_dict())
    a2 = evaluate(global_fg, test_loader, device)
    acc_fg.append(a2)

    print(f"FedAvg acc: {a1:.4f} | FoolsGold acc: {a2:.4f}")

print("\nFinal Accuracy:")
print(f"FedAvg:    {acc_fedavg[-1]:.4f}")
print(f"FoolsGold: {acc_fg[-1]:.4f}")

# Optional plot (only if you want it)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(acc_fedavg, label="FedAvg")
    plt.plot(acc_fg, label="FoolsGold")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.title(f"FedAvg vs FoolsGold (malicious_frac={MALICIOUS_FRAC})")
    plt.grid(True)
    plt.legend()
    plt.show()
except Exception as e:
    print("Plot skipped:", e)
