import torch, copy, random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.models.simplenet_mlp import SimpleNet
from src.data.loaders import load_fashionmnist
from src.data.partition import iid_partition
from src.utils.training import train_one_epoch, evaluate
from src.utils.flatten import get_update, apply_update
from src.aggregation.fltrust import aggregate
from src.attacks.random_update import apply_random_attack

device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds, test_ds = load_fashionmnist()
test_loader = DataLoader(test_ds, batch_size=256)

root_loader = DataLoader(Subset(train_ds, list(range(200))), batch_size=32, shuffle=True)
client_loaders = iid_partition(Subset(train_ds, list(range(200,len(train_ds)))), 5)

criterion = nn.CrossEntropyLoss()

global_model = SimpleNet().to(device)
server_model = SimpleNet().to(device)

opt_server = optim.SGD(server_model.parameters(), lr=0.1)

for _ in range(3):
    train_one_epoch(server_model, root_loader, device, criterion, opt_server)

global_model.load_state_dict(server_model.state_dict())

for rnd in range(20):

    before_server = copy.deepcopy(server_model.state_dict())
    train_one_epoch(server_model, root_loader, device, criterion, opt_server)
    server_update = get_update(before_server, server_model.state_dict())

    updates = []

    for loader in client_loaders:

        local = SimpleNet().to(device)
        local.load_state_dict(global_model.state_dict())
        opt = optim.SGD(local.parameters(), lr=0.1)

        if random.random() < 0.2:
            apply_random_attack(local)
        else:
            train_one_epoch(local, loader, device, criterion, opt)

        upd = get_update(global_model.state_dict(), local.state_dict())
        updates.append(upd)

    agg_update = aggregate(updates, server_update)
    apply_update(global_model, agg_update)

    print("Round", rnd, "acc:", evaluate(global_model, test_loader, device))
