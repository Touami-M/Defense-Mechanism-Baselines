import numpy as np
import random
from torch.utils.data import DataLoader, Subset

def iid_partition(dataset, num_clients, batch_size=32):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [DataLoader(Subset(dataset, list(idx)), batch_size=batch_size, shuffle=True)
            for idx in splits]
