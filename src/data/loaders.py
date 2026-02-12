from torchvision import datasets, transforms

def load_fashionmnist():
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.FashionMNIST("../data", train=True, download=True, transform=transform)
    test = datasets.FashionMNIST("../data", train=False, download=True, transform=transform)
    return train, test
