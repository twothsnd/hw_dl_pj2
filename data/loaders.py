"""CIFAR-10 data loaders for Project 2."""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets



class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_transforms(train=True, augment=False, cifar_stats=True):
    if cifar_stats:
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        )
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

    transform_list = []
    if train and augment:
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    transform_list.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(transform_list)


def get_cifar_dataset(root='./data', train=True, n_items=-1, augment=False, download=True, cifar_stats=True):
    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=get_cifar_transforms(train=train, augment=augment, cifar_stats=cifar_stats),
    )
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)
    return dataset


def get_cifar_loader(
    root='./data',
    batch_size=128,
    train=True,
    shuffle=None,
    num_workers=4,
    n_items=-1,
    augment=False,
    download=True,
    pin_memory=True,
    cifar_stats=True,
):
    if shuffle is None:
        shuffle = train

    dataset = get_cifar_dataset(
        root=root,
        train=train,
        n_items=n_items,
        augment=augment,
        download=download,
        cifar_stats=cifar_stats,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


if __name__ == '__main__':
    train_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        plt.imshow(img*0.5 + 0.5)
        plt.savefig('sample.png')
        print(X[0].max())
        print(X[0].min())
        break
