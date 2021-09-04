import csv, torchvision, numpy as np, random, os
from PIL import Image
import torch

from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Subset
from torchvision import transforms, datasets
from collections import defaultdict
import math
import random

from .random_erase import RandomErasing
from .cutout import Cutout


class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations


class DatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.targets = dataset.targets
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]



def load_dataset(args):
    class TwoCropsTransform:
        """Take two random crops of one image as the query and key."""

        def __init__(self, base_transform):
            self.base_transform = base_transform

        def __call__(self, x):
            q = self.base_transform(x)
            k = self.base_transform(x)
            return [q, k]

    transforms_list = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408],
                        [0.2675, 0.2565, 0.2761]),
    ]
    if args.data_aug == 'cutout':
        transforms_list.append(Cutout(n_holes=1, length=8))
    if args.data_aug == 'random_erase':
        transforms_list.append(RandomErasing(mean=[0.5071, 0.4867, 0.4408]))
    transform_train = transforms.Compose(transforms_list)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])


    if args.method == 'DDGSD':
        trainset = datasets.CIFAR100(args.data, train=True,  download=True, transform=TwoCropsTransform(transform_train))
    else:
        trainset = datasets.CIFAR100(args.data, train=True,  download=True, transform=transform_train)
    
    valset   = datasets.CIFAR100(args.data, train=False, download=True, transform=transform_test)


    if args.method == 'CS-KD':
        get_train_sampler = lambda d: PairBatchSampler(d, args.batch_size)
        trainset = DatasetWrapper(trainset)
        trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=args.num_workers)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True)
    return trainloader, valloader
