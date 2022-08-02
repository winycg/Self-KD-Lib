import csv, torchvision, numpy as np, random, os
from PIL import Image
import torch
import copy
from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Subset
from torchvision import transforms, datasets
from collections import defaultdict
from .random_erase import RandomErasing
from .cutout import Cutout
import math
import random
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from torchvision.transforms import AugMix
from torchvision.datasets import CIFAR100


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


class IdentityBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_instances, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            offset = k*self.batch_size%len(indices)
            batch_indices = indices[offset:offset+self.batch_size]

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                t = copy.deepcopy(self.dataset.classwise_indices[y])
                t.pop(t.index(idx))
                if len(t)>=(self.num_instances-1):
                    class_indices = np.random.choice(t, size=self.num_instances-1, replace=False)
                else:
                    class_indices = np.random.choice(t, size=self.num_instances-1, replace=True)
                pair_indices.extend(class_indices)

            yield batch_indices+pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // (self.batch_size)
        else:
            return self.num_iterations


class Custom_CIFAR100(CIFAR100):
    #------------------------
    #Custom CIFAR-100 dataset which returns returns 1 images, 1 target, image index
    #------------------------
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, [target, index]


class DatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        self.targets = dataset.targets
        self.classes = dataset.classes
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

class TwoCropsTransform:
        """Take two random crops of one image as the query and key."""

        def __init__(self, base_transform):
            self.base_transform = base_transform

        def __call__(self, x):
            q = self.base_transform(x)
            k = self.base_transform(x)
            return [q, k]

def load_dataset(args):
    name = args.dataset
    root = args.data
    if name.startswith('CIFAR-100'):
        transforms_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        ]
        if args.data_aug == 'auto_aug':
            transforms_list.append(transforms.AutoAugment(AutoAugmentPolicy.CIFAR10))
        if args.data_aug == 'randaug':
            transforms_list.insert(0, transforms.RandAugment())
        if args.data_aug == 'augmix':
            transforms_list.append(transforms.AugMix())
        if args.data_aug == 'trivialaug':
            transforms_list.append(transforms.TrivialAugmentWide())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize([0.5071, 0.4867, 0.4408],[0.2675, 0.2565, 0.2761]))
        
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
            trainset = datasets.CIFAR100(root, train=True,  download=True, transform=TwoCropsTransform(transform_train))
        elif args.method == 'PSKD':
            trainset = Custom_CIFAR100(root, train=True,  download=True, transform=transform_train)
        else:
            trainset = datasets.CIFAR100(root, train=True,  download=True, transform=transform_train)
        
        valset   = datasets.CIFAR100(root, train=False, download=True, transform=transform_test)

    elif name in ['imagenet','CUB200', 'Dogs', 'MIT67', 'Cars', 'Air']:
        transforms_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ]
        if args.data_aug == 'auto_aug':
            transforms_list.append(transforms.AutoAugment(AutoAugmentPolicy.CIFAR10))
        if args.data_aug == 'randaug':
            transforms_list.insert(0, transforms.RandAugment())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))


        transform_train = transforms.Compose(transforms_list)
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_val_dataset_dir = os.path.join(root, "train")
        test_dataset_dir = os.path.join(root, "val")

        if args.kd_method == 'DDGSD':
            trainset = datasets.ImageFolder(root=train_val_dataset_dir, transform=TwoCropsTransform(transform_train))
        else:
            trainset = datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train)
        
        valset   = datasets.ImageFolder(root=test_dataset_dir, transform=transform_test)

    else:
        raise Exception('Unknown dataset: {}'.format(name))

    # Sampler
    if args.method.startswith('CS-KD'):
        get_train_sampler = lambda d: PairBatchSampler(d, args.batch_size)
        trainset = DatasetWrapper(trainset)
        trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=args.num_workers)
    elif args.method.startswith('BAKE'):
        get_train_sampler = lambda d: IdentityBatchSampler(d, args.batch_size, args.intra_imgs+1)
        trainset = DatasetWrapper(trainset)
        trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=args.num_workers)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=(torch.cuda.is_available()))
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=(torch.cuda.is_available()))
    return trainloader, valloader
