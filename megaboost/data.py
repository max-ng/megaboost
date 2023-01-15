import logging
import math
import time
import warnings
from itertools import cycle, islice

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

from .augmentation import RandAugment, CutOutAug
from .autoaugment import CIFAR10Policy, SVHNPolicy
from .utils import AttrDict


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.data)

class TransformToy(object):
    def __init__(self):
        self.ori = transforms.Compose([])
        self.aug = transforms.Compose([])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return ori, aug


def prepare_cifar10(num_labeled=4000, classes=10, eval_step=1000, batch_size=64, resize=32, data_path="./data",randaug=[2,16]):
    args = {"num_labeled": num_labeled,
        "num_classes": classes,
        "eval_step": eval_step, 
        "batch_size": batch_size,
        "resize": resize,
        "data_path": data_path,
        "expand_labels": False,
        "randaug": randaug
       }
    args = AttrDict(args)
    return get_cifar10(args)

def get_toy(args):
    n_samples = 1500
    np.random.seed(0)
    noisy_moons = make_moons(n_samples=n_samples, noise=0.05)
    X, y = noisy_moons
    
    X = StandardScaler().fit_transform(X)
    random_choice = np.random.choice(X.shape[0], 5, replace=False)
    labeled_data = X[random_choice]
    targets  = y[random_choice]

    labeled_dataset = MyDataset(labeled_data, torch.from_numpy(targets).long())
    unlabeled_dataset =MyDataset(X, torch.from_numpy(y).long(), transform=TransformToy())
    test_dataset =MyDataset(X, torch.from_numpy(y).long())

    return labeled_dataset, unlabeled_dataset, test_dataset

def get_cifar10(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(args.data_path, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )

    train_unlabeled_dataset = CIFAR10SSL(
        args.data_path, train_unlabeled_idxs,
        train=True,
        transform=TransformMPL(args, mean=cifar10_mean, std=cifar10_std)
    )

    test_dataset = datasets.CIFAR10(args.data_path, train=False,
                                    transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.RandomCrop(size=args.resize,
                              padding=int(args.resize*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(args.data_path, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        args.data_path, train_labeled_idxs, train=True,
        transform=transform_labeled
    )

    train_unlabeled_dataset = CIFAR100SSL(
        args.data_path, train_unlabeled_idxs, train=True,
        transform=TransformMPL(args, mean=cifar100_mean, std=cifar100_std)
    )

    test_dataset = datasets.CIFAR100(args.data_path, train=False,
                                     transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all training data
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformMPL(object):
    def __init__(self, args, mean, std):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 16  # default

        self.ori = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize*0.125),
                                  padding_mode='reflect')])
        self.aug = transforms.Compose([
            CIFAR10Policy(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize*0.125),
                                  padding_mode='reflect'),
            RandAugment(n=n, m=m),
            CutOutAug(n=1, size=args.resize//4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=args.resize,
                                  padding=int(args.resize*0.125),
                                  padding_mode='reflect')
            ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)

class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
