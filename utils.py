import copy
import torch
from torchvision import datasets, transforms
import numpy as np


trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

def get_dataset(args):
    data_dir = args.data_dir + args.dataset
    if args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

        if args.iid:
            user_groups = creat_iid(train_dataset, args.num_users)
        else:
            pass
    
    elif args.dataset == 'cifar10':
        apply_transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        apply_transform_test = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform_test)

        if args.iid:
            user_groups = creat_iid(train_dataset, args.num_users)
        else:
            pass
    
    elif args.dataset == 'language':
        print('split the dataset here')
    
    return train_dataset, user_groups


def creat_iid(dataset, num_users):
    '''Create data index for iid style'''
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users





