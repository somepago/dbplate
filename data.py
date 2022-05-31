import torchvision
import torchvision.transforms as transforms
import torch
import random
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy

# class cifar10curr(Dataset):
#     def __init__(self,root, train=True, transform=None, target_transform=None, download=True):
#         self.cifar10 = datasets.CIFAR10(root=root,
#                                         download=download,
#                                         train=train,
#                                         transform=transform,target_transform=target_transform)
        
#     def __getitem__(self, index):
#         data, target = self.cifar10[index]        
#         return data, target, index

#     def __len__(self):
#         return len(self.cifar10)

from data_curate import curr_training


def get_indices(args,index_seed,sampling_style):
    # global options
    n = 50000 if args.dataset in ['CIFAR10','CIFAR10_partial'] else r'dataset case not written'
    m = int(args.partial_data_pc*n)
    if sampling_style == 'random':
        np.random.seed(index_seed)
        # np.random.seed(args.set_data_seed)
        train_indices = np.random.choice(n, m, replace=False)
    elif sampling_style == 'halfsplit':
        set_pairs = {30:20,50:1}
        if args.set_data_seed in set_pairs.keys():
            temp_indices = get_indices(args,set_pairs[args.set_data_seed],'random')
            lo_indices = set(np.arange(n)) - set(temp_indices)
            np.random.seed(args.set_data_seed)
            train_indices = np.random.choice(list(lo_indices), m, replace=False)
        elif args.set_data_seed in set_pairs.values():
            train_indices = np.random.choice(n, m, replace=False)     
        else:
            raise Exception('This seed is not valid with partial data half-split setting')
    elif sampling_style == 'curated':
        np.random.seed(index_seed)
        temp_indices = curr_training(args.partial_data_pc,k=args.k,set_seed=args.set_init_seed,set_data_seed= args.set_data_seed,opti = args.opt)
        train_indices = np.random.choice(temp_indices, m, replace=False)
        # raise Exception('curation case not written! Write ASAP woman')
    return train_indices


def get_data(args):

    if args.dataset == 'CIFAR10':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        torch.manual_seed(args.set_data_seed)
        if args.noise_rate > 0: 
            trainset = cifar10Nosiy(root='./data', train=True,transform=transform_train, download=True,
                                                    asym=args.asym,
                                                    nosiy_rate=args.noise_rate)
            
        else:
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=4)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=4)

    elif args.dataset == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071598291397095, 0.4866936206817627,0.44120192527770996), 
            (0.2673342823982239, 0.2564384639263153, 0.2761504650115967)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071598291397095, 0.4866936206817627,0.44120192527770996), 
            (0.2673342823982239, 0.2564384639263153, 0.2761504650115967)),
        ])
        torch.manual_seed(args.set_data_seed)
        if args.noise_rate > 0: 
            raise'This case is not written'
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
            download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=4)
        
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=4)

    elif args.dataset == 'CIFAR10_partial':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        torch.manual_seed(args.set_data_seed)
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(
        #     trainset, batch_size=128, shuffle=True, num_workers=4)
        # from torch.utils.data.sampler import SubsetRandomSampler
        # 
        # n = len(trainset)
        # m = int(args.partial_data_pc*n)
        # np.random.seed(args.set_data_seed)
        # train_indices = np.random.choice(n, m, replace=False)
        # train_sampler = SubsetRandomSampler(train_indices)
        
        train_indices = get_indices(args,args.set_data_seed,args.partial_data_style)
        print(args.set_init_seed, args.set_data_seed, train_indices[:10])
        # import ipdb; ipdb.set_trace()
        subset_train = torch.utils.data.Subset(trainset, train_indices)
        # subset_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, sampler=train_sampler,num_workers=4)
        subset_loader = torch.utils.data.DataLoader(subset_train, batch_size=128, shuffle=True,num_workers=4)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=4)

        return subset_train, subset_loader, testset, testloader

    # elif args.dataset == 'CIFAR10_curr':

    #     transform_train = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])

    #     transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])

    #     torch.manual_seed(args.set_data_seed)
    #     trainset = cifar10curr(root='./data', train=True,transform=transform_train, download=True)
    #     # import ipdb; ipdb.set_trace()
    #     trainloader = torch.utils.data.DataLoader(
    #         trainset, batch_size=128, shuffle=True, num_workers=4)
    #     testset = cifar10curr(root='./data', train=False,transform=transform_test, download=True)
    #     testloader = torch.utils.data.DataLoader(
    #         testset, batch_size=100, shuffle=False, num_workers=4)

  
    return trainset, trainloader, testset, testloader


def get_plane(img1, img2, img3):
    ''' Calculate the plane (basis vecs) spanned by 3 images
    Input: 3 image tensors of the same size
    Output: two (orthogonal) basis vectors for the plane spanned by them, and
    the second vector (before being made orthogonal)
    '''
    a = img2 - img1
    b = img3 - img1
    a_norm = torch.dot(a.flatten(), a.flatten()).sqrt()
    a = a / a_norm
    first_coef = torch.dot(a.flatten(), b.flatten())
    #first_coef = torch.dot(a.flatten(), b.flatten()) / torch.dot(a.flatten(), a.flatten())
    b_orthog = b - first_coef * a
    b_orthog_norm = torch.dot(b_orthog.flatten(), b_orthog.flatten()).sqrt()
    b_orthog = b_orthog / b_orthog_norm
    second_coef = torch.dot(b.flatten(), b_orthog.flatten())
    #second_coef = torch.dot(b_orthog.flatten(), b.flatten()) / torch.dot(b_orthog.flatten(), b_orthog.flatten())
    coords = [[0,0], [a_norm,0], [first_coef, second_coef]]
    return a, b_orthog, b, coords


class plane_dataset(torch.utils.data.Dataset):
    def __init__(self, base_img, vec1, vec2, coords, resolution=0.2,
                    range_l=.1, range_r=.1):
        self.base_img = base_img
        self.vec1 = vec1
        self.vec2 = vec2
        self.coords = coords
        self.resolution = resolution
        x_bounds = [coord[0] for coord in coords]
        y_bounds = [coord[1] for coord in coords]

        self.bound1 = [torch.min(torch.tensor(x_bounds)), torch.max(torch.tensor(x_bounds))]
        self.bound2 = [torch.min(torch.tensor(y_bounds)), torch.max(torch.tensor(y_bounds))]

        len1 = self.bound1[-1] - self.bound1[0]
        len2 = self.bound2[-1] - self.bound2[0]

        #list1 = torch.linspace(self.bound1[0] - 0.1*len1, self.bound1[1] + 0.1*len1, int(resolution))
        #list2 = torch.linspace(self.bound2[0] - 0.1*len2, self.bound2[1] + 0.1*len2, int(resolution))
        list1 = torch.linspace(self.bound1[0] - range_l*len1, self.bound1[1] + range_r*len1, int(resolution))
        list2 = torch.linspace(self.bound2[0] - range_l*len2, self.bound2[1] + range_r*len2, int(resolution))

        grid = torch.meshgrid([list1,list2])

        self.coefs1 = grid[0].flatten()
        self.coefs2 = grid[1].flatten()

    def __len__(self):
        return self.coefs1.shape[0]

    def __getitem__(self, idx):
        return self.base_img + self.coefs1[idx] * self.vec1 + self.coefs2[idx] * self.vec2

def make_planeloader(images, args):
    a, b_orthog, b, coords = get_plane(images[0], images[1], images[2])

    planeset = plane_dataset(images[0], a, b_orthog, coords, resolution=args.resolution, range_l=args.range_l, range_r=args.range_r)

    planeloader = torch.utils.data.DataLoader(
        planeset, batch_size=256, shuffle=False, num_workers=2)
    return planeloader
