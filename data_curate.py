'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import numpy as np
from models import *
from types import SimpleNamespace



def curr_training(limit,k,set_seed,set_data_seed,opti):
    options = SimpleNamespace()
    options.dataset = 'CIFAR10_curr'
    options.epochs = 201
    options.lr = 0.1
    options.k = k
    options.set_seed = set_seed
    options.set_data_seed = set_data_seed
    options.opt = opti
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    from data_utils import get_data
    trainset, trainloader, testset, testloader = get_data(options)

    torch.manual_seed(options.set_seed)
    num_cls = 10
    net = make_resnet18k(options.k,num_cls)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

    if options.opt == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=options.lr,
                            momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    elif options.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)


    # Training
    def train(epoch):
        print('\nCurri Epoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        corr_indices = []
        for batch_idx, (inputs, targets,indices) in enumerate(trainloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            corr_indices += list(indices[predicted.eq(targets).cpu().numpy()].numpy())
        print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # import ipdb; ipdb.set_trace()
        train_loss = train_loss/(batch_idx+1)
        train_acc = (correct/total)
        return train_acc, corr_indices


    for epoch in range(0,options.epochs):
        train_acc, corr_indices = train(epoch)
        if train_acc > limit:
            return corr_indices
        if options.opt == "SGD":
            scheduler.step()

    raise Exception('Did not go beyond needed test accuracy in given # of epochs')



