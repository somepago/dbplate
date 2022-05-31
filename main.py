'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import pickle

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from model import get_model
from data import get_data, make_planeloader
from utils import get_loss_function, get_scheduler, get_random_images, produce_plot, get_noisy_images, AttackPGD
from evaluation import train, test, test_on_trainset, decision_boundary, test_on_adv
from options import options
from utils import simple_lapsed_time

args = options().parse_args()
print(args)
#torch.manual_seed(args.set_init_seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = args.save_net
if args.active_log:
    import wandb
    idt = '_'.join(list(map(str,args.imgs)))
    wandb.init(project="decision_boundaries", name = '_'.join([args.net,args.train_mode,idt,'seed'+str(args.set_init_seed)]) )
    wandb.config.update(args)

# Data/other training stuff
torch.manual_seed(args.set_data_seed)
# import ipdb; ipdb.set_trace()
trainset, trainloader, testset, testloader = get_data(args)
torch.manual_seed(args.set_init_seed)
test_accs = []
train_accs = []
# import ipdb; ipdb.set_trace()
net = get_model(args, device)

test_acc, predicted = test(args, net, testloader, device, 0)
print("scratch prediction ", test_acc)

criterion = get_loss_function(args)
if args.opt == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = get_scheduler(args, optimizer)

elif args.opt == 'Adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

# Train or load base network
print("Training the network or loading the network")

start = time.time()
best_acc = 0  # best test accuracy
best_epoch = 0
if args.load_net is None:
    for epoch in range(args.epochs):
        train_acc = train(args, net, trainloader, optimizer, criterion, device, args.train_mode, sam_radius=args.sam_radius)
        test_acc, predicted = test(args, net, testloader, device, epoch)
        print(f'EPOCH:{epoch}, Test acc: {test_acc}')
        if args.opt == 'SGD':
            scheduler.step()

        # Save checkpoint.
        if epoch%10 ==0:
            os.makedirs(f'saved_models/{args.train_mode}/{str(args.set_init_seed)}', exist_ok=True)
            print(f'saved_models/{args.train_mode}/{str(args.set_init_seed)}/{args.save_net}.pth')
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(),
                            f'saved_models/{args.train_mode}/{str(args.set_init_seed)}/{args.save_net}.pth')
            else:
                torch.save(net.state_dict(),
                            f'saved_models/{args.train_mode}/{str(args.set_init_seed)}/{args.save_net}.pth')
        if test_acc > best_acc:
            print(f'The best epoch is: {epoch}')
            best_acc = test_acc
            best_epoch = epoch

else:
    net.load_state_dict(torch.load(args.load_net))
    

if args.load_net is None and args.active_log:
                wandb.log({'best_epoch': epoch ,'best_test_accuracy': best_acc,'test_acc':test_acc
                    })
else:
    print(f"best_epoch: {epoch} ,best_test_accuracy: {best_acc},test_acc:{test_acc}")
end = time.time()
simple_lapsed_time("Time taken to train/load the model", end-start)
