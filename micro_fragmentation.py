import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
import itertools

from models import *
from db_quant_utils import rel_index



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--k', default=64, type=int, help='depth of model')     
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--opt', type=str, default='Adam')
parser.add_argument('--model_path', type=str, default='./checkpoint')

parser.add_argument('--noise_rate', default=0.0, type=float, help='label noise')
parser.add_argument('--asym', action='store_true')

parser.add_argument('--resolution', default=100, type=int, help='resolution for plot')
parser.add_argument('--range_l', default=0.5, type=float, help='how far `left` to go in the plot')
parser.add_argument('--range_r', default=0.5, type=float, help='how far `right` to go in the plot')
parser.add_argument('--plot_method', default='train', type=str)
parser.add_argument('--active_log', action='store_true') 
parser.add_argument('--num_samples', default=100 , type=int)
parser.add_argument('--set_seed', default=1 , type=int)
parser.add_argument('--set_data_seed', default=1 , type=int)
parser.add_argument('--robusttrain', action='store_true')

## Attack
parser.add_argument('--attack_type', type=str, default='ddn',choices=['pgd','pgdl2','cw','ddn'])
#pgd params
parser.add_argument('--pgd_linf_epsilon', default=8 , type=int)
parser.add_argument('--pgd_l2_epsilon', default=1.0 , type=float)

parser.add_argument('--alpha', default=2 , type=int)
parser.add_argument('--PGD_steps', default=40 , type=int)
parser.add_argument('--no_pgd_random_start', action='store_false')
#cw params
parser.add_argument('--cw_max_steps', default=100 , type=int)
parser.add_argument('--cw_search_steps', default=2 , type=int)
#ddn
parser.add_argument('--DDN_steps', default=100 , type=int)

parser.add_argument('--targeted', action='store_true')
parser.add_argument('--target_class', default=-1 , type=int) # 6 is frog in CIFAR-10

## plane direction
parser.add_argument('--plane_dirs', type=str, default='adversarial',choices=['adversarial','random','otherpoints','advrand'])


args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
args.eval = True
if args.active_log:
    import wandb
    wandb.init(project="micro_adv_fragmentation", name = f'frag_{args.dataset}_{args.k}k_{args.set_seed}')
    wandb.config.update(args)

from data_utils import get_data_adv
_, _, testset, testloader = get_data_adv(args)
num_cls = 10 if args.dataset == 'CIFAR10' else 100

np.random.seed(args.set_seed)
l_test = np.random.choice(np.arange(len(testset.targets)), args.num_samples)
# print(l_test)
mp = args.model_path    
from adv_utils import get_norm_layer
if not args.robusttrain:
    from db.utils import load_model
    net = load_model(args,args.k, initseed=args.set_seed,dataseed=args.set_data_seed,device=device)
    # net.eval()
    from adv_utils import get_norm_layer
    norm_layer = get_norm_layer(args.dataset)
    model = nn.Sequential(
                norm_layer,
                net,
            )
    model = model.to(device)
    model.eval()
else:
    normalize = get_norm_layer(args.dataset)
    net = make_resnet18k(args.k, num_cls)
    model = nn.Sequential(normalize, net)
    model = torch.nn.DataParallel(model)
    modelpath = f'checkpoint/{args.dataset}_robust/{args.set_seed}/{args.set_data_seed}/dd_{args.k}/ckpt.pth'
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['net'])
    model.eval()

from adv_utils import get_attacker,adv_planeloader,adv_connected_components,getdirs_micro

attacker1 = get_attacker(args,model)
args.targeted = True
attacker2 = get_attacker(args,model)

def num_connected_components(dlist1, loader1, num_samples,net,attacker1,attacker2,device,args):
    cc_list = []
    for i in range(num_samples):
        numb = dlist1[i]
        images,dir1,dir2 = getdirs_micro(numb, loader1,attacker1,attacker2,device,args)
        # print([loader1.dataset[numb][1]], np.argmax(net(images).cpu().data.numpy(), 1),np.argmax(net(dir1+images).cpu().data.numpy(), 1),np.argmax(net(images+dir2).cpu().data.numpy(), 1))
        boundary_pred = adv_planeloader([images,dir1,dir2],net,args.resolution)
        if num_samples < 11:
            impath = f'/cmlscratch/gowthami/deci_bounds/micro_dbs/double-descent/temp_imgs/{args.resolution}/{args.plane_dirs}/{args.attack_type}/{args.k}/{numb}'
        else:
            impath = None
        ncc = adv_connected_components(boundary_pred,args,path = impath)
        cc_list.append(ncc)
        # print(dirlist,dirlist2,ncc)
        if i%100==0:
            if args.active_log:
                wandb.log({'iteration':i})
    return cc_list


alltest_alltest = num_connected_components(l_test,testloader,args.num_samples,model,attacker1,attacker2, device,args)


mean_fragmentation = {
        'alltest' : np.mean(alltest_alltest),
        'all_stderror' : np.std(alltest_alltest)/np.sqrt(args.num_samples)
        }
if args.active_log:
    wandb.log(mean_fragmentation)
else:
    print(mean_fragmentation)