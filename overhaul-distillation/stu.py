"""Train resnet on CIFAR10 with sparse regulartion.
   More detail can been find in paper: OICSR: Out-In-Channel 
   Sparsity Regularization for Compact Deep Neural Networks
"""
import os
import sys
import time
import torch
import random
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import distillation
from util import test, build_dataset, AverageMeter, accuracy

sys.path.append("..")

from model.cifar_resnet import *
from model.cifar_Resnet import *

def get_parser():
    parser = argparse.ArgumentParser(description='Knowledge distillation on CIFAR10 & CIFAR100')
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--t_model', default='resnet56', type=str, help='teacher model architecture',
                        choices=['resnet18', 'resnet34', 'resnet56', 'resnet110'])
    parser.add_argument('--s_model', default='resnet20', type=str, help='student model architecture',
                        choices=['resnet18', 'resnet20', 'resnet34', 'resnet56', 'resnet110'])
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--weight_decay', default=5e-4, type=float,help='weight decay for optimizers')
    parser.add_argument('--T', type=int, default=4, help='Temperature')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--device', default=0, type=int, help='GPU index')
    return parser

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def train_with_distill(d_net, optimizer, device, train_loader, criterion):
    
    
    d_net.to(device)
    d_net.s_net.to(device)
    d_net.t_net.to(device)
    d_net.train()
    d_net.s_net.train()
    d_net.t_net.train()
    top1 = AverageMeter() 
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
   
        inputs, targets = inputs.to(device), targets.to(device)
        

        batch_size = inputs.shape[0]
        outputs, loss_distill = d_net(inputs)
        loss_CE = criterion(outputs, targets)
        loss = loss_CE + 1e-4 * loss_distill.sum() / batch_size
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs = outputs.float()
        prec1 = accuracy(outputs.data, targets)[0]
        top1.update(prec1.item(), inputs.size(0))
        
    return top1.avg


def main():

    parser = get_parser()
    args = parser.parse_args()
    setup_seed(args.seed)
    device = 'cuda:' + str(args.device)
    train_loader, test_loader = build_dataset(args)
    train_accuracies = []
    test_accuracies = []
    class_num = 10 if args.dataset == 'cifar10' else 100

    t_net = {'resnet18': resnet18, 'resnet34': resnet34, 
           'resnet56': resnet56, 'resnet110': resnet110}[args.t_model](class_num)
    t_ckpt_name = 'SGD-CIFAR' + str(class_num) + '-' + args.t_model
    if args.dataset == 'cifar10':
        path = '../ckpt/checkpoint/cifar10/' + t_ckpt_name
    else:
        path = '../ckpt/checkpoint/cifar100/' + t_ckpt_name
    ckpt = torch.load(path, map_location=device)
    t_net.load_state_dict(ckpt['net'])
    t_net = t_net.to(device)
  
    s_ckpt_name = 'SGD-CIFAR' + str(class_num) + '-' + args.s_model + '-student' + '-overhaul2' 
    s_net = {'resnet18': resnet18, 'resnet20': resnet20, 'resnet34': resnet34, 
            'resnet56': resnet56, 'resnet110': resnet110}[args.s_model](class_num)
    s_net = s_net.to(device)
    optimizer = optim.SGD(s_net.parameters(), args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    d_net = distillation.Distiller(t_net, s_net)
    
    start_epoch = 0

    best_acc = 0
    start = time.time()
    for epoch in range(start_epoch, 150):
       
        if epoch in [80, 120]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        train_acc = train_with_distill(d_net, optimizer, device, train_loader, criterion)
        test_acc = test(s_net, device, test_loader, criterion)
        end = time.time()
        print('epoch %d, train %.3f, test %.3f, time %.3fs'%(epoch, train_acc, test_acc, end-start))
        start = time.time()
        
        # Save checkpoint.
        if best_acc < test_acc :
            best_acc = test_acc
            if epoch > 80:
                state = {
                    'net': s_net.state_dict(),
                }
                if not os.path.isdir('../ckpt/checkpoint'):
                    os.mkdir('../ckpt/checkpoint')
                if args.dataset == 'cifar10':
                    if not os.path.isdir('../ckpt/checkpoint/cifar10'):
                        os.mkdir('../ckpt/checkpoint/cifar10')
                    torch.save(state, os.path.join('../ckpt/checkpoint/cifar10', s_ckpt_name))
                elif args.dataset == 'cifar100':
                    if not os.path.isdir('../ckpt/checkpoint/cifar100'):
                        os.mkdir('../ckpt/checkpoint/cifar100')
                    torch.save(state, os.path.join('../ckpt/checkpoint/cifar100', s_ckpt_name))
  
        print('best_acc %.3f'%best_acc)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        if not os.path.isdir('../ckpt/curve'):
            os.mkdir('../ckpt/curve')
        if args.dataset == 'cifar10':
            if not os.path.isdir('../ckpt/curve/cifar10'):
                os.mkdir('../ckpt/curve/cifar10')
            torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                   os.path.join('../ckpt/curve/cifar10', s_ckpt_name))
        elif args.dataset == 'cifar100':
            if not os.path.isdir('../ckpt/curve/cifar100'):
                os.mkdir('../ckpt/curve/cifar100')
            torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                    os.path.join('../ckpt/curve/cifar100', s_ckpt_name))

if __name__ == '__main__':
    main()

