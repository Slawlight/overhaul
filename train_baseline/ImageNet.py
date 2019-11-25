"""Train IMAGENET with PyTorch."""
import os
import sys
import torch
import time
import argparse
import warnings
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from utils_ImageNet import train, test, build_dataset, AverageMeter, accuracy
sys.path.append('../')
from model.resnet import *

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', default='imagenet', type=str, choices=['imagenet'])
    parser.add_argument('--model', default='resnet18', type=str, help='model', choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--device', default=0, type=int, help='GPU index')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    return parser

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def main():
    
    warnings.filterwarnings('ignore')
    parser = get_parser()
    args = parser.parse_args()
    setup_seed(args.seed)
    device_num = args.device
    train_loader, val_loader = build_dataset(args)
    device = 'cuda:'+str(device_num)
    ckpt_name = 'SGD-ImageNet' + '-' + args.model
    
    net = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}[args.model]()
    net = net.to(device)
    if args.model == 'resnet50':
        net = torch.nn.DataParallel(net).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    
    start_epoch = 0
    train1_acc = []
    train5_acc = []
    val1_acc = []
    val5_acc = []
   
    best_acc1 = 0

    start = time.time()
    for epoch in range(start_epoch, 90):
        if epoch in [30, 60]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                
        train_1, train_5 = train(net, device, train_loader, optimizer, criterion)
        val_1, val_5 = test(net, device, val_loader, criterion)
        
        end = time.time()
        print('epoch %d, train@1 %.3f, test@1 %.3f, time %.3fs'%(epoch, train_1, val_1, end-start))
        start = time.time()
        # Save checkpoint.
        if val_1 > best_acc1:
            best_acc1 = val_1
            state = {
                'net': net.state_dict(),
                'epoch': epoch+1,
                'optimizer' : optimizer.state_dict(),
            }
            if not os.path.isdir('../ckpt/checkpoint/ImageNet'):
                os.mkdir('../ckpt/checkpoint/ImageNet')
            torch.save(state, os.path.join('../ckpt/checkpoint/ImageNet', ckpt_name))
        print('best_acc@1 %.3f'%best_acc1)
        train1_acc.append(train_1)
        train5_acc.append(train_5)
        val1_acc.append(val_1)
        val5_acc.append(val_5)
    
        if not os.path.isdir('../ckpt/curve/ImageNet/top1'):
            os.mkdir('../ckpt/curve/ImageNet/top1')
        if not os.path.isdir('../ckpt/curve/ImageNet/top5'):
            os.mkdir('../ckpt/curve/ImageNet/top5')
        torch.save({'train_acc': train1_acc, 'val_acc': val1_acc},
                   os.path.join('../ckpt/curve/ImageNet/top1', ckpt_name))
        torch.save({'train_acc': train5_acc, 'val_acc': val5_acc},
                   os.path.join('../ckpt/curve/ImageNet/top5', ckpt_name))

if __name__ == '__main__':
    main()
