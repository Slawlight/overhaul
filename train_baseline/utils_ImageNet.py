import os
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def build_dataset(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

    ])
    
    trainset = torchvision.datasets.ImageFolder(root='../data/ImageNet/train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=True)
    valset = torchvision.datasets.ImageFolder(root='../data/ImageNet/val', transform=transform_val)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=256, shuffle=False, num_workers=32, pin_memory=True)

    return train_loader, val_loader


def train(net, device, train_loader, optimizer, criterion):

    net.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter() 
    time1 = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):  
     
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
    return (top1.avg, top5.avg)

def test(net, device, val_loader, criterion):
   
    net.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
    return (top1.avg, top5.avg)


