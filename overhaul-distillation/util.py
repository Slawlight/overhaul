import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class AverageMeter(object):
    """Computes and stores the average and current value"""
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
    dataset = args.dataset

    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std), ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std), ])
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True, download=True,
                                            transform=transform_train)
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='../data/cifar100', train=True, download=True,
                                            transform=transform_train)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=4)
    if dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, download=True,
                                           transform=transform_test)
    elif dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='../data/cifar100', train=False, download=True,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    return train_loader, test_loader


def test(net, device, data_loader, criterion):
    
    top1 = AverageMeter()
    net.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            outputs = outputs.float()
            prec1 = accuracy(outputs.data, targets)[0]
            top1.update(prec1.item(), inputs.size(0))
    return top1.avg
