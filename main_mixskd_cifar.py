import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


import os
import shutil
import argparse
import numpy as np


import models
import methods
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds, correct_num, adjust_lr, DistillKL, AverageMeter

from dataloader.dataloaders import load_dataset
from bisect import bisect_right
import time
import math


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='CIFAR-100', type=str, help='Dataset name')
parser.add_argument('--arch', default='MixSKD_CIFAR_ResNet18', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--data-aug', default='None', type=str, help='extra data augmentation')
parser.add_argument('--method', default='cross_entropy', type=str, help='method')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[100,150], type=int, nargs='+')
parser.add_argument('--warmup-epoch', default=5, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=205, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='number workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--mixup-alpha', default=0.4, type=float, help='weight decay')
parser.add_argument('--kd_T', type=float, default=3, help='temperature for KD distillation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=str, help='checkpoint dir')
parser.add_argument('--eval-checkpoint', default='./checkpoint/', type=str, help='checkpoint dir')


# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


args.log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'arch'+ '_' + args.arch + '_'+\
          'dataset' + '_' +  args.dataset + '_'+\
          'seed'+ str(args.manual_seed)


args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.log_dir)
if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

log_txt = str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'arch' + '_' +  args.arch + '_'+\
          'dataset' + '_' +  args.dataset + '_'+\
          'kd_T'+ '_' +  str(args.kd_T) + '_'+\
          'mixup_alpha' + '_' +  str(args.mixup_alpha) + '_'+\
          'seed'+ str(args.manual_seed) +'.txt'

log_txt = os.path.join(args.checkpoint_dir, log_txt)

with open(log_txt, 'a+') as f:
    f.write("==========\nArgs:{}\n==========".format(args) + '\n')

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
torch.set_printoptions(precision=4)
# -----------------------------------------------------------------------------------------

trainloader, testloader = load_dataset(args)
print('Dataset: '+ args.dataset)

if args.dataset.startswith('CIFAR'):
    num_classes = len(set(trainloader.dataset.targets))
else:
    num_classes = len(set(trainloader.dataset.classes))

print('Number of train dataset: ' ,len(trainloader.dataset))
print('Number of validation dataset: ' ,len(testloader.dataset))
print('Number of classes: ' , num_classes)
C, H, W =  trainloader.dataset[0][0][0].size() if isinstance(trainloader.dataset[0][0], list) is True  else trainloader.dataset[0][0].size()
# --------------------------------------------------------------------------------------------

# Model
print('==> Building model..')
model = getattr(models, args.arch)
net = model(num_classes=num_classes)
net.eval()

resolution = (2, C, H, W)
print('Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
        % (args.arch, cal_param_size(net)/1e6, cal_multi_adds(net, resolution)/1e9))
del(net)


net = model(num_classes=num_classes).cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = True


def mixup_data(x, y,  alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam, index

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch, criterion_list, optimizer):
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_mixcls = AverageMeter('train_loss_mixcls', ':.4e')
    train_loss_div = AverageMeter('train_loss_div', ':.4e')
    train_loss_fea = AverageMeter('train_loss_fea', ':.4e')
    train_loss_gan = AverageMeter('train_loss_gan', ':.4e')

    global num_stage
    top1_num = [0 for stage in range(num_stage)]
    top5_num = [0 for stage in range(num_stage)]
    total = 0

    if epoch >= args.warmup_epoch:
        lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_gan = criterion_list[2]

    net.train()
    for batch_idx, (input, target) in enumerate(trainloader):
        batch_start_time = time.time()
        inputs = input.float().cuda()
        targets = target.cuda()

        bs = inputs.size(0)
        mix_inputs, targets_a, targets_b, lam, index = mixup_data(inputs, targets, alpha=args.mixup_alpha)
        all_inputs = torch.cat([inputs, mix_inputs], dim=0)

            
        if epoch < args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))

        optimizer.zero_grad()
        logits, features, ensemle_logits, ensemble_mixup_logits = net(all_inputs, lam, index)

        mix_logits = [lam * logit[:bs] + (1 - lam) * logit[index] for logit in logits]
        ensemble_features = [lam * (fea[:bs]) + (1 - lam) * (fea[index]) for fea in features]

        mixup_logits = [logit[bs:] for logit in logits]
        mixup_features = [fea[bs:] for fea in features]
        
        loss_cls = torch.tensor(0.).cuda()
        loss_mixcls = torch.tensor(0.).cuda()
        for logit in logits:
            loss_cls += criterion_cls(logit[:bs], targets)
            
        for mixup_logit in mixup_logits:
            loss_mixcls += mixup_criterion(criterion_cls, mixup_logit, targets_a, targets_b, lam)
        loss_mixcls += mixup_criterion(criterion_cls, ensemle_logits, targets_a, targets_b, lam)
        loss_mixcls += mixup_criterion(criterion_cls, ensemble_mixup_logits, targets_a, targets_b, lam)
        
        loss_div = torch.tensor(0.).cuda()
        loss_div += criterion_div(mix_logits[-1], ensemble_mixup_logits.detach())
        loss_div += criterion_div(mixup_logits[-1], ensemle_logits.detach())

        for i in range(len(mix_logits)-1):
            loss_div += criterion_div(mix_logits[i], mixup_logits[i].detach())
            loss_div += criterion_div(mixup_logits[i], mix_logits[i].detach())


        loss_fea = torch.tensor(0.).cuda()
        for fi in range(len(features)):
            loss_fea += F.mse_loss(ensemble_features[fi], mixup_features[fi])
        loss_gan = criterion_gan(ensemble_features, mixup_features)

        loss = loss_cls + loss_div + loss_fea + loss_gan
        loss.backward()
        optimizer.step()

        train_loss.update(loss_cls.item(), input.size(0))
        train_loss_cls.update(loss_cls.item(), input.size(0))
        train_loss_mixcls.update(loss_mixcls.item(), input.size(0))
        train_loss_div.update(loss_div.item(), input.size(0))
        train_loss_fea.update(loss_fea.item(), input.size(0))
        train_loss_gan.update(loss_gan.item(), input.size(0))

        for i in range(num_stage):
            top1, top5 = correct_num(logits[i][:bs], targets, topk=(1, 5))
            top1_num[i] += top1
            top5_num[i] += top5
        total += target.size(0)

        print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Top-1 Acc:{:.4f}'
              '\ttrain_loss_cls:{:.5f}\ttrain_loss_mixcls:{:.5f}\ttrain_loss_div:{:.5f}'
              '\ttrain_loss_fea:{:.5f}\t train_loss_gan:{:.5f}'.format(
            epoch, batch_idx, len(trainloader), lr, time.time()-batch_start_time, (top1_num[-1]/(total)).item(),
            train_loss_cls.avg, train_loss_mixcls.avg, train_loss_div.avg,
            train_loss_fea.avg, train_loss_gan.avg,))

    acc1 = [round((top1_num[i]/total).item(), 4) for i in range(num_stage)]
    with open(log_txt, 'a+') as f:
        f.write('Epoch:{}\t lr:{:.4f}\t duration:{:.3f}'
                '\ttrain_loss_cls:{:.5f}\ttrain_loss_mixcls:{:.5f}\ttrain_loss_div:{:.5f}'
                '\ttrain_loss_fea:{:.5f}\t train_loss_gan:{:.5f}'
                '\nTrain Top-1 accuracy: {} \n'
                .format(epoch, lr, time.time() - start_time,
                        train_loss_cls.avg, train_loss_mixcls.avg, train_loss_div.avg,
                        train_loss_fea.avg, train_loss_gan.avg,
                        str(acc1)))


def test(epoch, criterion_cls):
    net.eval()
    global best_acc
    test_loss_cls = AverageMeter('train_loss_cls', ':.4e')

    global num_stage
    top1_num = [0 for stage in range(num_stage)]
    top5_num = [0 for stage in range(num_stage)]
    total = 0
    

    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(testloader):
            batch_start_time = time.time()
            inputs, targets = inputs.cuda(), target.cuda()

            logits  = net(inputs)
            if isinstance(logits, tuple) or isinstance(logits, list):
                logits = logits[0]

            loss_cls = torch.tensor(0.).cuda()
            for logit in logits:
                loss_cls += criterion_cls(logit, targets)
            test_loss_cls.update(loss_cls.item(), inputs.size(0))

            for i in range(num_stage):
                top1, top5 = correct_num(logits[i], targets, topk=(1, 5))
                top1_num[i] += top1
                top5_num[i] += top5
            total += target.size(0)

            print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                epoch, batch_idx, len(testloader), time.time()-batch_start_time, (top1_num[-1]/(total)).item()))

        acc1 = [round((top1_num[i]/total).item(), 4) for i in range(num_stage)]

        with open(log_txt, 'a+') as f:
            f.write('test epoch:{}\t test_loss_cls:{:.5f}\t Top-1 accuracy:{}\n'
                    .format(epoch, test_loss_cls.avg, str(acc1)))
        print('test epoch:{}\t accuracy:{}\n'.format(epoch, str(acc1)))

    return max(acc1)


if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)

    net.eval()
    x = torch.randn(2, 3, 32, 32).cuda()
    logits, features = net(x)
    dim_ins = [fea.size(1) for fea in features]
    num_stage = len(dim_ins)
    if args.evaluate: 
        print('load pre-trained weights from: {}'.format(args.eval_checkpoint))     
        checkpoint = torch.load(args.eval_checkpoint,
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        test(start_epoch, criterion_cls)
    else:
        criterion_gan = getattr(methods, 'discriminatorLoss')(dim_ins).cuda()
        trainable_list = nn.ModuleList([])
        trainable_list.append(net)
        trainable_list.append(criterion_gan)
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
        criterion_list.append(criterion_gan)  # KL divergence loss, original knowledge distillation
        criterion_list.cuda()

        if args.resume:
            print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                    map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_cls)

            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

        print('Evaluate the best model:')
        print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_cls)

        with open(log_txt, 'a+') as f:
            f.write('best_accuracy: {} \n'.format(best_acc))
        print('best_accuracy: {} \n'.format(best_acc))