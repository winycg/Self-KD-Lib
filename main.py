import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np

from methods import *
import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds, correct_num, adjust_lr, DistillKL, AverageMeter
from dataloader.dataloaders import load_dataset


import time
import math

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--dataset', default='CIFAR-100', type=str, help='Dataset')
parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--arch', default='CIFAR_ResNet18', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--warmup-epoch', default=5, type=int, help='warmup epoch')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--data-aug', default='None', type=str, help='extra data augmentation')
parser.add_argument('--milestones', default=[100, 150], type=list, help='milestones for lr-multistep')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--method', default='cross_entropy', type=str, help='method')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--weight-cls', type=float, default=1, help='weight for cross-entropy loss')
parser.add_argument('--weight-kd', type=float, default=1, help='weight for KD loss')
parser.add_argument('--T', type=float, default=4, help='temperature for KD distillation')
parser.add_argument('--omega', default=0.5, type=float, help='ensembling weight in BAKE')
parser.add_argument('--intra-imgs', '-m', default=3, type=int, help='intra-class images, M in BAKE')
parser.add_argument('--alpha-T', default=0.8, type=float, help='alpha T in PS-KD')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=str, help='saved checkpoint directory')
parser.add_argument('--eval-checkpoint', default='./checkpoint/resnet18_best.pth', type=str, help='evaluate checkpoint directory')
parser.add_argument('--resume-checkpoint', default='./checkpoint/resnet18.pth', type=str, help='resume checkpoint directory')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')

# global hyperparameter set
args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


info = str(os.path.basename(__file__).split('.')[0]) \
          + '_dataset_' + args.dataset \
          + '_arch_' + args.arch \
          + '_method_' + args.method \
          + '_data_aug_' + args.data_aug \
          + '_' + str(args.manual_seed)


args.checkpoint_dir = os.path.join(args.checkpoint_dir, info)
if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
args.log_txt = os.path.join(args.checkpoint_dir, info + '.txt')

print('dir for checkpoint:', args.checkpoint_dir)
with open(args.log_txt, 'a+') as f:
    f.write("==========\nArgs:{}\n==========".format(args) + '\n')
        

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

trainloader, valloader = load_dataset(args=args)

print('Dataset: '+ args.dataset)
if args.dataset.startswith('CIFAR'):
    num_classes = len(set(trainloader.dataset.targets))
else:
    num_classes = len(set(trainloader.dataset.classes))

print('Number of train dataset: ' ,len(trainloader.dataset))
print('Number of validation dataset: ' ,len(valloader.dataset))
print('Number of classes: ' , num_classes)
C, H, W =  trainloader.dataset[0][0][0].size() if isinstance(trainloader.dataset[0][0], list) is True  else trainloader.dataset[0][0].size()
# --------------------------------------------------------------------------------------------

# Model
print('==> Building model..')
model = getattr(models, args.arch)

if args.method == 'virtual_softmax':
    net = model(num_classes=num_classes, is_bias=False).eval()
else:
    net = model(num_classes=num_classes).eval()


print('Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
    % (args.arch, cal_param_size(net) / 1e6, cal_multi_adds(net, (1, C, H, W)) / 1e9))

del (net)


if args.method == 'virtual_softmax':
    net = model(num_classes=num_classes, is_bias=False).cuda()
else:
    net = model(num_classes=num_classes).cuda()

net = torch.nn.DataParallel(net)
cudnn.benchmark = True



# Training
def train(epoch, criterion_list, optimizer):
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_div = AverageMeter('train_loss_div', ':.4e')
    top1_num = 0
    top5_num = 0
    total = 0

    if epoch >= args.warmup_epoch:
        lr = adjust_lr(optimizer, epoch, args)
    start_time = time.time()
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    if args.method.startswith('PSKD'):
        if epoch == 0:
            all_predictions = torch.zeros(len(trainloader.dataset), num_classes, dtype=torch.float32)
        else:
            all_predictions = torch.load(os.path.join(args.checkpoint_dir, 'predictions.pth.tar'), map_location=torch.device('cpu'))['prev_pred']
    
    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        batch_start_time = time.time()
        if isinstance(inputs, list) is False:
            inputs = inputs.cuda()
            batch_size = inputs.size(0)
        else:
            batch_size = inputs[0].size(0)
        
        if isinstance(targets, list) is False:
            targets = targets.cuda()
        else:
            input_indices = targets[1].cuda()
            targets = targets[0].cuda()

        if epoch < args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))

        loss_div = torch.tensor(0.).cuda()
        loss_cls = torch.tensor(0.).cuda()

        
        if args.method == 'cross_entropy':
            logit = net(inputs)
            loss_cls += criterion_cls(logit, targets)
        elif args.method == 'mixup':   
            logit, mixup_loss = Mixup(net, inputs, targets, criterion_cls, alpha=0.4)
            loss_cls += mixup_loss
        elif args.method == 'manifold_mixup':   
            logit, manifold_mixup_loss = ManifoldMixup(net, inputs, targets, criterion_cls, alpha=2.0)
            loss_cls += manifold_mixup_loss
        elif args.method == 'cutmix':   
            logit, cutmix_loss = CutMix(net, inputs, targets, criterion_cls, alpha=1.0)
            loss_cls += cutmix_loss
        elif args.method == 'label_smooth':
            logit = net(inputs)
            loss_cls += LabelSmooth(logit, targets, num_classes=num_classes)
        elif args.method == 'FocalLoss':
            logit = net(inputs) 
            loss_cls += FocalLoss(logit, targets)
        elif args.method == 'TF_KD_self_reg':
            logit = net(inputs)
            loss_cls += criterion_cls(logit, targets)
            loss_div += TF_KD_reg(logit, targets, num_classes, epsilon=0.1, T=20)
        elif args.method == 'virtual_softmax':
            logit = net(inputs, targets, loss_type='virtual_softmax')
            loss_cls += criterion_cls(logit, targets)
        elif args.method == 'Maximum_entropy':
            logit = net(inputs, targets)
            entropy = (F.softmax(logit, dim=1) * F.log_softmax(logit, dim=1)).mean()
            loss_cls += criterion_cls(logit, targets) + 0.5 * entropy

        elif args.method == 'DKS':
            logit, dks_loss_cls, dks_loss_div = DKS(net, inputs, targets, criterion_cls, criterion_div)
            loss_cls += dks_loss_cls
            loss_div += dks_loss_div

        elif args.method == 'SAD':
            logit, sad_loss_cls, sad_loss_div = SAD(net, inputs, targets, criterion_cls, criterion_div)
            loss_cls += sad_loss_cls
            loss_div += sad_loss_div
        
        elif args.method == 'BYOT':
            logit, byot_loss_cls, byot_loss_div = BYOT(net, inputs, targets, criterion_cls, criterion_div)
            loss_cls += byot_loss_cls
            loss_div += byot_loss_div

        elif args.method == 'DDGSD':
            logit, ddsgd_loss_cls, ddsgd_loss_div = DDGSD(net, inputs, targets, criterion_cls, criterion_div)
            loss_cls += ddsgd_loss_cls
            loss_div += ddsgd_loss_div

        elif args.method == 'CS-KD':
            logit, cs_kd_loss_cls, cs_kd_loss_div = CS_KD(net, inputs, targets, criterion_cls, criterion_div)
            targets = targets[:batch_size//2]
            batch_size = batch_size // 2
            loss_cls += cs_kd_loss_cls
            loss_div += cs_kd_loss_div
            
        elif args.method.startswith('FRSKD'):
            logit, frskd_loss_cls, frskd_loss_div = FRSKD(net, inputs, targets, criterion_cls, criterion_div)
            loss_cls += frskd_loss_cls
            loss_div += frskd_loss_div

        elif args.method.startswith('PSKD'):
            logit, pskd_loss_cls = PSKD(net, inputs, targets, input_indices, epoch, all_predictions, num_classes, args)
            loss_cls += pskd_loss_cls

        elif args.method.startswith('BAKE'):
            logit, bake_loss_cls, bake_loss_div = BAKE(net, inputs, targets, criterion_cls, criterion_div, args)
            loss_cls += bake_loss_cls
            loss_div += bake_loss_div
    
        else:
            raise ValueError('Unknown method: {}'.format(args.method))
        loss = loss_cls + loss_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), batch_size)
        train_loss_cls.update(loss_cls.item(), batch_size)
        train_loss_div.update(loss_div.item(), batch_size)

        top1, top5 = correct_num(logit, targets, topk=(1, 5))
        top1_num += top1
        top5_num += top5
        total += targets.size(0)
        
        print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Acc:{:.4f}, Duration:{:.2f}'.format(epoch, batch_idx, len(trainloader), lr, top1_num.item() / total, time.time()-batch_start_time))

    train_info = 'Epoch:{}\t lr:{:.5f}\t duration:{:.3f}'\
                 '\ntrain_loss:{:.5f}\t train_loss_cls:{:.5f}'\
                 '\t train_loss_div:{:.5f}' \
                 '\ntrain top1_acc: {:.4f} \t train top5_acc:{:.4f}' \
                .format(epoch, lr, time.time() - start_time,
                        train_loss.avg, train_loss_cls.avg,
                        train_loss_div.avg, (top1_num/total).item(), (top5_num/total).item())
    print(train_info)
    with open(args.log_txt, 'a+') as f:
        f.write(train_info+'\n')

    if args.method.startswith('PSKD'):
        torch.save({'prev_pred': all_predictions.cpu()}, os.path.join(args.checkpoint_dir, 'predictions.pth.tar'))



def test(epoch, criterion_list):
    test_loss = AverageMeter('test_loss', ':.4e')
    top1_num = 0
    top5_num = 0
    total = 0

    criterion_cls = criterion_list[0]
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            logit = net(inputs)
            
            if isinstance(logit, list) or isinstance(logit, tuple):
                logit = logit[0]
            loss_cls = criterion_cls(logit, targets)
            

            test_loss.update(loss_cls.item(), inputs.size(0))

            top1, top5 = correct_num(logit, targets, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += targets.size(0)

            print('Epoch:{}, batch_idx:{}/{}, Acc:{:.4f}'.format(epoch, batch_idx, len(trainloader), top1_num.item() / total))


    test_info = 'test_loss:{:.5f}\t test top1_acc:{:.4f} \t test top5_acc:{:.4f} \n' \
                .format(test_loss.avg, (top1_num/total).item(), (top5_num/total).item())
    with open(args.log_txt, 'a+') as f:
        f.write(test_info)
    print(test_info)

    return (top1_num/total).item()



if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)
    criterion_list.cuda()

    if args.evaluate:
        print('load trained weights from '+ args.eval_checkpoint)
        checkpoint = torch.load(args.eval_checkpoint_dir,
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        top1_acc = test(start_epoch, criterion_list)
    else:
        trainable_list = nn.ModuleList([])
        trainable_list.append(net)
        optimizer = optim.SGD(trainable_list.parameters(), lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

        if args.resume:
            print('Resume from '+ args.resume_checkpoint)
            checkpoint = torch.load(args.resume_checkpoint,
                                    map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']+1

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_list)

            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, model.__name__ + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, model.__name__ + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, model.__name__ + '_best.pth.tar'))

        print('Evaluate the best model:')
        args.evaluate = True
        checkpoint = torch.load(args.checkpoint_dir + '/' +  model.__name__ + '_best.pth.tar',
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_list)

        with open(args.log_txt, 'a+') as f:
            f.write('best_accuracy: {} \n'.format(best_acc))
        print('best_accuracy: {} \n'.format(best_acc))

