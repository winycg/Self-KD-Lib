import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import os
import shutil
import argparse 
import numpy as np


import models
import methods
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds


from bisect import bisect_right
import time
import math



parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='imagenet', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet34_imagenet_aux', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='learning rate')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[30,60,90], type=int, nargs='+', help='milestones for lr-multistep')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='the number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--kd_T', type=float, default=3, help='temperature for KD distillation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resume-checkpoint',  type=str, default='./checkpoint/XXX.pth.tar')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--mixup-alpha', default=0.1, type=float, help='learning rate')
parser.add_argument('--freezed', action='store_true', help='freezing backbone')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='directory for storing checkpoint files')
parser.add_argument('--pretrained-backbone', default='./pretrained_models/resnet34.pth', type=str, help='pretrained weights of backbone')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')                    


def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    args.log_txt =  str(os.path.basename(__file__).split('.')[0]) + '_'+\
            'arch' + '_' +  args.arch + '_'+\
            'dataset' + '_' +  args.dataset + '_'+\
            'mixup_alpha'+ '_' +  str(args.mixup_alpha) + '_'+\
            'kd_T'+ '_' +  str(args.kd_T) + '_'+\
            'seed'+ str(args.manual_seed) +'.txt'


    args.log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
            'arch'+ '_' + args.arch + '_'+\
            'dataset' + '_' +  args.dataset + '_'+\
            'seed'+ str(args.manual_seed)


    args.traindir = os.path.join(args.data, 'train')
    args.valdir = os.path.join(args.data, 'val')

    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.set_printoptions(precision=4)

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.log_dir)
    args.log_txt = os.path.join(args.checkpoint_dir, args.log_txt)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    with open(args.log_txt, 'a+') as f:
        f.write("==========\nArgs:{}\n==========".format(args) + '\n')
    
    print('==> Building model..')
    num_classes = 1000

    net = getattr(models, args.arch)(num_classes=num_classes)
    net.eval()
    print('Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
        % (args.arch, cal_param_size(net)/1e6, cal_multi_adds(net, (2, 3, 224, 224))/1e9))

    del(net)
    
    args.distributed = args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node', ngpus_per_node)
    args.world_size = ngpus_per_node * args.world_size
    print('multiprocessing_distributed')
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                        world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)

    num_classes = 1000
    model = getattr(models, args.arch)
    net = model(num_classes=num_classes).cuda(args.gpu)

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    cudnn.benchmark = True
    
    net.eval()
    x = torch.randn(2, 3, 224, 224).cuda()
    logits, features = net(x)
    dim_ins = [fea.size(1) for fea in features]
    num_stage = len(dim_ins)

    criterion_cls = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_div = DistillKL(args.kd_T).cuda(args.gpu)
    criterion_gan = getattr(methods, 'discriminatorLoss')(dim_ins).cuda(args.gpu)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls) 
    criterion_list.append(criterion_div) 
    criterion_list.append(criterion_gan)
    criterion_list.cuda()

    trainable_list = nn.ModuleList([])
    trainable_list.append(net)
    trainable_list.append(criterion_gan)
    

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=0.1, momentum=0.9, weight_decay=args.weight_decay)


    if args.resume:
        print('load pre-trained weights from: {}'.format(args.resume_checkpoint))     
        checkpoint = torch.load(args.resume_checkpoint,
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        print('resume successful!')

    train_set = torchvision.datasets.ImageFolder(
    args.traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

    test_set = torchvision.datasets.ImageFolder(
        args.valdir, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ]))

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    def mixup_data(x, y, args=None, alpha=0.2):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 0.

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
        train_loss_div = AverageMeter('train_loss_div', ':.4e')
        train_loss_fea = AverageMeter('train_loss_fea', ':.4e')
        train_loss_gan = AverageMeter('train_loss_gan', ':.4e')

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
            for logit in logits:
                loss_cls += criterion_cls(logit[:bs], targets)
            for mixup_logit in mixup_logits:
                loss_cls += mixup_criterion(criterion_cls, mixup_logit, targets_a, targets_b, lam)
            loss_cls += mixup_criterion(criterion_cls, ensemle_logits, targets_a, targets_b, lam)
            loss_cls += mixup_criterion(criterion_cls, ensemble_mixup_logits, targets_a, targets_b, lam)
            
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

            loss = loss_cls + loss_div + loss_fea+ loss_gan
            loss.backward()
            optimizer.step()

            train_loss.update(loss_cls.item(), input.size(0))
            train_loss_cls.update(loss_cls.item(), input.size(0))
            train_loss_div.update(loss_div.item(), input.size(0))
            train_loss_fea.update(loss_fea.item(), input.size(0))
            train_loss_gan.update(loss_gan.item(), input.size(0))

            for i in range(num_stage):
                top1, top5 = correct_num(logits[i][:bs], targets, topk=(1, 5))
                top1_num[i] += top1
                top5_num[i] += top5
            total += target.size(0)

            if args.rank == 0:
                print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                    epoch, batch_idx, len(trainloader), lr, time.time()-batch_start_time, (top1_num[-1]/(total)).item()))

        acc1 = [round((top1_num[i]/total).item(), 4) for i in range(num_stage)]
        acc5 = [round((top5_num[i]/total).item(), 4) for i in range(num_stage)]

        
        if args.rank == 0:
            with open(args.log_txt, 'a+') as f:
                f.write('Epoch:{}\t lr:{:.4f}\t duration:{:.3f}'
                        '\ntrain_loss:{:.5f}\t train_loss_cls:{:.5f}\t train_loss_div:{:.5f}'
                        '\t train_loss_fea:{:.5f}\t train_loss_gan:{:.5f}'
                        '\nTrain Top-1 accuracy: {}'
                        '\nTrain Top-5 accuracy: {}'
                        .format(epoch, lr, time.time() - start_time,
                                train_loss.avg, train_loss_cls.avg, train_loss_div.avg,
                                train_loss_fea.avg, train_loss_gan.avg,
                                str(acc1), str(acc5)))


    def test(epoch, criterion_cls):
        net.eval()
        test_loss_cls = AverageMeter('train_loss_cls', ':.4e')
        top1_num = [0 for stage in range(num_stage)]
        top5_num = [0 for stage in range(num_stage)]
        total = 0
        

        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(testloader):
                batch_start_time = time.time()
                inputs, targets = inputs.cuda(), target.cuda()
                logits, features = net(inputs)

                loss_cls = torch.tensor(0.).cuda()
                for logit in logits:
                    loss_cls += criterion_cls(logit, targets)
                test_loss_cls.update(loss_cls.item(), inputs.size(0))

                for i in range(num_stage):
                    top1, top5 = correct_num(logits[i], targets, topk=(1, 5))
                    top1_num[i] += top1
                    top5_num[i] += top5
                total += target.size(0)
                if args.rank == 0:
                    print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                        epoch, batch_idx, len(testloader), time.time()-batch_start_time, (top1_num[-1]/(total)).item()))

            acc1 = [round((top1_num[i]/total).item(), 4) for i in range(num_stage)]
            acc5 = [round((top5_num[i]/total).item(), 4) for i in range(num_stage)]

            if args.rank == 0:
                with open(args.log_txt, 'a+') as f:
                    f.write('test epoch:{}\t test_loss_cls:{:.5f}\t Top-1 accuracy:{}\n Top-5 accuracy:{}\n'
                            .format(epoch, test_loss_cls.avg, str(acc1), str(acc5)))
                print('test epoch:{}\t accuracy:{}\n'.format(epoch, str(acc1)))

        return max(acc1)

    if args.evaluate: 
        print('load pre-trained weights from: {}'.format(args.resume_checkpoint))     
        checkpoint = torch.load(args.resume_checkpoint,
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        
        test(start_epoch, criterion_cls, net)
    else:
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_cls)

            if args.rank == 0:
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
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_cls, net)

        with open(args.log_txt, 'a+') as f:
            f.write('best_accuracy: {} \n'.format(best_acc))
        print('best_accuracy: {} \n'.format(best_acc))


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss


def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k)
    return res


def adjust_lr(optimizer, epoch, args, step=0, all_iters_per_epoch=0):
    cur_lr = 0.
    if epoch < args.warmup_epoch:
        cur_lr = args.init_lr * float(1 + step + epoch*all_iters_per_epoch)/(args.warmup_epoch *all_iters_per_epoch)
    else:
        epoch = epoch - args.warmup_epoch
        if args.lr_type == 'multistep':
            cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)
        elif args.lr_type == 'cosine':
            cur_lr = args.init_lr * 0.5 * (1. + math.cos(math.pi * epoch / (args.epochs - args.warmup_epoch)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

    return cur_lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    main()