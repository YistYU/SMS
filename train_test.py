import argparse
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

import pcl.loader
import pcl.builder


from sklearn import manifold, datasets
from sklearn.cluster import KMeans
import torch.nn.functional as F

import scanpy as sc
import pandas as pd

from metrics import compute_metrics

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch scRNA-seq CLEAR Training')

# 1.input h5ad data
parser.add_argument('--input_h5ad_path', type=str, default= "",
                    help='path to input h5ad file')

parser.add_argument('--input_ATAC_h5ad_path', type=str, default= "",
                    help='path to input ATAC h5ad file')

parser.add_argument('--input_RNA_h5ad_path', type=str, default= "",
                    help='path to input RNA h5ad file')

parser.add_argument('--obs_label_colname', type=str, default= None,
                    help='column name of the label in obs')

# 2.hyper-parameters

parser.add_argument('--num_view', type=int, default= 2,
                    help='number of multi-omics views')

parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch_size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning_rate', default=5e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')

parser.add_argument('--wd', '--weight_decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--schedule', default=[100, 120], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x), if use cos, then it will not be activated')

parser.add_argument('--low_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--pcl_r', default=1024, type=int,
                    help='queue size; number of negative pairs; needs to be smaller than num_cluster (default: 16384)')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')

parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--warmup_epoch', default=5, type=int,
                    help='number of warm-up epochs to only train with InfoNCE loss')

# augmentation prob
parser.add_argument("--aug_prob", type=float, default=0.5,
                    help="The prob of doing augmentation")

# cluster
parser.add_argument('--cluster_name', default='kmeans', type=str,
                    help='name of clustering method', dest="cluster_name")

parser.add_argument('--num_cluster', default=-1, type=int,
                    help='number of clusters', dest="num_cluster")

# random
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')

# gpu
parser.add_argument('--gpu', default=0, type=int,   #None
                    help='GPU id to use.')

# logs and savings
parser.add_argument('-e', '--eval_freq', default=10, type=int,
                    metavar='N', help='Save frequency (default: 10)',
                    dest='eval_freq')

parser.add_argument('-l', '--log_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--exp_dir', default='./experiment_pcl', type=str,
                    help='experiment directory')
# CJY metric
parser.add_argument('--save_dir', default='./result', type=str,
                    help='result saving directory')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:
        # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main_worker(args):

    input_ATAC_h5ad_path = args.input_ATAC_h5ad_path
    processed_adata = sc.read_h5ad(input_ATAC_h5ad_path)
    # print(processed_adata)
    input_RNA_h5ad_path = args.input_RNA_h5ad_path
    processed_rdata = sc.read_h5ad(input_RNA_h5ad_path)
    # print(processed_rdata)
    obs_label_colname = None

    # print(obs_label_colname)

    # find dataset name
    ATAC_pre_path, ATAC_filename = os.path.split(input_ATAC_h5ad_path)
    ATAC_dataset_name, ATAC_ext = os.path.splitext(ATAC_filename)
    RNA_pre_path, RNA_filename = os.path.split(input_RNA_h5ad_path)
    RNA_dataset_name, RNA_ext = os.path.splitext(RNA_filename)

    # for batch effect dataset3
    if ATAC_dataset_name == "h5ad":
        ATAC_dataset_name = ATAC_pre_path.split("/")[-1]
        
    if ATAC_dataset_name == "":
        ATAC_dataset_name = "unknown"

    if RNA_dataset_name == "h5ad":
        RNA_dataset_name = RNA_pre_path.split("/")[-1]
    if RNA_dataset_name == "":
        RNA_dataset_name = "unknown"

    # save path
    save_path_ATAC = os.path.join(args.save_dir, "CLEAR_ATAC")
    if os.path.exists(save_path_ATAC) != True:
        os.makedirs(save_path_ATAC)
    save_path_RNA = os.path.join(args.save_dir, "CLEAR_RNA")
    if os.path.exists(save_path_RNA) != True:
        os.makedirs(save_path_RNA)

    # Define Transformation
    args_transformation = {
        # crop
        # without resize, it's better to remove crop
        
        # mask
        'mask_percentage': 0.2,
        'apply_mask_prob': args.aug_prob,
        
        # (Add) gaussian noise
        'noise_percentage': 0.8,
        'sigma': 0.2,
        'apply_noise_prob': args.aug_prob,

        # inner swap
        'swap_percentage': 0.1,
        'apply_swap_prob': args.aug_prob,
        
        # cross over with 1
        'cross_percentage': 0.25,
        'apply_cross_prob': args.aug_prob,
        
        # cross over with many
        'change_percentage': 0.25,
        'apply_mutation_prob': args.aug_prob
    }

    train_dataset_ATAC = pcl.loader.scMatrixInstance(
        adata=processed_adata,
        obs_label_colname=obs_label_colname,
        transform=True,
        args_transformation=args_transformation
        )

    train_dataset_RNA = pcl.loader.scMatrixInstance(
        adata=processed_rdata,
        obs_label_colname=obs_label_colname,
        transform=True,
        args_transformation=args_transformation
        )

    eval_dataset_ATAC = pcl.loader.scMatrixInstance(
        adata=processed_adata,
        obs_label_colname=obs_label_colname,
        transform=False
        )
    eval_dataset_RNA = pcl.loader.scMatrixInstance(
        adata=processed_rdata,
        obs_label_colname=obs_label_colname,
        transform=False
        ) 

    if train_dataset_ATAC.num_cells < 512:
        args.batch_size = train_dataset_ATAC.num_cells
        args.pcl_r = train_dataset_ATAC.num_cells
    
    if train_dataset_RNA.num_cells < 512:
        args.batch_size = train_dataset_RNA.num_cells
        args.pcl_r = train_dataset_RNA.num_cells

    train_sampler = None
    eval_sampler = None
    
    train_loader_RNA = torch.utils.data.DataLoader(
        train_dataset_RNA, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    train_loader_ATAC = torch.utils.data.DataLoader(
        train_dataset_ATAC, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
        
    # dataloader for center-cropped images, use larger batch size to increase speed

    eval_loader_ATAC = torch.utils.data.DataLoader(
        eval_dataset_ATAC, batch_size=args.batch_size * 5, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=True)
    eval_loader_RNA = torch.utils.data.DataLoader(
        eval_dataset_RNA, batch_size=args.batch_size * 5, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=True)
    
    # 2. Create Model_dict
    print("=> creating model 'Multi-omics MLP'")
    
    model_dict = {}
    num_cluster=args.num_cluster
    num_view=args.num_view
    dim_hvcdn = 9
    model_dict["E{:}".format(1)] = pcl.builder.MoCo(pcl.builder.MLPEncoder, int(train_dataset_ATAC.num_genes),args.low_dim, args.pcl_r, args.moco_m, args.temperature)
    #print(model_dict["E{:}".format(1)])
    model_dict["E{:}".format(2)] = pcl.builder.MoCo(pcl.builder.MLPEncoder, int(train_dataset_RNA.num_genes),args.low_dim, args.pcl_r, args.moco_m, args.temperature)
    #print(model_dict["E{:}".format(2)])
    model_dict["C{:}".format(1)] = pcl.builder.Classifier_1(int(train_dataset_ATAC.num_genes), num_hiddens=128, p_drop=0.0, num_clusters=3)
    model_dict["C{:}".format(2)] = pcl.builder.Classifier_1(int(train_dataset_ATAC.num_genes), num_hiddens=128, p_drop=0.0, num_clusters=3)
    model_dict["V"] = pcl.builder.VCDN(num_view=2, num_cls=3, hvcdn_dim=9)

    if num_view >= 2:
        model_dict["C"] = pcl.builder.MoCo_VCDN(pcl.builder.MLPEncoder_VCDN, num_view, num_cluster, dim_hvcdn, args.low_dim, args.pcl_r, args.moco_m, args.temperature)
    
    #print(model_dict["E{:}".format(1)])

    if args.gpu is None:
        raise Exception("Should specify GPU id for training with --gpu".format(args.gpu))
    else:
        print("Use GPU: {} for training".format(args.gpu))

    
    model_dict["E{:}".format(1)] = model_dict["E{:}".format(1)].cuda(args.gpu)
    model_dict["C{:}".format(1)] = model_dict["C{:}".format(1)].cuda(args.gpu)
    model_dict["E{:}".format(2)] = model_dict["E{:}".format(2)].cuda(args.gpu)
    model_dict["C{:}".format(2)] = model_dict["C{:}".format(2)].cuda(args.gpu)
    model_dict["C"] = model_dict["C"].cuda(args.gpu)

# define loss function (criterion) and optim_dict    
    criterion = nn.CrossEntropyLoss()   #.cuda(args.gpu)
    optim_dict = {}
    optim_dict["C{:}".format(1)] = torch.optim.SGD(list(model_dict["E{:}".format(1)].parameters())+list(model_dict["C{:}".format(1)].parameters()), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    optim_dict["C{:}".format(2)] = torch.optim.SGD(list(model_dict["E{:}".format(2)].parameters())+list(model_dict["C{:}".format(2)].parameters()), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 2. Train Encoder
    # train the model
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optim_dict["C{:}".format(1)], epoch, args)

        adjust_learning_rate(optim_dict["C{:}".format(2)], epoch, args)

        # train for one epoch
        train_unsupervised_metrics_ATAC = train(train_loader_ATAC, model_dict["E{:}".format(1)], criterion, optim_dict["C{:}".format(1)], epoch, args)
        train_unsupervised_metrics_RNA = train(train_loader_RNA, model_dict["E{:}".format(2)], criterion, optim_dict["C{:}".format(2)], epoch, args)
        
        seed = 0
        c = []
        pd_labels = []
        gt_labels = []
        # inference log & supervised metrics
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            embeddings_ATAC, gt_labels_ATAC = inference(eval_loader_ATAC, model_dict["E{:}".format(1)])
            embeddings_RNA, gt_labels_RNA = inference(eval_loader_RNA, model_dict["E{:}".format(2)])
            #print("embeddings")
            #print(embeddings)
            #print("TSNE Processing")
            tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
            embeddings_ATAC = tsne.fit_transform(embeddings_ATAC)
            embeddings_RNA = tsne.fit_transform(embeddings_RNA)
            ci_list = []
            prob_ATAC = torch.from_numpy(prob_ATAC).cuda(args.gpu)
            prob_RNA = torch.from_numpy(prob_RNA).cuda(args.gpu)
            ci_list.append(prob_ATAC)
            ci_list.append(prob_RNA)
            num_view = 2
            for i in range(num_view):
                ci_list[i] = torch.sigmoid(ci_list[i])
            x = torch.reshape(torch.matmul(ci_list[0].unsqueeze(-1), ci_list[1].unsqueeze(1)),(-1,pow(3,2),1))
            for i in range(2,num_view):
                x = torch.reshape(torch.matmul(x, ci_list[i].unsqueeze(1)),(3,i+1),1)
            vcdn_feat = torch.reshape(x, (-1,pow(3,num_view)))
            c = vcdn_feat.cpu().detach().numpy()
            print("c is here")
            print(c)
            prob = F.softmax(c, dim=1).data.cpu().numpy()
            pd_labels = prob.argmax(1)
            #print(embeddings)
            #print("embeddings after tsne")
            # perform kmeans
            gt_labels = gt_labels_ATAC
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        embeddings = tsne.fit_transform(c)
        if epoch > 0 and epoch%100 == 0 and args.cluster_name == "kmeans":
            
            data0 = []
            data1 = []
            data2 = []
            #data3 = []
            #data4 = []
            for i in range(len(gt_labels)):
                if pd_labels[i] == 0:
                    data0.append(embeddings[i])
                if pd_labels[i] == 1:
                    data1.append(embeddings[i])
                if pd_labels[i] == 2:
                    data2.append(embeddings[i])
                #if pd_labels[i] == 3:
                #    data3.append(embeddings[i])
                #if pd_labels[i] == 4:
                #    data4.append(embeddings[i])
            # Plot figure
            data0 = np.array(data0)
            data1 = np.array(data1)
            data2 = np.array(data2)
            #data3 = np.array(data3)
            #data4 = np.array(data4)
            fig = plt.figure(1)
            plt.subplot(211)
            plt.scatter(data0[:,0], data0[:,1], c='r', s=6)
            plt.scatter(data1[:,0], data1[:,1], c='b', s=6)
            plt.scatter(data2[:,0], data2[:,1], c='g', s=6)
            #plt.scatter(data3[:,0], data3[:,1], c='y', s=6)
            #plt.scatter(data4[:,0], data4[:,1], c='brown', s=6)
            plt.title("ALgorithm Classified data", fontsize = 10)
            data0 = []
            data1 = []
            data2 = []
            #data3 = []
            #data4 = []
            for i in range(len(gt_labels)):
                if gt_labels[i] == 0:
                    data0.append(embeddings[i])
                if gt_labels[i] == 1:
                    data1.append(embeddings[i])
                if gt_labels[i] == 2:
                    data2.append(embeddings[i])
                #if gt_labels[i] == 3:
                #    data3.append(embeddings[i])
                #if gt_labels[i] == 4:
                #    data4.append(embeddings[i])
            # Plot figure
            data0 = np.array(data0)
            data1 = np.array(data1)
            data2 = np.array(data2)
            #data3 = np.array(data3)
            #data4 = np.array(data4)
            plt.subplot(212)
            plt.scatter(data0[:,0], data0[:,1], c='r', s=6)
            plt.scatter(data1[:,0], data1[:,1], c='b', s=6)
            plt.scatter(data2[:,0], data2[:,1], c='g', s=6)
            #plt.scatter(data3[:,0], data3[:,1], c='y', s=6)
            #plt.scatter(data4[:,0], data4[:,1], c='brown', s=6)
            plt.title("Correct Classified data", fontsize = 10)
            plt.savefig('Result_After_Kmeans_changeData'  +'.pdf')
            plt.subplot()
            # compute metrics
            seed = 0
            best_ari, best_eval_supervised_metrics, best_pd_labels = -1, None, None
            eval_supervised_metrics = compute_metrics(gt_labels, pd_labels)
            if eval_supervised_metrics["ARI"] > best_ari:
                best_ari = eval_supervised_metrics["ARI"]
                best_eval_supervised_metrics = eval_supervised_metrics
                best_pd_labels = pd_labels
                print("Epoch: Kmeans {}\t {}\n".format(epoch, eval_supervised_metrics))



def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')   
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, index, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #import pdb; pdb.set_trace()

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
                
        # compute output
        output, target, output_proto, target_proto = model(im_q=images[0], im_k=images[1], cluster_result=None, index=index)
        
        # InfoNCE loss
        loss = criterion(output, target)   

        losses.update(loss.item(), images[0].size(0))
        acc = accuracy(output, target)[0] 
        acc_inst.update(acc[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time 
        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(i+1)

    unsupervised_metrics = {"accuracy": acc_inst.avg.item(), "loss": losses.avg}

    return unsupervised_metrics
            
def inference(eval_loader, model):
    print('Inference...')
    model.eval()
    features = []
    labels = []

    for i, (images, index, label) in enumerate(eval_loader):
        images = images.cuda()
        with torch.no_grad():
            feat = model(images, is_eval=True) 
        feat_pred = feat.data.cpu().numpy()
        label_true = label
        features.append(feat_pred)
        labels.append(label_true)
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:
        # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
