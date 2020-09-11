from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from PIL import Image
from reid import datasets
from reid import models
from reid.trainers import Trainer, CamStyleTrainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
# from reid.utils.data.Maskpreprocessor import maskPreprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
import os
import torchvision.transforms as tvt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_trans = tvt.Compose([
    tvt.Resize((224, 224)),
    tvt.ToTensor(),
    tvt.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])

def getdetection():
    model = models.create('resnet50', num_features=1024,
                          dropout=0.5, num_classes=13164)
    model = model.cuda()
    model.eval()
    checkpoint = load_checkpoint('./checkpointres50.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    label = torch.load('./renet50layer6label10class15000.pkl')
    Ic = np.zeros(1024)
    for i in range(0, 1024):
        if label[i] == 4:
            Ic[i] = 1
    Ic = torch.from_numpy(Ic)
    return model ,Ic



def get_data(detectmodel,ic,dataname, data_dir, height, width, batch_size, camstyle=0, re=0, workers=0):
    root = osp.join(data_dir, dataname)

    dataset = datasets.create(dataname, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(EPSILON=re),
    ])


    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])
    print('-------------------------------------')
    print(dataset.images_dir)
    print('-------------------------------------')
    train_loader = DataLoader(
        Preprocessor(detectmodel,ic,dataset.train, root=osp.join(dataset.train_path),
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)

    query_loader = DataLoader(
        Preprocessor(detectmodel,ic,dataset.query,
                     root=osp.join(dataset.query_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(detectmodel,ic,dataset.gallery,
                     root=osp.join(dataset.gallery_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, query_loader, gallery_loader




def main(args):
    cudnn.benchmark = True
    # Redirect print to both console and log file



    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    detectmodel, ic = getdetection()
    dataset, num_classes, train_loader, query_loader, gallery_loader = \
        get_data(detectmodel, ic, args.dataset, args.data_dir, args.height,
                 args.width, args.batch_size, args.re, args.workers)


    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes)
    model=model.cuda()
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {} "
              .format(start_epoch))


    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # base_param_ids = set(map(id, model.base.parameters()))
    # new_params = [p for p in model.parameters() if
    #               id(p) not in base_param_ids]
    # param_groups = [
    #     {'params': model.base.parameters(), 'lr_mult': 0.1},
    #     {'params': new_params, 'lr_mult': 0.1}]
    # param_groups = [
    #     {'params': new_params, 'lr_mult': 1}]

    param_groups = model.parameters()

    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    print('-----------args.camstyle-------------')
    print(args.camstyle)
    trainer = Trainer(model, criterion)


    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 15
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    #label = torch.load('/media/sdc2/wyf/DDT-master/pcalayer23.pkl')

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch , train_loader, optimizer)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
        }, fpath=osp.join('./resnet18_hd_bn_FC_three_branch.pth.tar'))

        print('\n * Finished epoch {:3d} \n'.
              format(epoch))
    torch.save(model, './resnet18_hd_bn_FC_three_branch.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CamStyle")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='VehicleID',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('--height', type=int, default=224,
                        help="input height, default: 224")
    parser.add_argument('--width', type=int, default=224,
                        help="input width, default: 224")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet18',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.4)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    #/home/weiying1/wyf/mycode/logs/checkpoint.pth.tar
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    #random erasing
    parser.add_argument('--re', type=float, default=0)
    # camstyle batchsize
    parser.add_argument('--camstyle', type=int, default=0)
    #  perform re-ranking
    parser.add_argument('--rerank', action='store_true', help="perform re-ranking")

    main(parser.parse_args())
