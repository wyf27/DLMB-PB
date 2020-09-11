from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import torch
from torch.utils.data import DataLoader
from reid import datasets
from reid import models
from reid.utils.data import transforms as T
#from reid.utils.data.preprocessor776 import Preprocessor
from reid.utils.data.preprocessor import Preprocessor2
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from torch.nn import functional as F
import torchvision.transforms as tvt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_trans = tvt.Compose([
    tvt.Resize((224, 224)),
    tvt.ToTensor(),
    tvt.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])

def get_data(dataname, data_dir, height, width, batch_size, re=0, workers=2):
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
        Preprocessor2(dataset.train, root=osp.join(dataset.train_path),
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)
    query_loader = DataLoader(
        Preprocessor2(dataset.query,
                     root=osp.join(dataset.query_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor2(dataset.gallery,
                     root=osp.join(dataset.gallery_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, query_loader, gallery_loader

def main(args):
    dataset, num_classes, train_loader, query_loader, gallery_loader = \
        get_data(args.dataset, args.data_dir, args.height,
                 args.width, args.batch_size, args.re, args.workers)
    model = models.create('resnet50', num_features=1024,
                          dropout=0.5, num_classes=13164)
    model = model.cuda()
    checkpoint = load_checkpoint('./checkpointres50.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    for batch_idx, (imgs, _, pids) in enumerate(train_loader):
        imgs = imgs.cuda()
        featurelist, _ = model(imgs)
        all = featurelist[6].data.cpu()
        all= F.normalize(all)
        break
    for batch_idx, (imgs, _, pids) in enumerate(train_loader):
        imgs = imgs.cuda()
        featurelist, _ = model(imgs)
        features = featurelist[6].data.cpu()
        features=F.normalize(features)
        all = torch.cat((all, features), 0)
        print(batch_idx)
        if batch_idx == 100:
            torch.save(all, './renet50vidlayer6.pkl')
            break
    torch.save(all, './renet50vidlayer6.pkl')
    print('done!')






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CamStyle")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='VehicleID',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=224,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=224,
                        help="input width, default: 128")
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
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