from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import torch
from torch.utils.data import DataLoader
from reid import datasets
from reid import models
from reid.utils.data import transforms as T
import torch.nn.functional as F
import cv2
from PIL import Image
#from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.preprocessor import Preprocessor2
from reid.utils.serialization import load_checkpoint, save_checkpoint
import numpy as np
import torchvision.transforms as tvt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_trans = tvt.Compose([
    tvt.Resize((224, 224)),
    tvt.ToTensor(),
    tvt.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])

def get_data(dataname, data_dir, height, width, batch_size, camstyle=0, re=0, workers=2):
    root = osp.join(data_dir, dataname)

    dataset = datasets.create(dataname, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids


    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])
    print('-------------------------------------')
    print(dataset.images_dir)
    print('-------------------------------------')
    #train_loader = DataLoader(
    #    Preprocessor(dataset.train, root=osp.join(dataset.train_path),
    #                 transform=train_transformer), num_workers=workers,
    #    batch_sampler=BalancedBatchSampler(dataset,20,5), pin_memory=True)
    train_loader = DataLoader(
        Preprocessor2(dataset.train, root=osp.join(dataset.train_path),
                     transform=test_transformer),
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

def SingleF(Label, Singlefeature,Sname):

    reshaped_Singlefeature = Singlefeature.view(Singlefeature.size(0), Singlefeature.size(1), -1) \
     .permute(1, 0, 2).contiguous().view(Singlefeature.size(1), -1)
    feature_mask = torch.matmul(Label.cuda().unsqueeze(0), reshaped_Singlefeature.cuda()).view(1, Singlefeature.size(0), -1) \
        .view(Singlefeature.size(0), Singlefeature.size(2), Singlefeature.size(3))
    maxv = feature_mask.max()
    minv = feature_mask.min()

    feature_mask *= (maxv + minv) / torch.abs(maxv + minv)
    maxv = feature_mask.view(feature_mask.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
    feature_mask /= maxv
    # original_image = cv2.resize(cv2.imread('E:/VehicleID_V1.0/VehicleID_V1.0/image/' + Sname), (224, 224))
    original_image = Image.open('E:/VehicleID_V1.0/VehicleID_V1.0/image/' + Sname).convert('RGB')
    original_image=original_image.resize((224, 224))


    mapdiv = 1.0
    direct = 0
    feature_maskup = feature_mask[0, 0:7, :]
    allmapsum = feature_maskup.sum()
    bestx1, besty1, bestx2, besty2 = 0, 0, 0, 0
    maxvaluearea = 0
    x1 = 0
    y1 = 0
    x2 = 14
    y2 = 7
    while (mapdiv > 0.85):
        newarea = feature_maskup[y1:y2, x1:x2]
        newsum = newarea.sum()
        mapdiv = newsum / allmapsum
        valuearea = newsum / (x2 - x1) * (y2 - y1)
        if maxvaluearea < valuearea:
            bestx1, besty1, bestx2, besty2 = x1, y1, x2, y2
            maxvaluearea = valuearea
        if direct == 4:
            direct = 0

        if direct == 0:
            x1 = x1 + 1
        elif direct == 1:
            y1 = y1 + 1
        elif direct == 2:
            x2 = x2 - 1
        elif direct == 3:
            y2 = y2 - 1
        direct = direct + 1
    LY = besty1 * 16
    LH = besty2 * 16
    LX = bestx1 * 16
    LW = bestx2 * 16
    croppedup = original_image.crop((LX, LY, LW, LH))
    # croppedup = original_image.crop((LY, LX, LH, LW))
    # croppedup = original_image[LY:LH, LX:LW]

    mapdiv = 1.0
    direct = 0
    feature_maskdown = feature_mask[0, 7:14, :]
    allmapsum = feature_maskdown.sum()
    bestx1, besty1, bestx2, besty2 = 0, 0, 0, 0
    maxvaluearea = 0
    x1 = 0
    y1 = 0
    x2 = 14
    y2 = 7
    while (mapdiv > 0.8):
        newarea = feature_maskdown[y1:y2, x1:x2]
        newsum = newarea.sum()
        mapdiv = newsum / allmapsum

        valuearea = newsum / (x2 - x1) * (y2 - y1)
        if maxvaluearea < valuearea:
            bestx1, besty1, bestx2, besty2 = x1, y1, x2, y2
            maxvaluearea = valuearea
        if direct == 4:
            direct = 0

        if direct == 0:
            x1 = x1 + 1
        elif direct == 1:
            y1 = y1 + 1
        elif direct == 2:
            x2 = x2 - 1
        elif direct == 3:
            y2 = y2 - 1
        direct = direct + 1
    LY = 112 + besty1 * 16
    LH = 112 + besty2 * 16
    LX = bestx1 * 16
    LW = bestx2 * 16
    croppeddown = original_image.crop((LX, LY, LW, LH))
    # croppeddown = original_image.crop((LY, LX, LH, LW))
    # croppeddown = original_image[LY:LH, LX:LW]
    croppedup.save('./up/' + Sname)
    croppeddown.save('./down/' + Sname)
    original_image.save('./' + Sname)
    # cv2.imwrite('./up/' + Sname, croppedup)
    # cv2.imwrite('./down/' + Sname, croppeddown)
    return croppedup,croppeddown
    # cv2.imwrite('/media/sdc2/wyf/DDT-master/subsetimagetest/oneSresultMask' + Sname, mask)
    # cv2.imwrite('/media/sdc2/wyf/DDT-master/subsetimagetest/oneSresult'+Sname, save_img)
    # maskheat = project_map[0].permute(1, 2, 0).detach().cpu().numpy()
    # maskheat = cv2.applyColorMap(maskheat.astype(np.uint8), cv2.COLORMAP_JET)
    # save_imgheat = cv2.addWeighted(orgimg, 0.5, maskheat, 0.5, 0.0)
    # cv2.imwrite('/media/sdc2/wyf/DDT-master/subsetimagetest/save_imgheat'+Sname, save_imgheat)

def MakeOne(Pec, Singlefeatures, namelist):
    for i in range(0 , len(namelist)):
        save_imgs=[]
        Sname = namelist[i]
        Singlefeature=Singlefeatures[i,:,:,:]
        SingleF(Pec, Singlefeature.unsqueeze(0), Sname)

def main(args):
    dataset, num_classes, train_loader, query_loader, gallery_loader = \
        get_data(args.dataset, args.data_dir, args.height,
                 args.width, args.batch_size, args.re, args.workers)
    model = models.create('resnet50', num_features=1024,
                          dropout=0.5, num_classes=13164)
    model = model.cuda()
    checkpoint = load_checkpoint('./checkpointres50.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    label=torch.load('./renet50layer6label10class15000.pkl')
    Ic = np.zeros(1024)
    for i in range(0, 1024):
        if label[i] == 4:
            Ic[i] = 1
    Ic = torch.from_numpy(Ic)

    for batch_idx, (imgs, Fname, pids) in enumerate(train_loader):
        imgs = imgs.cuda()
        Singlefeature, _ = model(imgs)
        LabelVec = Ic.type_as(Singlefeature[6]).cuda()
        MakeOne(LabelVec, Singlefeature[6], Fname)
        print(str(batch_idx))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CamStyle")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='VehicleID',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=8)
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