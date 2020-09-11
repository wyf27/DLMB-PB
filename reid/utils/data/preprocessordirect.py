from __future__ import absolute_import
import os.path as osp
import torch
from PIL import Image
import torchvision.transforms as tvt

part_trans = tvt.Compose([
    tvt.Resize((112, 224)),
    tvt.ToTensor(),
    tvt.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])

class Preprocessor(object):
    def __init__(self, detectmodel,ic,dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.parttransform =part_trans
        self.detectmodel = detectmodel
        self.ic = ic

    def __len__(self):
        return len(self.dataset)

    def _corpimg_(self,index):
        fname, pid ,did= self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        orgimg = Image.open(fpath).convert('RGB')
        orgimg=orgimg.resize((224,224))

        if self.transform is not None:
            img = self.transform(orgimg)
        img=img.unsqueeze(0)
        Singlefeature, _ = self.detectmodel(img)
        Singlefeature=Singlefeature[6]
        Label=self.ic.type_as(Singlefeature).cuda()
        reshaped_Singlefeature = Singlefeature.view(Singlefeature.size(0), Singlefeature.size(1), -1) \
            .permute(1, 0, 2).contiguous().view(Singlefeature.size(1), -1)
        feature_mask = torch.matmul(Label.cuda().unsqueeze(0), reshaped_Singlefeature.cuda()).view(1,
                                                                                                   Singlefeature.size(
                                                                                                       0), -1) \
            .view(Singlefeature.size(0), Singlefeature.size(2), Singlefeature.size(3))
        maxv = feature_mask.max()
        minv = feature_mask.min()

        feature_mask *= (maxv + minv) / torch.abs(maxv + minv)
        maxv = feature_mask.view(feature_mask.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        feature_mask /= maxv

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
        croppedup = orgimg.crop((LX,LY, LW,LH))


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
        croppeddown = orgimg.crop((LX,LY, LW,LH))

        return croppedup, croppeddown


    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid ,did= self.dataset[index]
        fpath = fname
        imgs=[]
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        imgup,imgdown=self._corpimg_(index)
        imgup =self.parttransform(imgup)
        imgdown = self.parttransform(imgdown)
        img=torch.cat((img,imgup,imgdown),1)

        return img, fname, pid,did

class Preprocessor2(object):
    def __init__(self,dataset, root=None, transform=None):
        super(Preprocessor2, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid