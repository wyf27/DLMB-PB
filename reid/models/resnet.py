from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
import torch
import numpy

__all__ = ['ResNet',  'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()
    def MaskFeature(self, CFeat, Vec):
        Vec = Vec.cuda()
        feat = CFeat.detach()
        chanelnum = feat.size(1)
        reshaped_Singlefeature = feat.view(feat.size(0), feat.size(1), -1) \
            .permute(1, 0, 2).contiguous().view(feat.size(1), -1)
        project_map = torch.matmul(Vec.unsqueeze(0), reshaped_Singlefeature).view(1, feat.size(0), -1) \
            .view(feat.size(0), feat.size(2), feat.size(3))

        maxv = project_map.max()
        minv = project_map.min()
        project_map = project_map * ((maxv + minv) / torch.abs(maxv + minv))
        # print(torch.abs(maxv + minv).data)
        maxv = project_map.view(project_map.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        project_map = project_map / (maxv + numpy.exp(-12))
        # print(maxv.data)
        for index in range(len(project_map)):
            aaa = project_map[index].repeat(chanelnum, 1, 1)
            feat[index] = aaa
        feat = feat.detach()
        return feat

    def forward(self, x, Vec=None , output_feature=None):
        x=x.cuda()
        # x = x.unsqueeze(0)
        conv = []
        featconv=x

        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            featconv=module(featconv)
            x = module(x)
            conv.append(x)
        # mask = self.MaskFeature(x, Vec)
        # mask = mask.detach()
        # featconv = featconv.mul(mask)
        x = featconv
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if output_feature == 'pool5':
            x = F.normalize(x)
            return x
        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return conv,x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
