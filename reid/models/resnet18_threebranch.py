from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
import torch
import numpy
from reid.models import resnet_feature, VGG

__all__ = ['ResNet', 'resnet18']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
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
        # self.base = ResNet.__factory[depth](pretrained=pretrained)
        # self.basepartup = ResNet.__factory[depth](pretrained=pretrained)
        # self.basepartdown = ResNet.__factory[depth](pretrained=pretrained)
        self.base = resnet_feature.resnet18(pretrained=True)
        self.basepartup = resnet_feature.resnet18(pretrained=True)
        self.basepartdown = resnet_feature.resnet18(pretrained=True)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            # out_planes = self.base.fc.in_features

            out_planes = 512
            # Append new layers
            if self.has_embedding:
                self.featwhole = nn.Linear(out_planes, self.num_features)
                # self.featwhole_bn = nn.BatchNorm1d(self.num_features)
                self.drop = nn.Dropout(self.dropout)
                init.kaiming_normal(self.featwhole.weight, mode='fan_out')
                init.constant(self.featwhole.bias, 0)
                # init.constant(self.featwhole_bn.weight, 1)
                # init.constant(self.featwhole_bn.bias, 0)

                self.featup = nn.Linear(out_planes, self.num_features)
                # self.featup_bn = nn.BatchNorm1d(self.num_features)
                self.drop = nn.Dropout(self.dropout)
                init.kaiming_normal(self.featup.weight, mode='fan_out')
                init.constant(self.featup.bias, 0)
                # init.constant(self.featup_bn.weight, 1)
                # init.constant(self.featup_bn.bias, 0)

                self.featdown = nn.Linear(out_planes, self.num_features)
                # self.featdown_bn = nn.BatchNorm1d(self.num_features)
                self.drop = nn.Dropout(self.dropout)
                init.kaiming_normal(self.featdown.weight, mode='fan_out')
                init.constant(self.featdown.bias, 0)
                # init.constant(self.featdown_bn.weight, 1)
                # init.constant(self.featdown_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifierwhole = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifierwhole.weight, std=0.001)
                init.constant(self.classifierwhole.bias, 0)

                self.classifierup = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifierup.weight, std=0.001)
                init.constant(self.classifierup.bias, 0)

                self.classifierdown = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifierdown.weight, std=0.001)
                init.constant(self.classifierdown.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, Vec=None , output_feature=None):
        x=x.cuda()
        xwhole=x[:,:,0:224,:]
        xup = x[:, :, 224:336, :]
        xdown = x[:, :, 336:448, :]

        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            xwhole = module(xwhole)
        for name, module in self.basepartup._modules.items():
            if name == 'avgpool':
                break
            xup = module(xup)
        for name, module in self.basepartdown._modules.items():
            if name == 'avgpool':
                break
            xdown = module(xdown)

        xwhole = F.avg_pool2d(xwhole, xwhole.size()[2:])
        xwhole = xwhole.view(xwhole.size(0), -1)
        xup = F.avg_pool2d(xup, xup.size()[2:])
        xup = xup.view(xup.size(0), -1)
        xdown = F.avg_pool2d(xdown, xdown.size()[2:])
        xdown = xdown.view(xdown.size(0), -1)

        if self.has_embedding:
            xwhole = self.featwhole(xwhole)
            # xwhole = self.featwhole_bn(xwhole)
            xup = self.featup(xup)
            # xup = self.featup_bn(xup)
            xdown = self.featdown(xdown)
            # xdown = self.featdown_bn(xdown)
        if self.norm:
            xwhole = F.normalize(xwhole)
            xup = F.normalize(xup)
            xdown = F.normalize(xdown)

        if not self.training:
            xwhole = F.normalize(xwhole)
            xup = F.normalize(xup)
            xdown = F.normalize(xdown)
            y=torch.cat((xwhole,xup,xdown),1)
            # y=xwhole
            # y=xup
            # y=xdown
            return y
        elif self.has_embedding:
            xwhole = F.relu(xwhole)
            xup = F.relu(xup)
            xdown = F.relu(xdown)
        if self.dropout > 0:
            xwhole = self.drop(xwhole)
            xup = self.drop(xup)
            xdown = self.drop(xdown)
        if self.num_classes > 0:
            xwhole = self.classifierwhole(xwhole)
            xup = self.classifierup(xup)
            xdown = self.classifierdown(xdown)
        y=[]
        y.append(xwhole)
        y.append(xup)
        y.append(xdown)
        # y = xdown
        # y=torch.cat((xwhole,xup,xdown),1)
        # y=F.normalize(y)

        return y

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

