from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb


__all__ = ['VggNet', 'vggnet11', 'vggnet13', 'vggnet16', 'vggnet19']


class VggNet(nn.Module):
    __factory = {
        11: torchvision.models.vgg11,
        13: torchvision.models.vgg13,
        16: torchvision.models.vgg16,
        19: torchvision.models.vgg19,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(VggNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in VggNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = VggNet.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = 512

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

    def forward(self, x, convfeats=None , output_feature=None):
        conv=[]
        for name, module in self.base._modules['features']._modules.items():
            x = module(x)
            conv.append(x)
            if name == '30':
                break
        # x= self.base._modules['avgpool'](x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        # elif self.has_embedding:
        #     x = F.relu(x)
        # if self.dropout > 0:
        #     x = self.drop(x)
        # if self.num_classes > 0:
        #     x = self.classifier(x)
        x = F.normalize(x)
        return conv, x

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


def vggnet11(**kwargs):
    return VggNet(11, **kwargs)


def vggnet13(**kwargs):
    return VggNet(13, **kwargs)


def vggnet16(**kwargs):
    return VggNet(16, **kwargs)


def vggnet19(**kwargs):
    return VggNet(19, **kwargs)

