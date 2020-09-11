import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy
from reid.models import resnet_feature


# class Model(nn.Module):
#     def __init__(self, num_classes=777):  # 构造函数中传入局部特征的输出通道数和类别数
#         super(Model, self).__init__()
#         self.base = resnet_feature.resnet18(pretrained=True)
#         self.classifier = nn.Linear(512, num_classes)
#         init.normal_(self.classifier.weight, std=0.001)
#         init.constant_(self.classifier.bias, 0)
#
#     def forward(self, x):
#         """
#         # Returns:
#         #   global_feat: shape [N, C]
#         #   local_feat: shape [N, H, c]
#         # """
#         # shape [N, C, H, W]
#         feat = self.base(x)
#         x = F.avg_pool2d(feat, feat.size()[2:])  # 就是取H,W做为池化的尺寸得到的是[N,C,1,1]的张量
#         # shape [N, C]
#         base_feature = x.view(x.size(0), -1)  # 将全局特征尺寸转化为[N,C]
#         x = self.classifier(base_feature)
#         return x, base_feature
class Model(nn.Module):
    def __init__(self, num_classes=777):  # 构造函数中传入局部特征的输出通道数和类别数
        super(Model, self).__init__()
        #self.pca =
        self.base = resnet_feature.resnet18(pretrained=True)
        self.classifier = nn.Linear(512, num_classes)
        init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.classifier.bias, 0)

    def getVec(self, features):  # features: NCWH
        reshaped_features = features.view(features.size(0), features.size(1), -1) \
            .permute(1, 0, 2).contiguous().view(features.size(1), -1)

        # cov=torch.from_numpy(np.corrcoef(reshaped_features.cpu().detach()))
        cov = torch.from_numpy(numpy.cov(reshaped_features.cpu().detach()))
        cov = cov.type_as(reshaped_features).cuda()
        eigval, eigvec = torch.eig(cov, eigenvectors=True)

        first_compo = eigvec[:, 0]

        return first_compo.cuda()

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
        project_map = project_map * ((maxv + minv) / (torch.abs(maxv + minv)+ numpy.exp(-12)))
        # print(torch.abs(maxv + minv).data)
        maxv = project_map.view(project_map.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        project_map = project_map / (maxv + numpy.exp(-12))
        # print(maxv.data)
        for index in range(len(project_map)):
            aaa = project_map[index].repeat(chanelnum, 1, 1)
            feat[index] = aaa
        feat = feat.detach()
        return feat

    def forward(self, x):
        """
        # Returns:
        #   global_feat: shape [N, C]
        #   local_feat: shape [N, H, c]
        # """
        # shape [N, C, H, W]
        feat = self.base(x)
        # feat1 = self.base(x)
        # feat1 = feat1.detach()
        # feat1 = feat

        # #Vec = torch.clamp(self.getVec(feat1), min=0)
        # mask = self.MaskFeature(feat1, Vec)
        # mask = mask.detach()
        # featconv = feat.mul(mask)
        # # print(mask.sum(), featconv.sum())
        #x = featconv
        x = F.avg_pool2d(feat, feat.size()[2:])  # 就是取H,W做为池化的尺寸得到的是[N,C,1,1]的张量
        #x = F.avg_pool2d(feat, feat.size()[2:])
        # shape [N, C]
        base_feature = x.view(x.size(0), -1)  # 将全局特征尺寸转化为[N,C]
        x = self.classifier(base_feature)
        return feat,x
