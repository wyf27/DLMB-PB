import torch.nn as nn
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn import manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
# from sklearn.preprocessing import Imputer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class LabelGenNet(nn.Module):
    def __init__(self):
        super(LabelGenNet, self).__init__()

    def labelclassier(self, features):

        reshaped_features = features.view(features.size(0), features.size(1), -1) \
            .permute(1, 0, 2).contiguous().view(features.size(1), -1)

        cov = torch.from_numpy(np.cov(reshaped_features.cpu().detach()))
        cov = cov.type_as(reshaped_features).cuda()

        # covcof = torch.from_numpy(np.corrcoef(reshaped_features.cpu().detach()))
        # covcof = covcof.type_as(reshaped_features).cuda()
        # Xcof=covcof.cpu().detach().numpy()
        # Xcof = Imputer().fit_transform(Xcof)

        X = cov.cpu().detach().numpy()
        label = SpectralClustering(n_clusters=10).fit_predict(X)
        torch.save(label, './renet50layer7label10class.pkl')

    def forward(self, features):     # features: NCWH
        self.labelclassier(features)

if __name__ == '__main__':
    lfn = LabelGenNet()

    allF = torch.load('./renet50vidlayer7.pkl')
    lfn(allF)

