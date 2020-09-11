import os

from PIL import Image
import torchvision.transforms as tvt
import torch
from sklearn import manifold
import torchvision.utils as tvu
import torch.nn.functional as F
import cv2
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from reid import models
from torch import nn
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.data import transforms as T
from torch.autograd import Variable
from reid.utils import to_torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_trans = tvt.Compose([
    tvt.Resize((224, 224)),
    tvt.ToTensor(),
    tvt.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])


def getVec(allfeats=None):  # features: NCWH


    allfeaturesChange = allfeats.permute(1, 0, 2, 3)
    allreshaped_features = allfeaturesChange.reshape(allfeaturesChange.size(0),
                                                     allfeaturesChange.size(1) * allfeaturesChange.size(
                                                         2) * allfeaturesChange.size(3))


    cov = torch.from_numpy(np.cov(allreshaped_features.cpu().detach()))
    cov = cov.type_as(allreshaped_features).cuda()

    eigval, eigvec = torch.eig(cov, eigenvectors=True)

    first_compo = eigvec[:, 0]

    return first_compo


def getPmap(features, first_compo):

    featuresChange=features.permute(1,0,2,3)
    reshaped_features=featuresChange.reshape(featuresChange.size(0),featuresChange.size(1)*featuresChange.size(2)*featuresChange.size(3))

    projected_map = torch.matmul(first_compo.cuda().unsqueeze(0), reshaped_features.cuda()).view(1,
                                                                                                features.size(0),
                                                                                                -1) \
        .view(features.size(0), features.size(2), features.size(3))

    maxv = projected_map.max()
    minv = projected_map.min()

    projected_map *= (maxv + minv) / torch.abs(maxv + minv)

    return projected_map

def getcood(features,BIGfeat,u,d,ur,ul,dr,dm,dl,theh):
    s1=0
    s2=14
    s3=28
    s4=8
    s5=20
    if u == 0:
        project_mapup = torch.zeros(features[:, :, s1:s2, :].size(0), features[:, :, s1:s2, :].size(2),
                                    features[:, :, s1:s2, :].size(3)).cuda()
    elif u == 1:
        # project_mapup = torch.clamp(pca(features[:, :, s1:s2, :], allfeats=BIGfeat[:, :, s1:s2, :], updown='up'), min=0)
        VecUp=getVec(allfeats=BIGfeat[:, :, s1:s2, :])
        torch.save(VecUp,'./VecUp.pkl')
        VecUp=torch.load('./VecUp.pkl')
        project_mapup=torch.clamp(getPmap(features[:, :, s1:s2, :],VecUp), min=0)
        maxv = project_mapup.view(project_mapup.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        project_mapup /= maxv

    if d == 0:
        project_mapdown = torch.zeros(features[:, :, s2:s3, :].size(0), features[:, :, s2:s3, :].size(2),
                                      features[:, :, s2:s3, :].size(3)).cuda()
    elif d == 1:
        # project_mapdown = torch.clamp(pca(features[:, :, s2:s3, :], allfeats=BIGfeat[:, :, s2:s3, :], updown='down'),
        #                               min=0)
        VecDown=getVec(allfeats=BIGfeat[:, :, s2:s3, :])
        torch.save(VecDown, './VecDown.pkl')
        VecDown = torch.load('./VecDown.pkl')
        project_mapdown=torch.clamp(getPmap(features[:, :, s2:s3, :],VecDown), min=0)
        maxv = project_mapdown.view(project_mapdown.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        project_mapdown /= maxv

    if ur == 0:
        project_mapup1 = torch.zeros(features[:, :, s1:s2, s1:s2].size(0), features[:, :, s1:s2, s1:s2].size(2),
                                     features[:, :, s1:s2, s1:s2].size(3)).cuda()
    # clamp 作用 小于0的，置为0
    elif ur == 1:
        project_mapup1 = torch.clamp(
            pca(features[:, :, s1:s2, s1:s2], allfeats=BIGfeat[:, :, s1:s2, s1:s2], updown='up'), min=0)
        maxv = project_mapup1.view(project_mapup1.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        project_mapup1 /= maxv

    if ul == 0:
        project_mapup2 = torch.zeros(features[:, :, s1:s2, s2:s3].size(0), features[:, :, s1:s2, s2:s3].size(2),
                                     features[:, :, s1:s2, s2:s3].size(3)).cuda()
    elif ul == 1:
        project_mapup2 = torch.clamp(
            pca(features[:, :, s1:s2, s2:s3], allfeats=BIGfeat[:, :, s1:s2, s2:s3], updown='up'), min=0)
        maxv = project_mapup2.view(project_mapup2.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        project_mapup2 /= maxv

    if dr == 0:
        project_mapdown1 = torch.zeros(features[:, :, s2:s3, s1:s4].size(0), features[:, :, s2:s3, s1:s4].size(2),
                                       features[:, :, s2:s3, s1:s4].size(3)).cuda()
    elif dr == 1:
        project_mapdown1 = torch.clamp(
            pca(features[:, :, s2:s3, s1:s4], allfeats=BIGfeat[:, :, s2:s3, s1:s4], updown='down'), min=0)
        maxv = project_mapdown1.view(project_mapdown1.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        project_mapdown1 /= maxv

    if dm == 0:
        project_mapdown2 = torch.zeros(features[:, :, s2:s3, s4:s5].size(0), features[:, :, s2:s3, s4:s5].size(2),
                                       features[:, :, s2:s3, s4:s5].size(3)).cuda()
    elif dm == 1:
        project_mapdown2 = torch.clamp(
            pca(features[:, :, s2:s3, s4:s5], allfeats=BIGfeat[:, :, s2:s3, s4:s5], updown='down'), min=0)
        maxv = project_mapdown2.view(project_mapdown2.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        project_mapdown2 /= maxv

    if dl == 0:
        project_mapdown3 = torch.zeros(features[:, :, s2:s3, s5:s3].size(0), features[:, :, s2:s3, s5:s3].size(2),
                                       features[:, :, s2:s3, s5:s3].size(3)).cuda()
    elif dl == 1:
        project_mapdown3 = torch.clamp(
            pca(features[:, :, s2:s3, s5:s3], allfeats=BIGfeat[:, :, s2:s3, s5:s3], updown='down'), min=0)
        maxv = project_mapdown3.view(project_mapdown3.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        project_mapdown3 /= maxv

    if u == 0 and d == 0:
        project_mapup = torch.cat((project_mapup1, project_mapup2), 2)
        project_mapdown = torch.cat((project_mapdown1, project_mapdown2, project_mapdown3), 2)

    project_map = torch.cat((project_mapup, project_mapdown), 1)
    # project_map=F.adaptive_avg_pool2d(project_map,(7,7))

    coordinate = []
    feat = []

    for i in range(project_map.size(0)):
        Sproject_map = project_map[i]
        Singlefeature = features[i, :, :, :]
        meanmaskvalue = Sproject_map.sum() / (Sproject_map.size(0) * Sproject_map.size(1))
        Sproject_map[Sproject_map < theh] = 0
        allcood = torch.nonzero(Sproject_map).cpu().numpy()
        Sproject_map = Sproject_map.unsqueeze(0)
        # Smask = Sproject_map.repeat(Singlefeature.size(0), 1, 1)
        SOproject_map = Sproject_map.squeeze()
        SOproject_map = SOproject_map.reshape(SOproject_map.size(0) * SOproject_map.size(1))
        Singlefeature = Singlefeature.squeeze()
        Singlefeature = Singlefeature.permute(1, 2, 0)
        Singlefeature = Singlefeature.reshape(Singlefeature.size(0) * Singlefeature.size(1), Singlefeature.size(2))
        cnt = 0
        ClusterFeat = torch.zeros(Singlefeature.size(1)).unsqueeze(0)
        for j in range(SOproject_map.size(0)):
            if SOproject_map[j] > theh and cnt == 0:
                ClusterFeat = F.normalize(Singlefeature[j].unsqueeze(0))
                cnt = cnt + 1
            elif SOproject_map[j] >theh and cnt != 0:
                SF = F.normalize(Singlefeature[j].unsqueeze(0))
                ClusterFeat = torch.cat((ClusterFeat, SF), 0)
                cnt = cnt + 1
        ClusterFeatmean = torch.mean(ClusterFeat, dim=0)
        ClusterFeatmean = ClusterFeatmean.unsqueeze(0)
        ClusterFeatmax = torch.max(ClusterFeat, dim=0)
        ClusterFeatmax = ClusterFeatmax[0].unsqueeze(0)

        ClusterFeatmax = F.normalize(ClusterFeatmax)
        ClusterFeatmean = F.normalize(ClusterFeatmean)
        SingleClusterFeat = torch.cat((ClusterFeatmax, ClusterFeatmean), 1)
        SingleClusterFeat = SingleClusterFeat.squeeze()
        # Singlefeature = Singlefeature.mul(Smask)
        # Singlefeature = Singlefeature.reshape(Singlefeature.size(0), Singlefeature.size(1) * Singlefeature.size(2))
        Singlefeature = SingleClusterFeat.cpu().detach().numpy()
        # SPFeat = np.sum(Singlefeature, axis=1, keepdims=False) / (len(allcood))
        # ccc=Singlefeature.sum()/(len(aa))
        cood = np.mean(allcood, axis=0, keepdims=True)
        feat.append(Singlefeature)
        coordinate.append(cood)
        # Sproject_map[Sproject_map > meanmaskvalue * (0.5)] = 1
        project_map[i] = Sproject_map.reshape(project_map.size(1), project_map.size(2))

    project_map = F.interpolate(project_map.unsqueeze(1), size=(imgs.size(2), imgs.size(3)), mode='bilinear',
                                align_corners=False) * 255.

    coordinate = np.array(coordinate)
    coordinate = np.squeeze(coordinate)
    feat = np.array(feat)
    # tsne2 = manifold.TSNE(n_components=3, init='pca', random_state=501)
    # X_tsne2 = tsne2.fit_transform(feat)
    # x_min, x_max = X_tsne2.min(0), X_tsne2.max(0)
    # feat = (X_tsne2 - x_min) / (x_max - x_min)
    return coordinate,feat,project_map

# datamanager = torchreid.data.ImageDataManager(
#     root='reid-data',
#     sources='VehicleID',
#     height=224,
#     width=224,
#     batch_size=32,
# )

datadir='E:\\work\\DLMB-PB\\DLMB-PB\\data'

model = models.create('resnet50', num_features=1024,
                      dropout=0.5, num_classes=13164)
model = model.cuda()
checkpoint = load_checkpoint('E:\\work\\DLMB-PB\\DLMB-PB\\checkpointres50.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
data_list = os.listdir(datadir)

imgs = []

for name in data_list:
    img = image_trans(Image.open(os.path.join(datadir+'\\'+name)).convert('RGB'))
    imgs.append(img.unsqueeze(0))

imgs = torch.cat(imgs)

featurelist,_ = model(imgs.cuda())
#第15层的尺寸是56*56 第16层的尺寸是28*28 23层开始后是14*14
features=featurelist[6]
pca = PCAProjectNet()
BIGfeat=torch.load('./renet50vidlayer6.pkl')
coordinate,feat1,_=getcood(BIGfeat[0:2000, :, :, :].cuda(),BIGfeat[0:2000, :, :, :],1,0,0,0,0,0,0,theh=0.4)
Scoordinate,Sfeat1,project_map=getcood(features.cuda(),BIGfeat[0:2000, :, :, :],1,0,0,0,0,0,0,theh=0.4)
torch.save(feat1,'./feat1.pkl')
torch.save(Sfeat1,'./Sfeat1.pkl')

feat1=torch.load('./feat1.pkl')
Sfeat1=torch.load('./Sfeat1.pkl')

coordinate,feat2,_=getcood(BIGfeat[0:2000, :, :, :].cuda(),BIGfeat[0:2000, :, :, :],0,1,0,0,0,0,0,theh=0.4)
Scoordinate,Sfeat2,project_map=getcood(features.cuda(),BIGfeat[0:2000, :, :, :],0,1,0,0,0,0,0,theh=0.4)
torch.save(feat2,'./feat2.pkl')
torch.save(Sfeat2,'./Sfeat2.pkl')

feat2=torch.load('./feat2.pkl')
Sfeat2=torch.load('./Sfeat2.pkl')

feat=np.concatenate((feat1,feat2),axis=1)
Sfeat=np.concatenate((Sfeat1,Sfeat2),axis=1)


# coordinate,feat,_=getcood(features.cuda(),features.cuda(),0,1,0,0,0,0,0,theh=0.2)
# Scoordinate,Sfeat,project_map=getcood(features.cuda(),features.cuda(),0,1,0,0,0,0,0,theh=0.2)
#
# feat=np.concatenate((coordinate,feat),axis=1)
# Sfeat=np.concatenate((Scoordinate,Sfeat),axis=1)
# label = SpectralClustering(n_clusters=2).fit_predict(Sfeat)

km_cluster = KMeans(n_clusters=2, max_iter=3000, n_init=2, \
                    init='k-means++', n_jobs=-1)
km_cluster.fit(Sfeat)
# km_cluster=torch.load('./kmclustersub.pkl')
label = km_cluster.predict(feat)
torch.save(km_cluster,'./kmcluster.pkl')



save_imgs = []

for i, name in enumerate(data_list):
    # if i>0:
    #     break
    original_image = cv2.resize(cv2.imread(os.path.join(datadir, name)), (224, 224))
    mask = project_map[i].repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy()
    #mask=project_map[i].permute(1, 2, 0).detach().cpu().numpy()
    #img = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2BGR)
    orgimg = cv2.cvtColor(np.asarray(original_image), cv2.COLOR_RGB2BGR)
    # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 125, 255, 0)
    # thresh = np.clip(thresh, 0, 255)  # 归一化也行
    # thresh = np.array(thresh, np.uint8)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnt=0
    # for c in contours:
    #     M = cv2.moments(c)
    #     x, y, w, h = cv2.boundingRect(c)
    #     #if w*h<100 or w>100 or h>100 or w*h>4000:
    #     #    continue
    #     #scores = 1
    #     cnt=cnt+1
    #     cv2.rectangle(original_image,(x,y), (x+w,y+h), (153,153,0), 5)
        #region = original_image.crop((x, y, x + w, y + h))
    mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
    save_img = cv2.addWeighted(orgimg, 0.5, mask, 0.5, 0.0)
    if label[i]==0:
        save_img=cv2.rectangle(save_img, (int(0), int(0)), (int(224), int(224)), (0, 255, 0), 3)
    # elif label[i]==1:
    #     save_img=cv2.rectangle(save_img, (int(0), int(0)), (int(224), int(224)), (0, 0, 255), 3)
    # elif label[i]==2:
    #     save_img=cv2.rectangle(save_img, (int(0), int(0)), (int(224), int(224)), (255, 0, 0), 3)
    save_imgs.append(save_img)
save_imgs = np.concatenate(save_imgs, 1)
cv2.imwrite('./testddtvec.jpg', save_imgs)

