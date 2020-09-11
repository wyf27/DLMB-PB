from PCAVec import PCAVec
import os
from vgg import *
from PIL import Image
import torchvision.transforms as tvt
import torch
import torchvision.utils as tvu
import torch.nn.functional as F
import cv2
import numpy as np
from reid.utils.data import transforms as T
from torch.autograd import Variable
from reid.utils import to_torch

image_trans = tvt.Compose([
    tvt.Resize((224, 224)),
    tvt.ToTensor(),
    tvt.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])

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

def getcood(features,u,d,theh):
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
        VecDown = torch.load('./VecDown.pkl')
        project_mapdown=torch.clamp(getPmap(features[:, :, s2:s3, :],VecDown), min=0)
        maxv = project_mapdown.view(project_mapdown.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        project_mapdown /= maxv

    # if ur == 0:
    #     project_mapup1 = torch.zeros(features[:, :, s1:s2, s1:s2].size(0), features[:, :, s1:s2, s1:s2].size(2),
    #                                  features[:, :, s1:s2, s1:s2].size(3)).cuda()
    # # clamp 作用 小于0的，置为0
    # elif ur == 1:
    #     project_mapup1 = torch.clamp(
    #         pca(features[:, :, s1:s2, s1:s2], allfeats=BIGfeat[:, :, s1:s2, s1:s2], updown='up'), min=0)
    #     maxv = project_mapup1.view(project_mapup1.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
    #     project_mapup1 /= maxv
    #
    # if ul == 0:
    #     project_mapup2 = torch.zeros(features[:, :, s1:s2, s2:s3].size(0), features[:, :, s1:s2, s2:s3].size(2),
    #                                  features[:, :, s1:s2, s2:s3].size(3)).cuda()
    # elif ul == 1:
    #     project_mapup2 = torch.clamp(
    #         pca(features[:, :, s1:s2, s2:s3], allfeats=BIGfeat[:, :, s1:s2, s2:s3], updown='up'), min=0)
    #     maxv = project_mapup2.view(project_mapup2.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
    #     project_mapup2 /= maxv
    #
    # if dr == 0:
    #     project_mapdown1 = torch.zeros(features[:, :, s2:s3, s1:s4].size(0), features[:, :, s2:s3, s1:s4].size(2),
    #                                    features[:, :, s2:s3, s1:s4].size(3)).cuda()
    # elif dr == 1:
    #     project_mapdown1 = torch.clamp(
    #         pca(features[:, :, s2:s3, s1:s4], allfeats=BIGfeat[:, :, s2:s3, s1:s4], updown='down'), min=0)
    #     maxv = project_mapdown1.view(project_mapdown1.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
    #     project_mapdown1 /= maxv
    #
    # if dm == 0:
    #     project_mapdown2 = torch.zeros(features[:, :, s2:s3, s4:s5].size(0), features[:, :, s2:s3, s4:s5].size(2),
    #                                    features[:, :, s2:s3, s4:s5].size(3)).cuda()
    # elif dm == 1:
    #     project_mapdown2 = torch.clamp(
    #         pca(features[:, :, s2:s3, s4:s5], allfeats=BIGfeat[:, :, s2:s3, s4:s5], updown='down'), min=0)
    #     maxv = project_mapdown2.view(project_mapdown2.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
    #     project_mapdown2 /= maxv
    #
    # if dl == 0:
    #     project_mapdown3 = torch.zeros(features[:, :, s2:s3, s5:s3].size(0), features[:, :, s2:s3, s5:s3].size(2),
    #                                    features[:, :, s2:s3, s5:s3].size(3)).cuda()
    # elif dl == 1:
    #     project_mapdown3 = torch.clamp(
    #         pca(features[:, :, s2:s3, s5:s3], allfeats=BIGfeat[:, :, s2:s3, s5:s3], updown='down'), min=0)
    #     maxv = project_mapdown3.view(project_mapdown3.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
    #     project_mapdown3 /= maxv
    #
    # if u == 0 and d == 0:
    #     project_mapup = torch.cat((project_mapup1, project_mapup2), 2)
    #     project_mapdown = torch.cat((project_mapdown1, project_mapdown2, project_mapdown3), 2)

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


    coordinate = np.array(coordinate)
    coordinate = np.squeeze(coordinate)
    feat = np.array(feat)
    # tsne2 = manifold.TSNE(n_components=3, init='pca', random_state=501)
    # X_tsne2 = tsne2.fit_transform(feat)
    # x_min, x_max = X_tsne2.min(0), X_tsne2.max(0)
    # feat = (X_tsne2 - x_min) / (x_max - x_min)
    return coordinate,feat,project_map



def MakeOne(kmeans,kmeanssubback,kmeanssubfront, Singlefeatures, namelist,cidlist,fnew):
    for i in range(0 , len(namelist)):
        Sname = namelist[i]
        cid = cidlist[i]
        Singlefeature=Singlefeatures[i,:,:,:]
        _, featup, _=getcood(Singlefeature.unsqueeze(0),1,0,0.4)
        _, featdown, _ = getcood(Singlefeature.unsqueeze(0), 0, 1, 0.4)
        feat = np.concatenate((featup, featdown), axis=1)
        label = kmeans.predict(feat)
        labelsubback = kmeanssubback.predict(feat)
        labelsubfront = kmeanssubfront.predict(feat)
        # img=Image.open('/home/wyf/Work/VehicleID_V1.0/VehicleID_V1.0/image/' + Sname).convert('RGB')

        if label==1:
            if labelsubback==0:
                newline=Sname + ' ' +cid+' '+ str(0)+'\n'
                fnew.writelines(newline)
                # img.save('./direction/galleryfront/' + Sname)
            elif labelsubback==1:
                newline = Sname + ' ' + cid + ' ' + str(1) + '\n'
                # img.save('./direction/galleryback/' + Sname)
                fnew.writelines(newline)
        elif label==0:
            if labelsubfront==1:
                newline = Sname + ' ' + cid + ' ' + str(0) + '\n'
                fnew.writelines(newline)
                # img.save('./direction/galleryfront/' + Sname)
            elif labelsubfront==0:
                newline=Sname + ' ' +cid+' '+ str(1)+'\n'
                # img.save('./direction/galleryback/' + Sname)
                fnew.writelines(newline)
if __name__ == '__main__':

    model=torch.load('/home/wyf/Work/modeldata/SConvVIDVGG.pkl')
    batchcnt = 0
    SingleImgs = []
    namelist = []
    cidlist = []
    linelist = []
    cnt = 0
    kmeans = torch.load('./kmcluster.pkl')
    kmeanssubback = torch.load('./kmclustersubback.pkl')
    kmeanssubfront = torch.load('./kmclustersubfront.pkl')
    f = open("/home/wyf/Work/VehicleID_V1.0/VehicleID_V1.0/train_test_split/gallery2400.txt")
    imagedir = '/home/wyf/Work/VehicleID_V1.0/VehicleID_V1.0/image/'
    fnew = open(r'/home/wyf/Work/VehicleID_V1.0/VehicleID_V1.0/train_test_split/gallery2400_direct.txt', 'w')
    # fback = open(r'/home/wyf/Work/VehicleID_V1.0/VehicleID_V1.0/train_test_split/gallery2400subsetback.txt', 'w')
    cnt = 0
    line = f.readline()
    while line:
        Sname, cid = line.split()
        fpath = imagedir + Sname + '.jpg'
        # if os.path.exists('/home/wyf/pythonworkspace/DDT-master/subsetimagetest/' + Sname):
        #     continue
        SOImg = image_trans(Image.open('/home/wyf/Work/VehicleID_V1.0/VehicleID_V1.0/image/' + Sname + '.jpg').convert('RGB'))
        img = cv2.resize(cv2.imread('/home/wyf/Work/VehicleID_V1.0/VehicleID_V1.0/image/' + Sname + '.jpg'), (224, 224))
        SingleImgs.append(SOImg.unsqueeze(0))
        namelist.append(Sname)
        cidlist.append(cid)
        linelist.append(line)
        batchcnt = batchcnt + 1

        if batchcnt == 10:
            SingleImgs = torch.cat(SingleImgs)
            Singlefeature, _ = model(SingleImgs.cuda())
            MakeOne(kmeans,kmeanssubback,kmeanssubfront, Singlefeature[19], namelist,cidlist,fnew)
            SingleImgs = []
            namelist = []
            cidlist = []
            linelist = []
            batchcnt = 0
        line = f.readline()
        print(str(cnt))
        cnt=cnt+1





