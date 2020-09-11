from __future__ import print_function
from __future__ import division
from collections import defaultdict
import time
import torch
import datetime
import scipy.io
import numpy as np
import copy
from utils.avgmeter import AverageMeter

def pairwise_distance(query_feature, gallery_feature, is_save=False, save_path=None):
    x = query_feature
    y = gallery_feature
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())

    return dist.numpy()

def compute_dist_mat(query_feature, gallery_feature, is_save=False, save_path=None):
    qim_cnt = len(query_feature)
    gim_cnt = len(gallery_feature)
    dist_mat = np.zeros((qim_cnt, gim_cnt), dtype='float32')
    for i in range(qim_cnt):

        dist_mat[i] = np.dot(gallery_feature, query_feature[i])
        # x = query_feature[i]
        # y = gallery_feature
        # x = x.unsqueeze(0)
        # m, n = x.size(0), y.size(0)
        # x = x.view(m, -1)
        # y = y.view(n, -1)
        #
        # dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #        torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # dist.addmm_(1, -2, x, y.t())
        # dist_mat[i]=dist
        # print(i)
    if is_save:
        dist_mat_dict = {'dist_mat': dist_mat}
        scipy.io.savemat(save_path, dist_mat_dict)
    else:
        return dist_mat

def direct_reranking(querydirectID,gallerydirectID):

    reranking_indices=[]

    for j in range(len(gallerydirectID)):
        if querydirectID != gallerydirectID[j]:
            reranking_indices.append(j)

    for i in range(len(gallerydirectID)):
        if querydirectID == gallerydirectID[i]:
            reranking_indices.append(i)

    reranking_indices=np.array(reranking_indices)

    return reranking_indices

def cmc_common_oneshot_v2(W,query_feature,query_directionid, query_label, gallery_feature, gallery_directionid,gallery_label, repeat=10, topk=50):
    quim_cnt = len(query_feature)
    id_dict = defaultdict(list)
    for index, key in enumerate(gallery_label):
        id_dict[key].append(index)
    if topk is None:
        cmc = np.zeros(len(id_dict))
    else:
        cmc = np.zeros(topk)
    num_valid_queries = 0

    ave_cmc = []
    ave_mAP = []
    for _ in range(repeat):
        # num_valid_queries = 0
        # cmc = np.zeros(len(id_dict))
        gallery_index = []
        for key, index_list in id_dict.items():
            # i = np.random.choice(index_list)
            # for j in range(len(index_list)):
            #     k = index_list[j]
            #     gallery_index.append(k)
            i = index_list[0]
            gallery_index.append(i)
        gallery_f = gallery_feature[gallery_index]
        # part_gallery_f = gallery_feature_part[gallery_index]
        # 多特征reranking候选集特征提取
        gallery_l = gallery_label[gallery_index]
        gallery_d =gallery_directionid[gallery_index]
        dist_mat = compute_dist_mat(query_feature, gallery_f)

        # dist_mat =pairwise_distance(query_feature, gallery_f)
        aaaa=np.max(dist_mat)
        indices = np.argsort(dist_mat, axis=1)
        indices = indices[:, ::-1]
        dist_mat=dist_mat/(np.max(dist_mat)-np.min(dist_mat))
        # indices = indices[:, ::-1]
        aa=indices[:, 0]
        mindist=[]
        reranklist=[]
        begin = time.time()
        for i in range(len(aa)):

            top1galleryind=indices[i,0]
            gdirect=gallery_d[top1galleryind]
            qdirect=query_directionid[i]
            Sdist=dist_mat[i,aa[i]]
            # if dist_mat[i, indices[i, 0]] < W :
            if Sdist< W and gdirect==qdirect:
                head=[]
                rear=[]
                for j in range(100):
                    s_gdirect=gallery_d[indices[i,j]]
                    if s_gdirect==qdirect:
                        rear.append(indices[i,j])
                    elif s_gdirect!=qdirect:
                        head.append(indices[i,j])
                newlist=head+rear

                newarray = np.asarray(newlist)
                indices[i, 0:100]=newarray
                # reranklist.append(i)
            # mindist.append(Sdist)
        # mindist =np.array(mindist)

        # 单reranking阶段
        # for m in reranklist:
        #     reranking_num=50
        #     matchlist=indices[m]
        #     newtop = copy.deepcopy(matchlist[0:reranking_num])
        #     querydirectID=query_directionid[m]
        #     gallerydirectID = gallery_d[newtop]
        #     reranking_indices=direct_reranking(querydirectID,gallerydirectID)
        #
        #     for n in range(reranking_num):
        #         indices[m,n]=newtop[reranking_indices[n]]

        end = time.time()
        spend = end - begin
        print(spend)

        matches = (gallery_l[indices] == query_label[:, np.newaxis]).astype(np.int32)
        #
        # torch.save(matches,'./97match.pkl')

        # matches =torch.load('./68match.pkl')
        # a1=np.sum(matches)
        # matches2 = torch.load('./97match.pkl')
        # a2=np.sum(matches2)
        all_cmc = []
        all_AP = []
        num_valid_q = 0.0   # number of valid query
        for i in range(quim_cnt):
            orig_cmc = matches[i]

            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:topk])
            num_valid_q += 1.0

            num_rel = orig_cmc.sum()
            AP = 0
            matched_count = 0
            tmp_cmc = orig_cmc.cumsum().astype(np.float32)

            if orig_cmc[0]:
                AP += 1.0 / num_rel
                matched_count += 1
            for i in range(1, len(orig_cmc)):
                if orig_cmc[i]:
                    AP += ((tmp_cmc[i] - tmp_cmc[i - 1]) / num_rel) \
                          * ((tmp_cmc[i] / (i + 1.) + tmp_cmc[i - 1] / i) / 2)
                    matched_count += 1
                if matched_count == num_rel:
                    break

            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
        ave_cmc.append(all_cmc)
        ave_mAP.append(mAP)
        print(_, 'Rank1-5:', all_cmc[:5], 'mAP:', mAP)

    ave_cmc = np.asarray(ave_cmc).sum(0) / repeat
    ave_mAP = np.mean(ave_mAP)

    return ave_cmc, ave_mAP


def test_PECvid(model,W,queryloader, galleryloader, train_query_loader, train_gallery_loader, test_batch,
                   loss_type, euclidean_distance_loss, epoch, use_metric_cuhk03=False, ranks=[1, 5, 10, 20],
                   return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    qf, q_pids, q_dirIDs, q_paths = [], [], [], []
    for batch_idx, (imgs, _, pids ,q_dirID) in enumerate(queryloader):
        imgs = imgs.cuda()

        end = time.time()
        # features= model(imgs)
        batch_time.update(time.time() - end)
        # features = features.data.cpu()
        # qf.append(features)
        q_pids.extend(pids)
        q_dirIDs.extend(q_dirID)
    # qf = torch.cat(qf, 0)
    # torch.save(qf,'./vggqf')
    qf = torch.load('./STN_qf.pkl')
    q_pids = np.asarray(q_pids)
    q_dirIDs = np.asarray(q_dirIDs)

    # q_paths = np.asarray(q_paths)
    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))
    gf, g_pids, g_dirIDs, g_paths = [], [], [], []
    for batch_idx, (imgs, _, pids ,g_dirID) in enumerate(galleryloader):
        imgs = imgs.cuda()

        end = time.time()
        # features= model(imgs)
        batch_time.update(time.time() - end)
        # features = features.data.cpu()
        # gf.append(features)
        g_pids.extend(pids)
        g_dirIDs.extend(g_dirID)
    # gf = torch.cat(gf, 0)
    # torch.save(gf, './vgggf')
    gf = torch.load('./STN_gf.pkl')
    g_pids = np.asarray(g_pids)
    g_dirIDs = np.asarray(g_dirIDs)


    start_time = time.time()
    # cmc, mAP = cmc_common_oneshot_v2(qf.numpy(),part_qf, q_pids, gf.numpy(),part_gf, g_pids, repeat=1, topk=50)
    for i in range(200):
        i=i+200
        w=i/200
        print(str(w))
        cmc, mAP = cmc_common_oneshot_v2(w,qf.cpu(), q_dirIDs, q_pids, gf.cpu(), g_dirIDs, g_pids, repeat=1, topk=50)
    # cmc, mAP = cmc_common_oneshot_v2(0.86, qf.cpu(), q_dirIDs, q_pids, gf.cpu(), g_dirIDs, g_pids, repeat=1, topk=50)
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))

    return cmc[0], mAP
