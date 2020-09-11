from __future__ import print_function
from __future__ import division

import time
import torch
import datetime
import scipy.io
import numpy as np
from torch.autograd import Variable

from utils.avgmeter import AverageMeter
from eval_lib.eval_metrics import evaluate
# from eval_lib.evaluate_vehicleid import cmc_common_oneshot_v2, cmc_vehicleid
from directreranking import cmc_common_oneshot_v2







def test_vehicleid(model, queryloader, galleryloader, train_query_loader, train_gallery_loader, test_batch,
                   loss_type, euclidean_distance_loss, epoch, use_metric_cuhk03=False, ranks=[1, 5, 10, 20],
                   return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    # with torch.no_grad():
    #     qf, q_pids,qdids, q_paths = [], [], [] ,[]
    #     for batch_idx, (imgs, _, pids,dids) in enumerate(queryloader):
    #         imgs = imgs.cuda()
    #
    #         end = time.time()
    #         features = model(imgs)
    #         batch_time.update(time.time() - end)
    #         features = features.data.cpu()
    #
    #         qf.append(features)
    #         q_pids.extend(pids)
    #         qdids.extend(dids)
    # #     #    q_paths.extend(paths)
    #     qf = torch.cat(qf, 0)
    #     q_pids = np.asarray(q_pids)
    #     qdids = np.asarray(qdids)
    #     #q_paths = np.asarray(q_paths)
    #     print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
    #     print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))
    #     gf, g_pids, gdids, g_paths = [], [], [],[]
    #     for batch_idx, (imgs, _, pids,dids) in enumerate(galleryloader):
    #         imgs = imgs.cuda()
    #
    #         end = time.time()
    #         features = model(imgs)
    #         batch_time.update(time.time() - end)
    #         features = features.data.cpu()
    #
    #         gf.append(features)
    #         g_pids.extend(pids)
    #         gdids.extend(dids)
    #     #    g_paths.extend(paths)
    #     gf = torch.cat(gf, 0)
    #     g_pids = np.asarray(g_pids)
    #     gdids = np.asarray(gdids)
    # #     #g_paths = np.asarray(g_paths)
    # print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    # print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))

    # result = {'query_f': qf.numpy(),
    #           'query_cam': q_camids, 'query_label': q_pids, 'quim_path': q_paths,
    #           'gallery_f': gf.numpy(),
    #           'gallery_cam': g_camids, 'gallery_label': g_pids, 'gaim_path': g_paths}
    # scipy.io.savemat(os.path.join(args.save_dir, 'features_' + str(60) + '.mat'), result)
    # dist_mat_dict = {'dist_mat': distmat}
    # scipy.io.savemat(os.path.join(args.save_dir, 'features_' + str(60) + '_dist.mat'), dist_mat_dict)
    print("Start computing CMC and mAP")
    start_time = time.time()
    # torch.save(qf,'./qf')
    # torch.save(qdids, './qdids')
    # torch.save(q_pids, './q_pids')
    # torch.save(gf, './gf')
    # torch.save(gdids, './gdids')
    # torch.save(g_pids, './g_pids')
    qf=torch.load('./qf')
    qdids = torch.load('./qdids')
    q_pids = torch.load('./q_pids')
    gf = torch.load('./gf')
    gdids = torch.load('./gdids')
    g_pids = torch.load('./g_pids')
    for w in range(100,200):
        print("----------------" + str(w / 100) + "------------")
        cmc, mAP = cmc_common_oneshot_v2(w/100, qf.numpy(), qdids, q_pids, gf.numpy(), gdids, g_pids, repeat=1, topk=50)
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    epoch=0

    print("Evaluate test data time (h:m:s): {}.".format(elapsed))
    print("Test data results ----------")
    print("Epoch {} temAP: {:.2%}".format(epoch, mAP))
    print("CMC curve")
    for r in ranks:
        print("Epoch {} teRank-{:<3}: {:.2%}".format(epoch, r, cmc[r - 1]))
    print("------------------")

    return cmc[0], mAP


def testCATfeature(model1,model2, queryloader, galleryloader, test_batch, loss_type,
         euclidean_distance_loss, epoch, use_metric_cuhk03=False, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model1.eval()
    model2.eval()
    with torch.no_grad():
        tqf, tq_pids, tq_camids = [], [], []
        for batch_idx, (imgs1,imgs2, _, pids, camids) in enumerate(queryloader):

            imgs1 = Variable(imgs1.cuda())
            imgs2 = Variable(imgs2.cuda())
            end = time.time()
            features1 = model1(imgs1)
            features2 = model2(imgs2)
            features = torch.cat((features1,features2 ),1)
            #features = features1
            batch_time.update(time.time() - end)
            features = features.data.cpu()

            tqf.append(features)
            tq_pids.extend(pids)
            tq_camids.extend(camids)
        tqf = torch.cat(tqf, 0)
        tq_pids = np.asarray(tq_pids)
        tq_camids = np.asarray(tq_camids)
        print("Extracted features for train_query set, obtained {}-by-{} matrix".format(tqf.size(0), tqf.size(1)))
        print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))
        tgf, tg_pids, tg_camids = [], [], []
        for batch_idx, (imgs1,imgs2, _, pids, camids) in enumerate(galleryloader):
            imgs1 = imgs1.cuda()
            imgs2 = imgs2.cuda()
            end = time.time()
            features1 = model1(imgs1)
            features2 = model2(imgs2)
            features = torch.cat((features1,features2 ),1)
            #features=features1
            batch_time.update(time.time() - end)
            features = features.data.cpu()

            tgf.append(features)
            tg_pids.extend(pids)
            tg_camids.extend(camids)
        tgf = torch.cat(tgf, 0)
        tg_pids = np.asarray(tg_pids)
        tg_camids = np.asarray(tg_camids)
        print("Extracted features for train_gallery set, obtained {}-by-{} matrix".format(tgf.size(0), tgf.size(1)))
        print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))

    print("Start compute distmat.")
    if loss_type in euclidean_distance_loss:
        m, n = tqf.size(0), tgf.size(0)
        distmat = torch.pow(tqf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(tgf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, tqf, tgf.t())
        distmat = distmat.numpy()
    elif loss_type == 'angle':
        tvec_dot = torch.matmul(tqf, tgf.t())
        tqf_len = tqf.norm(dim=1, keepdim=True)
        tgf_len = tgf.norm(dim=1, keepdim=True)
        tvec_len = torch.matmul(tqf_len, tgf_len.t()) + 1e-5
        distmat = -torch.div(tvec_dot, tvec_len).numpy()
    else:
        raise KeyError("Unsupported loss: {}".format(loss_type))
    print("Compute distmat done.")
    print("distmat shape:", distmat.shape)
    print("Start computing CMC and mAP")
    start_time = time.time()
    cmc, mAP = evaluate(distmat, tq_pids, tg_pids, tq_camids, tg_camids,
                        use_metric_cuhk03=use_metric_cuhk03, use_cython=False)
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Evaluate train data time (h:m:s): {}.".format(elapsed))
    print("Train data results ----------")
    print("Epoch {} trmAP: {:.2%}".format(epoch, mAP))
    print("CMC curve")
    for r in ranks:
        print("Epoch {} trRank-{:<3}: {:.2%}".format(epoch, r, cmc[r - 1]))
    print("------------------")
    if return_distmat:
        return distmat
    return cmc[0], mAP

def test_vehicleid_formal(model, probeloader, galleryloader, train_query_loader, train_gallery_loader, use_gpu,
                          test_batch, loss_type, euclidean_distance_loss, epoch, use_metric_cuhk03=False,
                          ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        pf, p_pids, p_paths = [], [], []
        for batch_idx, (imgs, _, pids) in enumerate(probeloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()

            pf.append(features)
            p_pids.extend(pids)
        #    p_paths.extend(paths)
        pf = torch.cat(pf, 0)
        p_pids = np.asarray(p_pids)
        #p_paths = np.asarray(p_paths)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(pf.size(0), pf.size(1)))
        print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))

    # result = {'query_f': qf.numpy(),
    #           'query_cam': q_camids, 'query_label': q_pids, 'quim_path': q_paths,
    #           'gallery_f': gf.numpy(),
    #           'gallery_cam': g_camids, 'gallery_label': g_pids, 'gaim_path': g_paths}
    # scipy.io.savemat(os.path.join(args.save_dir, 'features_' + str(60) + '.mat'), result)
    # dist_mat_dict = {'dist_mat': distmat}
    # scipy.io.savemat(os.path.join(args.save_dir, 'features_' + str(60) + '_dist.mat'), dist_mat_dict)
    print("Start computing CMC and mAP")
    start_time = time.time()
    cmc, mAP = cmc_vehicleid(pf.numpy(), p_pids, repeat=10, topk=50)
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Evaluate test data time (h:m:s): {}.".format(elapsed))
    print("Test data results ----------")
    print("Epoch {} temAP: {:.2%}".format(epoch, mAP))
    print("CMC curve")
    for r in ranks:
        print("Epoch {} teRank-{:<3}: {:.2%}".format(epoch, r, cmc[r - 1]))
    print("------------------")

    return cmc[0], mAP



