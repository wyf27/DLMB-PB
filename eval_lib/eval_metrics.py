from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import copy
from collections import defaultdict
import sys

try:
    from torchreid.eval_lib.cython_eval import eval_market1501_wrap
    CYTHON_EVAL_AVAI = True
    print("Cython evaluation is AVAILABLE")
except ImportError:
    CYTHON_EVAL_AVAI = False
    print("Warning: Cython evaluation is UNAVAILABLE")


def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, N=100):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed N times (default: N=100).
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc, AP = 0., 0.
        for repeat_idx in range(N):
            mask = np.zeros(len(orig_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_orig_cmc = orig_cmc[mask]
            _cmc = masked_orig_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)
            # compute AP
            num_rel = masked_orig_cmc.sum()
            tmp_cmc = masked_orig_cmc.cumsum()
            tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * masked_orig_cmc
            AP += tmp_cmc.sum() / num_rel
        cmc /= N
        AP /= N
        all_cmc.append(cmc)
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, matches=None):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    with_matches = False
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    if matches is None:
        indices = np.argsort(distmat, axis=1)
        aa = len(indices)
        bb = len(indices[0])
        flag = np.ones((aa,bb))
        indexlist=[]
        #去除同一个摄像头下的结果
        for i in range (len(indices)):
            cnt = 0
            temp=indices[i]
            tempcam=q_camids[i]
            mylist = []
            print(str(i))
            for j in range (len(temp)):

                tempgallcamid=temp[j]
                tempgallcam=g_camids[tempgallcamid]
                if tempcam==tempgallcam:
                    continue
                else:
                    mylist.append(temp[j])
                    cnt = cnt + 1

                if cnt == 9000:
                    break
            mylist = np.array(mylist)
            # indices[i]=indices[i][mylist]
            indexlist.append(mylist)
        newindices=np.array(indexlist)
        newindices0=newindices[:]
                    # np.delete(indices,[i,j])
                    # indices[i].pop(j)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)


    else:
        with_matches = True

    # compute cmc curve for each query
    # all_cmc_new = np.zeros(num_q, max_rank)
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        if with_matches:
            orig_cmc = matches[q_idx]
        else:
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches

        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()

        # tmp_cmc = orig_cmc.cumsum()
        # tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        # tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        # AP = tmp_cmc.sum() / num_rel
        # all_AP.append(AP)

        AP = 0
        matched_count = 0
        tmp_cmc = orig_cmc.cumsum().astype(np.float32)

        if orig_cmc[0]:
            AP += 1.0 / num_rel
            matched_count += 1
        if num_rel >= 2:
            for i in range(1, len(orig_cmc)):
                if orig_cmc[i]:
                    AP += ((tmp_cmc[i] - tmp_cmc[i - 1]) / num_rel) * ((tmp_cmc[i] / (i + 1.) + tmp_cmc[i - 1] / i) / 2)
                    matched_count += 1
                if matched_count == num_rel:
                    break
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_metric_cuhk03=False,
             use_cython=True, matches=None):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    else:
        if use_cython and CYTHON_EVAL_AVAI:
            return eval_market1501_wrap(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
        else:
            return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, matches)


