from __future__ import print_function, absolute_import
import os.path as osp
import random
import numpy as np
import pdb
from glob import glob
import re


class VehicleID(object):

    def __init__(self, root):

        self.images_dir = 'E:\\VehicleID_V1.0\\VehicleID_V1.0\\image\\'
        self.labelfile = 'E:\\VehicleID_V1.0\\VehicleID_V1.0\\train_test_split\\'
        self.train_path = self.images_dir
        self.gallery_path = self.images_dir
        self.query_path = self.images_dir
        self.train_file_path = self.labelfile+'train_list.txt'
        self.gallery_file__path = self.labelfile+'gallery2400.txt'
        self.query_file__path = self.labelfile+'query2400.txt'
        self.camstyle_path = 'bounding_box_train_camstyle'
        self.train, self.query, self.gallery, self.camstyle = [], [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids, self.num_camstyle_ids = 0, 0, 0, 0
        self.load()

    def preprocess(self, path, relabel=True):
        all_cids = {}
        ret = []
        # fpaths = sorted(glob(osp.join(self.images_dir, path, '*.jpg')))
        f = open(path)
        line = f.readline()
        while line:
            fname, cid = line.split()
            line = f.readline()
            if cid == -1: continue
            if relabel:
                if cid not in all_cids:
                    all_cids[cid] = len(all_cids)
            else:
                if cid not in all_cids:
                    all_cids[cid] = cid
            cid = all_cids[cid]
            # cam -= 1
            ret.append((fname + '.jpg', cid))

        f.close()
        return ret, int(len(all_cids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess(self.train_file_path)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_file__path, False)
        self.query, self.num_query_ids = self.preprocess(self.query_file__path, False)
#        self.camstyle, self.num_camstyle_ids = self.preprocess(self.camstyle_path)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
     #  print("  camstyle  | {:5d} | {:8d}"
     #       .format(self.num_camstyle_ids, len(self.camstyle)))
