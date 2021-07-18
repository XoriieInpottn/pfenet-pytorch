#!/usr/bin/env python3

"""
@author: xi
@since: 2021-07-19
"""

import collections
import random
from typing import Iterable

import cv2 as cv
import numpy as np
from docset import DocSet
from imgaug import SegmentationMapsOnImage
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
from tqdm import tqdm

DEFAULT_AUG = [
    iaa.Fliplr(p=0.5),
    iaa.Flipud(p=0.5),
    iaa.Crop(percent=(0, 0.1))
]


class AugmenterWrapper(object):

    def __init__(self, augmenters: Iterable[iaa.Augmenter]):
        self._augmenter = iaa.Sequential(augmenters)

    def __call__(self, image, mask):
        seg_maps = SegmentationMapsOnImage(mask, shape=mask.shape)
        image, seg_maps = self._augmenter(image=image, segmentation_maps=seg_maps)
        mask = seg_maps.arr
        mask = mask.squeeze(2)
        return image, mask


class SegmentationDataset(Dataset):
    """Dataset for k-shot image segmentation.
    """

    DS_CACHE = {}

    def __init__(self,
                 path,
                 sub_class_list,
                 num_shots,
                 augmenters=None):
        self._num_shots = num_shots
        self._transform = None
        if augmenters is not None:
            self._transform = AugmenterWrapper(augmenters)

        # Shaban uses these lines to remove small objects:
        # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
        #    filtered_item.append(item)
        # which means the mask will be downsampled to 1/32 of the original size and the valid area should be
        # larger than 2, therefore the area in original size should be accordingly larger than 2 * 32 * 32
        self._doc_list = []  # type: list[dict]
        self._sub_class_dict = collections.defaultdict(list)
        for doc in self._load_ds(path):
            label = doc['mask']
            label_class = []
            for c in doc['classes']:
                if c in sub_class_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0], target_pix[1]] = 1
                    if tmp_label.sum() >= 2 * 32 * 32:
                        label_class.append(c)
            if len(label_class) > 0:
                self._doc_list.append(doc)
                for c in label_class:
                    if c in sub_class_list:
                        self._sub_class_dict[c].append(doc)

    def _load_ds(self, path):
        if path in self.DS_CACHE:
            return self.DS_CACHE[path]
        doc_list = []
        self.DS_CACHE[path] = doc_list
        with DocSet(path, 'r') as ds:
            for doc in tqdm(ds):
                doc_list.append(doc)
        return doc_list

    def __getitem__(self, i):
        doc = self._doc_list[i]
        class_chosen = random.choice([int(a) for a in np.unique(doc['mask']) if a != 0 and a != 255])

        image = cv.imdecode(np.frombuffer(doc['image'], np.byte), cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        label = self._make_label(doc['mask'], class_chosen)
        if callable(self._transform):
            image, label = self._transform(image, label)

        supp_image_list = []
        supp_label_list = []
        docs_chosen = random.sample(self._sub_class_dict[class_chosen], self._num_shots)
        for doc in docs_chosen:
            image = cv.imdecode(np.frombuffer(doc['image'], np.byte), cv.IMREAD_COLOR)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            label = self._make_label(doc['mask'], class_chosen)
            if callable(self._transform):
                image, label = self._transform(image, label)

    @staticmethod
    def _make_label(raw_label, class_chosen):
        label = np.zeros_like(raw_label)
        target_pix = np.where(label == class_chosen)
        ignore_pix = np.where(label == 255)
        label[:, :] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0], target_pix[1]] = 1
        label[ignore_pix[0], ignore_pix[1]] = 255
        return label
