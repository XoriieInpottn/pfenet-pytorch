#!/usr/bin/env python3

"""
@author: xi
@since: 2021-07-19
"""

import collections
import csv
import os
import random
from typing import Iterable

import cv2 as cv
import numpy as np
import torch
from imgaug import SegmentationMapsOnImage
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
from tqdm import tqdm

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an image to torch tensor.

    :param image: np.ndarray, dtype=uint8, shape=(h, w, c)
    :return: torch.Tensor, dtype=float32, shape=(c, h, w)
    """
    tensor = torch.tensor(image, dtype=torch.float32)
    tensor = (tensor - MEAN) / STD
    tensor = tensor.permute((2, 0, 1))
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.permute((1, 2, 0))
    tensor = tensor * STD + MEAN
    tensor = torch.clip(tensor, 0, 255)
    image = np.array(tensor.numpy(), np.uint8)
    return image


def label_to_tensor(label: np.ndarray) -> torch.Tensor:
    """Convert a label to torch tensor.

    :param label: np.ndarray, dtype=uint8, shape=(h, w)
    :return: torch.Tensor, dtype=int64, shape=(h, w)
    """
    return torch.tensor(label, dtype=torch.int64)


def tensor_to_label(tensor: torch.Tensor) -> np.ndarray:
    return np.array(tensor.numpy(), np.uint8)


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

    def __init__(self,
                 path,
                 sub_class_list,
                 num_shots,
                 image_size=(448, 448),
                 augmenters=None):
        self._num_shots = num_shots
        if not isinstance(image_size, (tuple, list)):
            image_size = (image_size, image_size)
        if augmenters is None:
            augmenters = []
        self._transform = AugmenterWrapper([
            *augmenters,
            iaa.PadToAspectRatio(1.0, position='center-center').to_deterministic(),
            iaa.Resize(image_size)
        ])

        self._dir_path = os.path.dirname(path)
        docs = []
        with open(path, 'r') as f:
            csv_reader = csv.DictReader(f)
            for doc in csv_reader:
                doc['image'] = os.path.join(self._dir_path, doc['image'])
                doc['label'] = os.path.join(self._dir_path, doc['label'])
                docs.append(doc)

        self._doc_list = []  # type: list[dict]
        self._sub_class_dict = collections.defaultdict(list)
        for doc in tqdm(docs):
            label = np.load(doc['label'])
            label_class = []
            for c in np.unique(label):
                c = int(c)
                if c == 0 or c == 255:
                    continue
                if c not in sub_class_list:
                    continue
                tmp_label = np.zeros_like(label)
                target_pix = np.where(label == c)
                tmp_label[target_pix[0], target_pix[1]] = 1
                # Shaban uses these lines to remove small objects:
                # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
                #    filtered_item.append(item)
                # which means the mask will be downsampled to 1/32 of the original size and the valid area should be
                # larger than 2, therefore the area in original size should be accordingly larger than 2 * 32 * 32
                if tmp_label.sum() >= 2 * 32 * 32:
                    label_class.append(c)
            if len(label_class) > 0:
                self._doc_list.append(doc)
                for c in label_class:
                    if c in sub_class_list:
                        self._sub_class_dict[c].append(doc)

    def __getitem__(self, i):
        doc = self._doc_list[i]
        image = cv.imread(doc['image'], cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        raw_label = np.load(doc['label'])
        class_chosen = random.choice([int(a) for a in np.unique(raw_label) if a in self._sub_class_dict])
        label = self._make_label(raw_label, class_chosen)
        if callable(self._transform):
            image, label = self._transform(image, label)
        query_doc = {
            'image': image_to_tensor(image),
            'label': label_to_tensor(label)
        }

        supp_image_list = []
        supp_label_list = []
        docs_chosen = random.sample(self._sub_class_dict[class_chosen], self._num_shots)
        for doc in docs_chosen:
            image = cv.imread(doc['image'], cv.IMREAD_COLOR)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            raw_label = np.load(doc['label'])
            label = self._make_label(raw_label, class_chosen)
            if callable(self._transform):
                image, label = self._transform(image, label)
            supp_image_list.append(image_to_tensor(image))
            supp_label_list.append(label_to_tensor(label))
        supp_doc = {
            'image': torch.stack(supp_image_list),  # (k, c, h, w)
            'label': torch.stack(supp_label_list)  # (k, h, w)
        }

        return supp_doc, query_doc

    @staticmethod
    def _make_label(raw_label, class_chosen):
        label = np.zeros_like(raw_label)
        target_pix = np.where(raw_label == class_chosen)
        # ignore_pix = np.where(raw_label == 255)
        label[:, :] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0], target_pix[1]] = 1
        # label[ignore_pix[0], ignore_pix[1]] = 255
        return label
