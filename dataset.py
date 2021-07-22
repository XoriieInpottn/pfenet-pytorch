#!/usr/bin/env python3

"""
@author: xi
@since: 2021-07-19
"""

import collections
import json
import os
import random
from typing import Iterable

import cv2 as cv
import numpy as np
from imgaug import SegmentationMapsOnImage
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
from tqdm import tqdm

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def encode_image(image: np.ndarray) -> np.ndarray:
    """Convert an image to float32 and CHW format.

    :param image: np.ndarray, dtype=uint8, shape=(h, w, c)
    :return: np.ndarray, dtype=float32, shape=(c, h, w)
    """
    image = image.astype(np.float32)
    image = (image / 255.0 - MEAN) / STD
    image = np.transpose(image, (2, 0, 1))
    return image


def decode_image(tensor: np.ndarray) -> np.ndarray:
    """Convert float tensor back to an image.

    :param tensor: np.ndarray, dtype=float32, shape=(c, h, w)
    :return: np.ndarray, dtype=uint8, shape=(h, w, c)
    """
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = (tensor * STD + MEAN) * 255.0
    tensor = np.clip(tensor, 0, 255)
    return tensor.astype(np.uint8)


def encode_label(label: np.ndarray) -> np.ndarray:
    return label.astype(np.int64)


def decode_label(tensor: np.ndarray) -> np.ndarray:
    return tensor.astype(np.uint8)


class AugmenterWrapper(object):

    def __init__(self, augmenters: Iterable[iaa.Augmenter]):
        self._augmenter = iaa.Sequential(augmenters)

    def __call__(self, image, mask):
        seg_maps = SegmentationMapsOnImage(mask, shape=mask.shape)
        image, seg_maps = self._augmenter(image=image, segmentation_maps=seg_maps)
        mask = seg_maps.arr
        mask = mask.squeeze(2)
        return image, mask


IGNORE_CLASS = 255


class SegmentationDataset(Dataset):
    """Dataset for k-shot image segmentation.
    """

    def __init__(self,
                 path,
                 sub_class_list,
                 num_shots,
                 image_size,
                 is_train=False,
                 parse_class=False):
        self._num_shots = num_shots
        if not isinstance(image_size, (tuple, list)):
            image_size = (image_size, image_size)
        self._transform = AugmenterWrapper([
            iaa.Resize({'longer-side': (473, 500), 'shorter-side': 'keep-aspect-ratio'}, 'linear'),
            iaa.Fliplr(0.5),
            iaa.Rotate((-10, 10), cval=127),
            iaa.GaussianBlur((0.0, 0.1)),
            iaa.PadToAspectRatio(1.0, pad_cval=127, position='center-center'),
            iaa.CropToFixedSize(473, 473)
        ]) if is_train else AugmenterWrapper([
            iaa.Resize({'longer-side': 473, 'shorter-side': 'keep-aspect-ratio'}, 'linear'),
            iaa.PadToAspectRatio(1.0, pad_cval=127, position='center-center')
        ])

        self._dir_path = os.path.dirname(path)
        docs = []
        with open(path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                doc['image'] = os.path.join(self._dir_path, doc['image'])
                doc['label'] = os.path.join(self._dir_path, doc['label'])
                docs.append(doc)

        self._doc_list = []
        self._sub_class_dict = collections.defaultdict(list)
        for doc in tqdm(docs, leave=False):
            # label_class = []

            if parse_class:
                label = np.load(doc['label'])
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
                        # label_class.append(c)
                        self._doc_list.append((doc, c))
                        self._sub_class_dict[c].append(doc)
            else:
                for c in doc['class']:
                    if c not in sub_class_list:
                        continue
                    # label_class.append(c)
                    self._doc_list.append((doc, c))
                    self._sub_class_dict[c].append(doc)

            # if len(label_class) > 0:
            #     self._doc_list.append(doc)
            #     for c in label_class:
            #         if c in sub_class_list:
            #             self._sub_class_dict[c].append(doc)

    def __len__(self):
        return len(self._doc_list)

    def __getitem__(self, i):
        doc, class_chosen = self._doc_list[i]
        image = cv.imread(doc['image'], cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        raw_label = np.load(doc['label'])
        label = self._make_label(raw_label, class_chosen)
        if callable(self._transform):
            image, label = self._transform(image, label)
        query_doc = {
            'image': encode_image(image),
            'label': encode_label(label),
            'class': class_chosen
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
            supp_image_list.append(encode_image(image))
            supp_label_list.append(encode_label(label))
        supp_doc = {
            'image': np.stack(supp_image_list),  # (k, c, h, w)
            'label': np.stack(supp_label_list)  # (k, h, w)
        }

        return supp_doc, query_doc

    @staticmethod
    def _make_label(raw_label, class_chosen):
        label = np.zeros_like(raw_label)
        target_pix = np.where(raw_label == class_chosen)
        ignore_pix = np.where(raw_label == IGNORE_CLASS)
        label[:, :] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0], target_pix[1]] = 1
        label[ignore_pix[0], ignore_pix[1]] = IGNORE_CLASS
        return label


def test():
    ds = SegmentationDataset(
        'data/name.json',
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        num_shots=5,
        image_size=(473, 473),
        is_train=True
    )
    # for supp_doc, query_doc in tqdm(ds):
    #     pass
    # exit()
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        num_workers=40,
        pin_memory=True
    )
    print(len(loader))
    for supp_doc, query_doc in tqdm(loader):
        pass
        # print('image', query_doc['image'].shape)
        # print('label', query_doc['label'].shape)
        # print('class', query_doc['class'])
    return 0


if __name__ == '__main__':
    raise SystemExit(test())
