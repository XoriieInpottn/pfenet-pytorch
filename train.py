#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-07-19
"""

import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import dataset
import pfenet
import utils


class Trainer(object):

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use.')
        parser.add_argument('--data-path', required=True, help='Path of the directory that contains the data files.')
        parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
        parser.add_argument('--num-epochs', type=int, default=100, help='The number of epochs to train.')
        parser.add_argument('--max-lr', type=float, default=1e-3, help='The maximum value of learning rate.')
        parser.add_argument('--weight-decay', type=float, default=0.3, help='The weight decay value.')
        parser.add_argument('--optimizer', default='AdamW', help='Name of the optimizer to use.')

        parser.add_argument('--num-shots', type=int, default=5)
        parser.add_argument('--image-size', type=int, default=224)
        parser.add_argument('--output-dir', default='output')
        self._args = parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = self._args.gpu
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._create_dataset()
        self._create_model()
        self._create_optimizer()

    def _create_dataset(self):
        train_dataset = dataset.SegmentationDataset(
            self._args.data_path,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            num_shots=self._args.num_shots,
            image_size=self._args.image_size,
            augmenters=dataset.DEFAULT_AUG
        )
        self._train_loader = DataLoader(
            train_dataset,
            batch_size=self._args.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True
        )
        test_dataset = dataset.SegmentationDataset(
            self._args.data_path,
            [16, 17, 18, 19, 20],
            num_shots=self._args.num_shots,
            image_size=self._args.image_size,
            augmenters=None
        )
        self._test_loader = DataLoader(
            test_dataset,
            batch_size=self._args.batch_size,
            num_workers=10,
            pin_memory=True
        )

    def _create_model(self):
        self._model = pfenet.PFENet(output_size=self._args.image_size)
        self._parameters = [
            *self._model.down_query.parameters(),
            *self._model.down_supp.parameters(),
            *self._model.init_merge.parameters(),
            *self._model.alpha_conv.parameters(),
            *self._model.beta_conv.parameters(),
            *self._model.inner_cls.parameters(),
            *self._model.res1.parameters(),
            *self._model.res2.parameters(),
            *self._model.cls.parameters(),
        ]
        self._loss = pfenet.Loss()

    def _create_optimizer(self):
        optimizer_class = getattr(optim, self._args.optimizer)
        self._optimizer = optimizer_class(
            self._parameters,
            lr=self._args.max_lr,
            weight_decay=self._args.weight_decay
        )
        num_loops = self._args.num_epochs * len(self._train_loader)
        self._scheduler = utils.CosineWarmUpAnnealingLR(self._optimizer, num_loops)

    def train(self):
        loss_g = 0.0
        for epoch in range(self._args.num_epochs):
            self._model.train()
            loop = tqdm(self._train_loader, dynamic_ncols=True, leave=False)
            for supp_doc, query_doc in loop:
                loss, lr = self._train_step(
                    supp_doc['image'],
                    supp_doc['label'],
                    query_doc['image'],
                    query_doc['label']
                )
                loss = float(loss.numpy())
                loss_g = 0.9 * loss_g + 0.1 * loss
                loop.set_description(f'[{epoch + 1}/{self._args.num_epochs}] L={loss_g:.06f} lr={lr:.01e}', False)
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self._args.num_epochs:
                self._model.eval()
                iou_result = self._evaluate(5)
                loop.set_description(
                    f'[{epoch + 1}/{self._args.num_epochs}] '
                    f'L={loss_g:.06f} '
                    f'precision={iou_result:.02%} '
                )

    def _evaluate(self, num_loops):
        iou_list = []
        for epoch in range(num_loops):
            loop = tqdm(self._test_loader, dynamic_ncols=True, leave=False)
            for supp_doc, query_doc in loop:
                output = self._predict_step(
                    supp_doc['image'],
                    supp_doc['label'],
                    query_doc['image']
                )
                target = query_doc['label']
                output = output.view(-1)
                target = target.view(-1)

                pred_inds = (output > 0)
                target_inds = (target > 0)

                intersection = (pred_inds[target_inds]).long().sum().data.cpu[0]
                union = pred_inds.long().sum().data.cpu[0] + target_inds.long().sum().data.cup[0] - intersection
                iou = float(intersection) / float(union)
                iou_list.append(iou)
        return np.mean(iou_list)

    def _train_step(self, sx, sy, qx, qy):
        sx = sx.to(self._device)
        sy = sy.to(self._device)
        qx = qx.to(self._device)
        qy = qy.to(self._device)
        pred, pred_aux = self._model(sx, sy, qx)
        loss = self._loss(pred, qy, pred_aux)
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._scheduler.step()
        return loss.detach().cpu(), self._scheduler.get_last_lr()[0]

    def _predict_step(self, sx, sy, qx):
        sx = sx.to(self._device)
        sy = sy.to(self._device)
        qx = qx.to(self._device)
        pred = self._model(sx, sy, qx)
        qy_ = torch.argmax(pred, 1)
        return qy_.detach().cpu()


if __name__ == '__main__':
    raise SystemExit(Trainer().train())
