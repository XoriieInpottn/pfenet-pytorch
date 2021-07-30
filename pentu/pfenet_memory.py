#!/usr/bin/env python3

'''
Author: Haoyu
Date: 2021-07-28 14:28:00
'''

import os
import torch
from torch import nn
import torchvision.models as models
from torch.nn import functional as F

from update import Memory_network


def resize(feat, size):
    return F.interpolate(feat, size=size, mode='bilinear', align_corners=True)


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PFENet(nn.Module):

    def __init__(self,
                 layer0: nn.Module,
                 layer1: nn.Module,
                 layer2: nn.Module,
                 layer3: nn.Module,
                 layer4: nn.Module,
                 backbone_feat_size,
                 feat_size=256,
                 num_classes=2,
                 ppm_scales=(60, 30, 15, 8)):
        super(PFENet, self).__init__()
        self._ppm_scales = ppm_scales
        self.feat_matrix = None
        # self.memory_network = Memory_network(feat_channel=256, matrix_size=10)
        self.memory_network = None

        def _no_update(layer: nn.Module) -> nn.Module:
            layer.eval()
            for p in layer.parameters():
                p.requires_grad = False
            return layer

        self.layer0 = _no_update(layer0)
        self.layer1 = _no_update(layer1)
        self.layer2 = _no_update(layer2)
        self.layer3 = _no_update(layer3)
        self.layer4 = _no_update(layer4)

        self.down_query = nn.Sequential(
            nn.Conv2d(backbone_feat_size, feat_size,
                      kernel_size=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(backbone_feat_size, feat_size,
                      kernel_size=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.init_merge = nn.ModuleList()
        self.alpha_conv = nn.ModuleList()
        self.beta_conv = nn.ModuleList()
        self.inner_cls = nn.ModuleList()
        for i in range(len(self._ppm_scales)):
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(feat_size * 3 + 1, feat_size,
                          kernel_size=(1, 1), bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_size, feat_size,
                          kernel_size=(1, 1), bias=False),
                nn.ReLU(inplace=True)
            ))
            if i > 0:
                # alpha_conv is used to fuse the feature from the last scale
                # so at the first scale, there is nothing to conv
                self.alpha_conv.append(nn.Sequential(
                    nn.Conv2d(feat_size * 2, feat_size,
                              kernel_size=(1, 1), bias=False),
                    nn.ReLU(inplace=True)
                ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(feat_size, feat_size, kernel_size=(
                    3, 3), padding=(1, 1), bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_size, feat_size, kernel_size=(
                    3, 3), padding=(1, 1), bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(feat_size, feat_size, kernel_size=(
                    3, 3), padding=(1, 1), bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(feat_size, num_classes, kernel_size=(1, 1))
            ))

        self.res1 = nn.Sequential(
            nn.Conv2d(feat_size * len(self._ppm_scales),
                      feat_size, kernel_size=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(feat_size, feat_size, kernel_size=(
                3, 3), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_size, feat_size, kernel_size=(
                3, 3), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Sequential(
            nn.Conv2d(feat_size, feat_size, kernel_size=(
                3, 3), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(feat_size, num_classes, kernel_size=(1, 1))
        )

    def memory_choose(self, supp_feat, query_feat):
        if (self.memory_network is None):
            return supp_feat
        else:
            supp_feat_new = self.memory_network.renew_feat_matrix(supp_feat, query_feat)
            return supp_feat_new

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            if module in {self.layer0, self.layer1, self.layer2, self.layer3, self.layer4}:
                continue
            module.train(mode)
        return self

    def forward(self, sx, sy, qx):
        """PFENet

        :param sx: dtype=float32, shape=(n, k, c, h, w)
        :param sy: dtype=int64, shape=(n, k, h, w)
        :param qx: dtype=float32, shape=(n, c, h, w)
        :return: out: dtype=float32, shape=(n, num_classes, output_h, output_w)
                 out_aux: dtype=float32, shape=(num_scales, n, num_classes, output_h, output_w)
        """
        # get the deep feature of the query sample
        # query_feat_4 is used for generate the prior mask
        # query_feat is used for fusion and prediction
        with torch.no_grad():
            query_feat_0 = self.layer0(qx)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)  # (n, f, h4, w4)
            if query_feat_2.shape[2:4] != query_feat_3.shape[2:4]:
                query_feat_2 = resize(
                    query_feat_2, (query_feat_3.size(2), query_feat_3.size(3)))
        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)  # (n, d, h3, w3)

        # get the deep feature of the support sample
        # supp_feat_4 is used for generate the prior mask
        # supp_feat is used for fusion and prediction
        # not that supp_feat_4 and supp_feat have different shape
        # (n, k, c, h, w) -> (nk, c, h, w)
        sx_flat = sx.view((-1, *sx.shape[2:]))
        # (n, k, h, w) -> (nk, 1, h, w)
        sy_flat = sy.float().view((-1, 1, *sy.shape[2:]))
        with torch.no_grad():
            supp_feat_0 = self.layer0(sx_flat)
            supp_feat_1 = self.layer1(supp_feat_0)
            supp_feat_2 = self.layer2(supp_feat_1)
            supp_feat_3 = self.layer3(supp_feat_2)
            supp_feat_3_hw = (supp_feat_3.size(2), supp_feat_3.size(3))
            supp_feat_3_mask = resize(sy_flat, supp_feat_3_hw)
            supp_feat_4 = self.layer4(supp_feat_3 * supp_feat_3_mask)
            supp_feat_4_hw = (supp_feat_4.size(2), supp_feat_4.size(3))
            supp_feat_4_mask = resize(sy_flat, supp_feat_4_hw)
            supp_feat_4 = supp_feat_4 * supp_feat_4_mask  # (nk, f, h4, w4)
            if supp_feat_2.shape[2:4] != supp_feat_3.shape[2:4]:
                supp_feat_2 = resize(
                    supp_feat_2, (supp_feat_3.size(2), supp_feat_3.size(3)))
        supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)  # (nk,f,h3,w3)
        supp_feat = self.down_supp(supp_feat)
        supp_feat_fg = self._weighted_gap(
            supp_feat, supp_feat_3_mask)  # (nk, d, 1, 1)
        supp_feat_bg = self._weighted_gap(
            supp_feat, (1.0 - supp_feat_3_mask))  # (nk, d, 1, 1)
        supp_feat = torch.cat([supp_feat_fg, supp_feat_bg], 1)

        supp_feat = supp_feat.view(
            (sx.size(0), -1, *supp_feat.size()[1:]))  # (n, k, 2d, 1, 1)
        supp_feat = supp_feat.mean(1)  # (n, 2d, 1, 1)
        # 加入 memory network
        query_feat_in = query_feat
        query_feat_in = query_feat_in.mean((2, 3))
        supp_feat = self.memory_choose(supp_feat, query_feat_in)
          
        feat_size = (query_feat.size(2), query_feat.size(3))
        prior = self._make_prior(supp_feat_4, query_feat_4)
        pyramid_feat_list, aux_list = self._pyramid(
            supp_feat, query_feat, prior)
        feat = torch.cat([resize(pyramid_feat, feat_size)
                         for pyramid_feat in pyramid_feat_list], 1)

        feat = self.res1(feat)
        feat = self.res2(feat) + feat
        output = self.cls(feat)

        return output, aux_list

    @staticmethod
    def _weighted_gap(supp_feat, mask, eps=1e-4):
        # supp_feat: (n, d, h, w)
        # mask: (n, 1, h, w)
        weight = mask.mean((2, 3), keepdims=True) + eps
        return (supp_feat * mask).mean((2, 3), keepdims=True) / weight

    @staticmethod
    def _make_prior(supp_feat, query_feat):
        # supp_feat: (nk, c, h, w)
        # query_feat: (n, c, h, w)
        n, c, h, w = query_feat.shape
        query_feat = F.normalize(query_feat, 2, 1)
        supp_feat = F.normalize(supp_feat, 2, 1)
        query_feat = query_feat.view((n, c, -1))  # (n, c, hw)
        supp_feat = supp_feat.view(n, -1, c, h * w)  # (n, k, c, hw)
        sim = torch.einsum('nca,nkcb->nkab', query_feat,
                           supp_feat).max(3)[0]  # (n, k, hw)
        sim_min = sim.min(2, keepdim=True)[0]  # (n, k, 1)
        sim_max = sim.max(2, keepdim=True)[0]  # (n, k, 1)
        prior = (sim - sim_min) / (sim_max - sim_min + 1e-10)
        prior = prior.view(
            (prior.size(0), prior.size(1), h, w))  # (n, k, h, w)
        # prior = resize(prior, out_size)
        prior = prior.mean(1, keepdim=True)  # (n, 1, h, w)
        return prior

    def _pyramid(self, supp_feat, query_feat, prior):
        # supp_feat: (n, d, 1, 1)
        # query_feat: (n, d, h, w)
        pyramid_feat_list = []
        aux_list = []
        for idx, tmp_bin in enumerate(self._ppm_scales):
            if isinstance(tmp_bin, (tuple, list)):
                bin_h, bin_w = tmp_bin
            else:
                bin_h, bin_w = tmp_bin, tmp_bin
            if bin_h <= 1.0:
                bin_h = (int(query_feat.size(2) * bin_h),
                         int(query_feat.size(3) * bin_h))
            if bin_w < 1.0:
                bin_w = (int(query_feat.size(2) * bin_w),
                         int(query_feat.size(3) * bin_w))

            query_feat_bin = F.adaptive_avg_pool2d(
                query_feat, (bin_h, bin_w))  # (n, d, bin_h, bin_w)
            # (n, d, bin_h, bin_w)
            supp_feat_bin = supp_feat.expand(-1, -1, bin_h, bin_w)
            prior_bin = resize(prior, (bin_h, bin_w))  # (n, 1, bin_h, bin_w)
            # (n, 2d + 1, bin_h, bin_w)
            merge_feat_bin = torch.cat(
                [query_feat_bin, supp_feat_bin, prior_bin], 1)
            merge_feat_bin = self.init_merge[idx](
                merge_feat_bin)  # (n, d, bin_h, bin_w)

            if idx > 0:
                pre_feat_bin = pyramid_feat_list[idx - 1]  # .clone()
                pre_feat_bin = resize(pre_feat_bin, (bin_h, bin_w))
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                # (n, d, bin_h, bin_w)
                merge_feat_bin = self.alpha_conv[idx -
                                                 1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](
                merge_feat_bin) + merge_feat_bin  # (n, d, bin_h, bin_w)
            pyramid_feat_list.append(merge_feat_bin)

            if self.training:
                inner_out_bin = self.inner_cls[idx](
                    merge_feat_bin)  # (n, num_class, bin_h, bin_w)
                aux_list.append(inner_out_bin)

        return pyramid_feat_list, aux_list


class Loss(nn.Module):

    def __init__(self, eps=1e-10):
        super(Loss, self).__init__()
        self._eps = eps

    def forward(self, output, target, aux_list=None):
        """Focal loss.

        :param output: dtype=float32, shape=(n, c, ?, ?)
        :param target: dtype=int64, shape=(n, h, w)
        :param aux_list: list of dtype=float32, shape=(n, c, ?, ?)
        :return: loss
        """
        size = (target.size(1), target.size(2))

        target = F.one_hot(target, output.size(1)).float()  # (n, h, w, c)
        target = target.permute((0, 3, 1, 2))  # (n, c, h, w)
        output = resize(output, size)
        output = F.softmax(output, 1)
        p = (target * output).sum(1)
        loss = -torch.pow(1.0 - p, 3) * torch.log(p + self._eps)
        loss = loss.sum((1, 2)).mean()

        if aux_list:
            target = target.unsqueeze(0)  # (1, n, c, h, w)
            aux = torch.stack([resize(aux, size)
                              for aux in aux_list])  # (?, n, c, h, w)
            aux = F.softmax(aux, 2)

            p = (target * aux).sum(2)
            loss_aux = -torch.pow(1.0 - p, 3) * torch.log(p + self._eps)
            loss_aux = loss_aux.sum((2, 3)).mean()
            loss = loss + loss_aux

        return loss


def get_vgg16_layers(pretrained=True):
    from torchvision.models import vgg
    net = vgg.vgg16_bn(pretrained=pretrained)
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [net.features[idx]]
    for idx in layer1_idx:
        layers_1 += [net.features[idx]]
    for idx in layer2_idx:
        layers_2 += [net.features[idx]]
    for idx in layer3_idx:
        layers_3 += [net.features[idx]]
    for idx in layer4_idx:
        layers_4 += [net.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4, 512 + 256


def get_resnet18_layers(pretrained=True):
    from torchvision.models import resnet
    net = resnet.resnet18(pretrained=pretrained)
    layer0 = nn.Sequential(
        net.conv1, net.bn1, net.relu,
        net.maxpool
    )
    layer1 = net.layer1
    layer2 = net.layer2
    layer3 = net.layer3
    layer4 = net.layer4
    return layer0, layer1, layer2, layer3, layer4, 256 + 128


def get_resnet34_layers(pretrained=True):
    from torchvision.models import resnet
    net = resnet.resnet18(pretrained=pretrained)
    layer0 = nn.Sequential(
        net.conv1, net.bn1, net.relu,
        net.maxpool
    )
    layer1 = net.layer1
    layer2 = net.layer2
    layer3 = net.layer3
    layer4 = net.layer4
    return layer0, layer1, layer2, layer3, layer4, 256 + 128


def get_resnet50_layers(pretrained=True):
    from torchvision.models import resnet
    net = resnet.resnet50(pretrained=pretrained)
    layer0 = nn.Sequential(
        net.conv1, net.bn1, net.relu,
        net.maxpool
    )
    layer1 = net.layer1
    layer2 = net.layer2
    layer3 = net.layer3
    layer4 = net.layer4
    return layer0, layer1, layer2, layer3, layer4, 1024 + 512


def get_mobilenet_v2_layers(pretrained=True):
    net = models.mobilenet_v2(pretrained=pretrained)
    # net.load_state_dict(torch.load("./base_models_dict/mobilenet_v2-b0353104.pth"))
    # net.load_state_dict(torch.load("./model_mobilenet_v2_MNIST_Images_freeze1_epoch1000_curEpoch325"))

    layer0_idx = range(0, 1)
    layer1_idx = range(1, 3)
    layer2_idx = range(3, 5)
    layer3_idx = range(5, 8)
    layer4_idx = range(8, 19)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [net.features[idx]]
    for idx in layer1_idx:
        layers_1 += [net.features[idx]]
    for idx in layer2_idx:
        layers_2 += [net.features[idx]]
    for idx in layer3_idx:
        layers_3 += [net.features[idx]]
    for idx in layer4_idx:
        layers_4 += [net.features[idx]]
    layer0 = nn.Sequential(*layers_0)  # 224 -> 112, channel = 32
    layer1 = nn.Sequential(*layers_1)  # 112 -> 56, channel = 24
    layer2 = nn.Sequential(*layers_2)  # 56 -> 28, channel = 32
    layer3 = nn.Sequential(*layers_3)  # 28 -> 14, channel = 64
    layer4 = nn.Sequential(*layers_4)  # 14 -> 7, channel = 1280
    return layer0, layer1, layer2, layer3, layer4, 32 + 64


def main():
    model = PFENet(*get_vgg16_layers()).to(device)
    loss_fn = Loss()

    sx = torch.normal(0.0, 1.0, (4, 5, 3, 473, 473),
                      dtype=torch.float32).to(device)
    sy = torch.randint(0, 1, (4, 5, 473, 473), dtype=torch.int64).to(device)
    qx = torch.normal(0.0, 1.0, (4, 3, 473, 473),
                      dtype=torch.float32).to(device)
    qy = torch.ones(4, 473, 473, dtype=torch.int64).to(device)
    for i in range(20):
        model.train()
        out, out_aux = model(sx, sy, qx)

    return 0


if __name__ == '__main__':
    exit(main())