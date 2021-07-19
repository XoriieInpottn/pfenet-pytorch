#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-07-16
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


def resize(feat, size):
    return F.interpolate(feat, size=size, mode='bilinear', align_corners=True)


def weighted_gap(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class PFENet(nn.Module):

    def __init__(self,
                 backbone_feat_size=1024 + 512,
                 feat_size=256,
                 num_classes=2,
                 ppm_scales=(60, 30, 15, 8),
                 output_size=None):
        super(PFENet, self).__init__()
        self._ppm_scales = ppm_scales
        self._output_size = output_size

        resnet = resnet50(True)
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            # resnet.conv2, resnet.bn2, resnet.relu2,
            # resnet.conv3, resnet.bn3, resnet.relu3,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.down_query = nn.Sequential(
            nn.Conv2d(backbone_feat_size, feat_size, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(backbone_feat_size, feat_size, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.init_merge = nn.ModuleList()
        self.alpha_conv = nn.ModuleList()
        self.beta_conv = nn.ModuleList()
        self.inner_cls = nn.ModuleList()
        for i in range(len(self._ppm_scales)):
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(feat_size * 2 + 1, feat_size, kernel_size=(1, 1), padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            if i > 0:
                # alpha_conv is used to fuse the feature from the last scale
                # so at the first scale, there is nothing to conv
                self.alpha_conv.append(nn.Sequential(
                    nn.Conv2d(feat_size * 2, feat_size, kernel_size=(1, 1), padding=0, bias=False),
                    nn.ReLU(inplace=True)
                ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(feat_size, feat_size, kernel_size=(3, 3), padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_size, feat_size, kernel_size=(3, 3), padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(feat_size, feat_size, kernel_size=(3, 3), padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(feat_size, num_classes, kernel_size=(1, 1))
            ))

        self.res1 = nn.Sequential(
            nn.Conv2d(feat_size * len(self._ppm_scales), feat_size, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(feat_size, feat_size, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_size, feat_size, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Sequential(
            nn.Conv2d(feat_size, feat_size, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(feat_size, num_classes, kernel_size=(1, 1))
        )

    def forward(self, sx, sy, qx):
        """PFENet

        :param sx: dtype=float32, shape=(n, k, c, h, w)
        :param sy: dtype=int64, shape=(n, k, h, w)
        :param qx: dtype=float32, shape=(n, c, h, w)
        :return: out: dtype=float32, shape=(n, num_classes, output_h, output_w)
                 out_aux: dtype=float32, shape=(num_scales, n, num_classes, output_h, output_w)
        """
        # get the deep feature of the query sample
        # query_feat_4 is used for generate the pior mask
        # query_feat is used for fusion and prediction
        with torch.no_grad():
            query_feat_0 = self.layer0(qx)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)  # (n, f, h4, w4)
        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)  # (n, d, h3, w3)

        # get the deep feature of the support sample
        # supp_feat_4 is used for generate the pior mask
        # supp_feat is used for fusion and prediction
        # not that supp_feat_4 and supp_feat have different shape
        sx_flat = sx.view((-1, *sx.shape[2:]))  # (n, k, c, h, w) -> (nk, c, h, w)
        sy_flat = sy.float().view((-1, 1, *sy.shape[2:]))  # (n, k, h, w) -> (nk, 1, h, w)
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
        supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
        supp_feat = self.down_supp(supp_feat)
        supp_feat = weighted_gap(supp_feat, supp_feat_3_mask)  # (nk, d, 1, 1)
        supp_feat = supp_feat.view((sx.size(0), -1, *supp_feat.size()[1:]))  # (n, k, d, 1, 1)
        supp_feat = supp_feat.mean(1)  # (n, d, 1, 1)

        corr = self._make_corr(supp_feat_4, query_feat_4, (query_feat.size(2), query_feat.size(3)))
        pyramid_feat, aux_out = self._pyramid(supp_feat, query_feat, corr)

        feat = self.res1(pyramid_feat)
        feat = self.res2(feat) + feat
        out = self.cls(feat)

        if self._output_size is not None:
            out = resize(out, self._output_size)

        if self.training:
            return out, aux_out
        else:
            return out

    @staticmethod
    def _make_corr(supp_feat_4, query_feat_4, query_feat_hw):
        q = query_feat_4  # (n, f, h4, w4)
        s = supp_feat_4  # (nk, f, h4, w4)
        tmp_query = q
        tmp_query = tmp_query.view((q.size(0), q.size(1), -1))  # (n, f, h4w4)
        tmp_query = F.normalize(tmp_query, 2, 1)
        tmp_supp = s
        tmp_supp = tmp_supp.view(q.size(0), -1, s.size(1), s.size(2) * s.size(3))  # (n, k, f, h4w4)
        tmp_supp = F.normalize(tmp_supp, 2, 2)
        sim = torch.einsum('nda,nkdb->nkab', tmp_query, tmp_supp).max(3)[0]  # (n, k, a)
        sim_min = sim.min(2, keepdim=True)[0]  # (n, k, 1)
        sim_max = sim.max(2, keepdim=True)[0]  # (n, k, 1)
        corr = (sim - sim_min) / (sim_max - sim_min + 1e-10)
        corr = corr.view((corr.size(0), corr.size(1), q.size(2), q.size(3)))  # (n, k, h4, w4)
        corr = resize(corr, query_feat_hw)
        corr = corr.mean(1, keepdim=True)  # (n, 1, h4, w4)
        return corr

    def _pyramid(self, supp_feat, query_feat, corr_mask):
        # supp_feat: (nk, d, 1, 1)
        # query_feat: (n, d, h3, w3)
        query_feat_hw = (query_feat.size(2), query_feat.size(3))
        pyramid_feat_list = []
        out_list = []
        for idx, tmp_bin in enumerate(self._ppm_scales):
            if isinstance(tmp_bin, (tuple, list)):
                bin_h, bin_w = tmp_bin
            else:
                bin_h, bin_w = tmp_bin, tmp_bin
            if bin_h <= 1.0:
                bin_h = (int(query_feat.size(2) * bin_h), int(query_feat.size(3) * bin_h))
            if bin_w < 1.0:
                bin_w = (int(query_feat.size(2) * bin_w), int(query_feat.size(3) * bin_w))
            query_feat_bin = F.adaptive_avg_pool2d(query_feat, (bin_h, bin_w))  # (n, d, bin_h, bin_w)
            supp_feat_bin = supp_feat.expand(-1, -1, bin_h, bin_w)  # (n, d, bin_h, bin_w)
            corr_mask_bin = resize(corr_mask, (bin_h, bin_w))
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)  # (n, d, bin_h, _bin_w)

            if idx > 0:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = resize(pre_feat_bin, (bin_h, bin_w))
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin  # (n, d, bin_h, bin_w)

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin  # (n, d, bin_h, bin_w)
            if self.training:
                inner_out_bin = self.inner_cls[idx](merge_feat_bin)  # (n, num_class, bin_h, bin_w)
                inner_out_bin = resize(inner_out_bin, self._output_size)  # (n, num_class, output_h, output_w)
                out_list.append(inner_out_bin)

            merge_feat_bin = resize(merge_feat_bin, query_feat_hw)  # (n, d, h3, w3)
            pyramid_feat_list.append(merge_feat_bin)

        pyramid_feat = torch.cat(pyramid_feat_list, 1)
        if self.training:
            aux_out = torch.stack(out_list)  # (?, n, num_class, output_h, output_w)
            return pyramid_feat, aux_out
        return pyramid_feat, None


class Loss(nn.Module):

    def __init__(self, eps=1e-10):
        super(Loss, self).__init__()
        self._eps = eps

    def forward(self, pred, target, pred_aux=None):
        """Cross-entropy loss.

        :param pred: dtype=float32, shape=(n, c, h, w)
        :param target: dtype=int64, shape=(n, h, w)
        :param pred_aux: dtype=float32, shape=(?, n, c, h, w)
        :return: loss
        """
        target = F.one_hot(target, pred.size(1)).float()  # (n, h, w, c)
        target = target.permute((0, 3, 1, 2))  # (n, c, h, w)
        out = F.softmax(pred, 1)
        loss = -target * torch.log(out + self._eps) - (1.0 - target) * torch.log(1.0 - out + self._eps)
        loss = loss.sum(1).mean()

        target = target.unsqueeze(0)  # (1, n, c, h, w)
        pred_aux = F.softmax(pred_aux, 2)
        loss_aux = -target * torch.log(pred_aux + self._eps) - (1.0 - target) * torch.log(1.0 - pred_aux + self._eps)
        loss_aux = loss_aux.sum(2).mean(0).mean()

        return loss + loss_aux


def main():
    model = PFENet(output_size=(224, 224))
    sx = torch.normal(0.0, 1.0, (4, 5, 3, 224, 224), dtype=torch.float32)
    sy = torch.randint(0, 1, (4, 5, 224, 224), dtype=torch.int64)
    qx = torch.normal(0.0, 1.0, (4, 3, 224, 224), dtype=torch.float32)
    qy = torch.ones(4, 224, 224, dtype=torch.int64)

    l = Loss()

    model.train()
    out, out_aux = model(sx, sy, qx)
    print(l(out, qy, out_aux))

    return 0


if __name__ == '__main__':
    exit(main())
