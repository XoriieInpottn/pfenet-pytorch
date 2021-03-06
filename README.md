# PFENet

This is the implementation of the good work [**PFENet: Prior Guided Feature Enrichment Network for Few-shot Segmentation**](http://arxiv.org/abs/2008.01449). Based on this paper, we rewrite the code for **higher training efficiency** and **better code reuse**. Thanks to the [official implementation](https://github.com/dvlab-research/PFENet).

## Dataset

In this paper, PASCAL-5i and COCO 2014 are used for evaluation.

We have preprocessed PASCAL-5i, and you can [download](https://pan.baidu.com/s/1lNR1scGg8MqpD94TeTaafQ) (pass code: b74i) it directly.

All datasets are stored in [DocSet](https://github.com/XoriieInpottn/docset) (*.ds) files, since this file format is portable and efficient. To support ds files in your system, install the "dcoset" package for your  python environment.

```bash
pip3 install docset
```

## Run The Code

```bash
python3 train.py --data-path /path/of/pascal5i.ds --gpu 0 --batch-size 8 --num-epochs 30
```

