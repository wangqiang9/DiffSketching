# DiffSketching: Sketch Control Image Synthesis with Diffusion Models
In this repository, you can find the official PyTorch implementation of [DiffSketching: Sketch Control Image Synthesis with Diffusion Models](https://bmvc2022.mpi-inf.mpg.de/0067.pdf) (BMVC2022).
![Fig.2](https://github.com/XDUWQ/DiffSketching/blob/main/images/Fig2_V14.png)

## Datasets
### ImageNet
Please go to the [ImageNet official website](https://image-net.org/) to download the complete datasets.

### Sketchy
Please go to the [Sketchy official website](https://sketchy.eye.gatech.edu/) to download the datasets.

### Train

train feature extraction network.
```
python scripts/feature_train.py 
```

train photosketch network.
```
python train.py --dataroot [sketchy-datasets] --model pix2pix  --which_model_netG resnet_9blocks  --which_model_netD global_np 
```

train diffusion network.
```
python scripts/image_train.py --data_dir [imagenet-datasets] --iterations 1000000 --anneal_lr True --batch_size 512 --lr 4e-4 --save_interval 10000 --weight_decay 0.05
```

train classifier network.
```
python scripts/classifier_train.py --data_dir [imagenet-datasets]  --iterations 1000000 --anneal_lr True --batch_size 512 --lr 4e-4 --save_interval 10000 --weight_decay 0.05 --image_size 256 --classifier_width 256 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True
```

## Citation
```
@inproceedings{Wang_2022_BMVC,
author    = {Qiang Wang and Di Kong and Fengyin Lin and Yonggang Qi},
title     = {DiffSketching: Sketch Control Image Synthesis with Diffusion Models},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0067.pdf}
}
```
