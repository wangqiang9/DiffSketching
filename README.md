# DiffSketching: Sketch Control Image Synthesis with Diffusion Models

![Python 3.6](https://img.shields.io/badge/python-3.6-green) ![Pytorch 1.13](https://img.shields.io/badge/pytorch-1.13-green) ![MIT License](https://img.shields.io/badge/licence-MIT-green)

In this repository, you can find the official PyTorch implementation of [DiffSketching: Sketch Control Image Synthesis with Diffusion Models](https://bmvc2022.mpi-inf.mpg.de/0067.pdf) (BMVC2022). Supplements can be found [here](https://bmvc2022.mpi-inf.mpg.de/0067_supp.zip).

Authors: [Qiang Wang](https://scholar.google.com/citations?user=lXyi3t4AAAAJ&hl=en), [Di Kong](https://scholar.google.com/citations?user=7jUAmi8AAAAJ&hl=en), [Fengyin Lin](https://github.com/MercuryMUMU) and [Yonggang Qi](https://qugank.github.io/), Beijing University of Posts and Telecommunications, Beijing, China.

> Abstract: Creative sketch is a universal way of visual expression, but translating images from an abstract sketch is very challenging. Traditionally, creating a deep learning model for sketch-to-image synthesis needs to overcome the distorted input sketch without visual details, and requires to collect large-scale sketch-image datasets. We first study this task by using diffusion models. Our model matches sketches through the cross domain constraints, and uses a classifier to guide the image synthesis more accurately. Extensive experiments confirmed that our method can not only be faithful to user’s input sketches, but also maintain the diversity and imagination of synthetic image results. Our model can beat GAN-based method in terms of generation quality and human evaluation, and does not rely on massive sketch-image datasets. Additionally, we present applications of our method in image editing and interpolation.

![Fig.1](https://github.com/XDUWQ/DiffSketching/blob/main/images/all_generation_V5.png)
![Fig.2](https://github.com/XDUWQ/DiffSketching/blob/main/images/Fig2_V14.png)

## Datasets
### ImageNet
Please go to the [ImageNet official website](https://image-net.org/) to download the complete datasets.

### Sketchy
Please go to the [Sketchy official website](https://sketchy.eye.gatech.edu/) to download the datasets.

### QuickDraw
Please go to the [QuickDraw official website](https://github.com/googlecreativelab/quickdraw-dataset) to download the datasets. 

## Installation
The requirements of this repo can be found in [requirements.txt](https://github.com/XDUWQ/DiffSketching/blob/main/requirements.txt).

## Train

train the feature extraction network.
```
python scripts/feature_train.py 
```

train the photosketch network.
```
python scripts/photosketch_train.py --dataroot [path/to/sketchy-datasets] --model pix2pix  --which_model_netG resnet_9blocks  --which_model_netD global_np 
```

train the diffusion network.
```
python scripts/image_train.py --data_dir [path/to/imagenet-datasets] --iterations 1000000 --anneal_lr True --batch_size 512 --lr 4e-4 --save_interval 10000 --weight_decay 0.05
```

train the classifier network.
```
python scripts/classifier_train.py --data_dir [path/to/imagenet-datasets]  --iterations 1000000 --anneal_lr True --batch_size 512 --lr 4e-4 --save_interval 10000 --weight_decay 0.05 --image_size 256 --classifier_width 256 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True
```

## Inference
```
python scripts/image_sample.py --model_path [/path/to/model] --image_root [/path/to/reference-image] --sketch_root [/path/to/reference-sketch] --save_path [/path/to/save] --batch_size 4 --num_samples 50000 --timestep_respacing ddim25 --use_ddim True --class_cond True --image_size 256 --learn_sigma True --use_fp16 True --use_scale_shift_norm True
```

## Interpolation
```
python scripts/interpolation.py --interval 0.14 --appoint_class [category-name] --save_path [path/to/save] --model_path [path/to/model]--use_ddim True 
```

## Evaluation
Please package the results to be evaluated in `.npz` format.
```
python evaluations/evaluator.py [/path/to/reference-data] [/path/to/generate-data]
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

## Acknowledgements
* [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
* [PhotoSketch](https://github.com/mtli/PhotoSketch)
* [SketchyDatabase](https://github.com/CDOTAD/SketchyDatabase)
* [guided-diffusion](https://github.com/openai/guided-diffusion)
