import argparse
import os

import numpy as np
import torch as th
import torch.nn as nn
import torch.distributed as dist
from torchvision import transforms, utils
from PIL import Image
import torchvision
import lpips

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.image_datasets import TripleDataset, cycle
from photosketch.pix2pix_model import Pix2PixModel, OPT

if th.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

opt = OPT()
if not os.path.isdir(opt.results_dir):
    os.makedirs(opt.results_dir)

photosketch_model = Pix2PixModel()
photosketch_model.initialize(opt)

# Fine-tuned resnet50 feature extractor in sketchy
image_feature_path = "" #
sketch_feature_path = ""

resnet = torchvision.models.resnet50(pretrained=False)
image_feature_model = nn.Sequential(*list(resnet.children())[:-2])
image_feature_model = image_feature_model.cuda()

checkpoint = th.load(image_feature_path)
state_dict = image_feature_model.state_dict()
checkpoint = {k: v for k, v in checkpoint.items() if k in state_dict}
state_dict.update(checkpoint)
image_feature_model.load_state_dict(state_dict)

for param in image_feature_model.parameters():
    param.requires_grad = False

resnet = torchvision.models.resnet50(pretrained=False)
sketch_feature_model = nn.Sequential(*list(resnet.children())[:-2])
sketch_feature_model = sketch_feature_model.cuda()

checkpoint = th.load(sketch_feature_path)
state_dict = sketch_feature_model.state_dict()
checkpoint = {k: v for k, v in checkpoint.items() if k in state_dict}
state_dict.update(checkpoint)
sketch_feature_model.load_state_dict(state_dict)

for param in sketch_feature_model.parameters():
    param.requires_grad = False

loss_fn_alex = lpips.LPIPS(net='alex', eval_mode=True).to(dist_util.dev())  #

def image_feature(image_data):
    with th.no_grad():
        output = image_feature_model(image_data)
    return output

def sketch_feature(sketch_data):
    with th.no_grad():
        output = sketch_feature_model(sketch_data)  
    return output

def main():
    args = create_argparser().parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.train()

    optimizer = th.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    image_loss_origin = nn.MSELoss()
    image_loss = nn.L1Loss()
    sketch_loss = nn.MSELoss()

    logger.log("loading datasets……")
    ds_all = TripleDataset(args.image_root, args.sketch_root, args.appoint_class)
    dl_all = cycle(th.utils.data.DataLoader(ds_all, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                        pin_memory=True))

    logger.log("sampling...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:

        optimizer.zero_grad()

        data_all = next(dl_all)
        o_image = data_all["P"].cuda()
        o_sketch = data_all["S"].cuda()

        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        g_image_feature = image_feature(sample)
        o_image_feature = image_feature_model(o_image)

        photosketch_model.set_input(sample)
        g_sketch = photosketch_model.test().requires_grad_()

        loss_image = image_loss(g_image_feature, o_image_feature)
        loss_percept = loss_fn_alex(g_sketch, o_sketch).cuda()
        loss = loss_image + loss_percept

        loss.backward()
        optimizer.step()

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=200,
        batch_size=4,
        use_ddim=False,
        appoint_class="",
        image_root="",
        sketch_root="",
        model_path="",
        save_path=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
