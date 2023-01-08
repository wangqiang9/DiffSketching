import argparse
import os

import numpy as np
import torch as th
import torch.nn as nn
import torch.distributed as dist
from torchvision import transforms, utils
import math
from PIL import Image
from torch.utils import data
from pathlib import Path

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png'], augment_horizontal_flip = False):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

if th.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

def slerp(z1, z2, alpha):
    theta = th.acos(th.sum(z1 * z2) / (th.norm(z1) * th.norm(z2)))
    return (
        th.sin((1 - alpha) * theta) / th.sin(theta) * z1
        + th.sin(alpha * theta) / th.sin(theta) * z2
    )

def slerp_theta(z1, z2, theta):
    return math.cos(theta) * z1 + math.sin(theta) * z2

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
    model.eval()

    logger.log("interpolation sampling...")
    for j in range(200):

        z1 = th.randn(1, 3, args.image_size, args.image_size, device=dist_util.dev())
        z2 = th.randn(1, 3, args.image_size, args.image_size, device=dist_util.dev())

        classes = th.randint(low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev())

        theta = th.arange(0, 1, args.interval).to(z1.device)
        z_ = []
        for i in range(theta.size(0)):
            z_.append(slerp_theta(z1, z2, theta[i]))
        x = th.cat(z_, dim=0)
        xs = []
        model_kwargs = {}
        model_kwargs["y"] = classes
        ind = 0
        with th.no_grad():
            sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
            for i in range(x.size(0)):
                sample = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    clip_denoised=args.clip_denoised,
                    noise=x[i : i + 1],
                    model_kwargs=model_kwargs,
                )
                xs.append(sample)
                ind += 1
            save_x = th.cat(xs, dim=0)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=200,
        batch_size=1,
        use_ddim=False,
        appoint_class="shoe",    # sketchy class name in 125 classes
        image_root="",
        model_path="",
        save_path="",
        interval=0.14
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
