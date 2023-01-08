import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
from torchvision import transforms, utils
import os

# sketchy datasets
def cycle(dl):
    while True:
        for data in dl:
            yield data

def make_dataset(root):
    images = []

    cnames = os.listdir(root)
    for cname in cnames:
        c_path = os.path.join(root, cname)
        if os.path.isdir(c_path):
            fnames = os.listdir(c_path)
            for fname in fnames:
                path = os.path.join(c_path, fname)
                images.append(path)

    return images

def make_appoint_dataset(root, appoint_class):
    images = []

    cnames = [appoint_class]
    for cname in cnames:
        c_path = os.path.join(root, cname)
        if os.path.isdir(c_path):
            fnames = os.listdir(c_path)
            for fname in fnames:
                path = os.path.join(c_path, fname)
                images.append(path)

    return images

def find_classes(root):
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    classes = ["zebra"]
    class_to_idex = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idex

def find_appoint_classes(root, appoint_class):
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    classes = [appoint_class]
    class_to_idex = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idex

class TripleDataset(data.Dataset):
    def __init__(self, photo_root, sketch_root, appoint_class="zebra"):
        super(TripleDataset, self).__init__()

        self.tranform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # tranform rgb to sketch
        self.sketch_tranform = transforms.Compose([transforms.functional.rgb_to_grayscale])

        # classes, class_to_idx = find_classes(photo_root)
        classes, class_to_idx = find_appoint_classes(photo_root, appoint_class)

        self.photo_root = photo_root
        self.sketch_root = sketch_root
        self.photo_paths = sorted(make_appoint_dataset(self.photo_root, appoint_class))
        self.class_to_idx = class_to_idx
        self.len = len(self.photo_paths)

    def __getitem__(self, index):

        photo_path = self.photo_paths[index]
        sketch_path, label = self._getrelate_sketch(photo_path)
        # print(photo_path, sketch_path)
        photo = Image.open(photo_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('RGB')

        P = self.tranform(photo)
        S = self.tranform(sketch)
        S = self.sketch_tranform(S) # tranform rgb to gray
        L = label
        return {'P': P, 'S': S, 'L': L}

    def __len__(self):
        return self.len

    def _getrelate_sketch(self, photo_path):

        paths = photo_path.split('/')
        fname = paths[-1].split('.')[0]
        cname = paths[-2]

        label = self.class_to_idx[cname]

        sketchs = sorted(os.listdir(os.path.join(self.sketch_root, cname)))

        sketch_rel = []
        for sketch_name in sketchs:
            if sketch_name.split('-')[0] == fname:
                sketch_rel.append(sketch_name)

        rnd = np.random.randint(0, len(sketch_rel))

        sketch = sketch_rel[rnd]

        return os.path.join(self.sketch_root, cname, sketch), label

if __name__ == "__main__":
    image_root = "../../sketchy/photo/tx_000000000000_ready"
    sketch_root = "../../sketchy/sketch/tx_000000000000_ready"
    ds_all = TripleDataset(image_root, sketch_root)
    import torch
    dl_all = cycle(torch.utils.data.DataLoader(ds_all, batch_size=4, shuffle=True, num_workers=4, drop_last=True,
                        pin_memory=True))
    while True:
        data = next(dl_all)
        print(data["S"].size())


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
