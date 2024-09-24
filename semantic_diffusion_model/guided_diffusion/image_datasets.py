import glob
import math
import os
import random

import blobfile as bf
import numpy as np
from PIL import Image, ImageOps
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



def load_data(cfg):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    """

    if not cfg.DATASETS.DATADIR:
        raise ValueError("unspecified data directory")

    if cfg.DATASETS.DATASET_MODE == 'cityscapes':
        all_files = _list_image_files_recursively(
            os.path.join(cfg.DATASETS.DATADIR, 'leftImg8bit', 'train' if cfg.TRAIN.IS_TRAIN else 'val'))
        labels_file = _list_image_files_recursively(
            os.path.join(cfg.DATASETS.DATADIR, 'gtFine', 'train' if cfg.TRAIN.IS_TRAIN else 'val'))
        classes = [x for x in labels_file if x.endswith('_labelIds.png')]
        instances = [x for x in labels_file if x.endswith('_instanceIds.png')]
    
    elif cfg.DATASETS.DATASET_MODE == 'ade20k':
        all_files = _list_image_files_recursively(
            os.path.join(cfg.DATASETS.DATADIR, 'images', 'training' if cfg.TRAIN.IS_TRAIN else 'validation'))
        classes = _list_image_files_recursively(
            os.path.join(cfg.DATASETS.DATADIR, 'annotations', 'training' if cfg.TRAIN.IS_TRAIN else 'validation'))
        instances = None

    elif cfg.DATASETS.DATASET_MODE == 'camus':
        if cfg.TEST.INFERENCE_ON_TRAIN and not cfg.TRAIN.IS_TRAIN:  # inference on train in one go to make synthetic image generation easier
            all_files = glob.glob(os.path.join(cfg.DATASETS.DATADIR, 'images', 'training', '*.png'))
            all_files = all_files + glob.glob(os.path.join(cfg.DATASETS.DATADIR, 'images', 'validation', '*.png'))
            all_files.sort()
            classes = glob.glob(os.path.join(cfg.DATASETS.DATADIR, 'sector_annotations', 'training', '*.png'))
            classes = classes + glob.glob(
                os.path.join(cfg.DATASETS.DATADIR, 'sector_annotations', 'validation', '*.png'))
            classes.sort()
            instances = None
        else:
            print("The images are in directory:", os.path.join(cfg.DATASETS.DATADIR, 'images', 'training'))
            all_files = _list_image_files_recursively(
                os.path.join(cfg.DATASETS.DATADIR, 'images', 'training' if cfg.TRAIN.IS_TRAIN else 'validation'))
            print("The labels are in directory:", os.path.join(cfg.DATASETS.DATADIR, 'sector_annotations', 'training'))
            classes = _list_image_files_recursively(
                os.path.join(cfg.DATASETS.DATADIR, 'sector_annotations',
                             'training' if cfg.TRAIN.IS_TRAIN else 'validation'))
            instances = None
    
    elif cfg.DATASETS.DATASET_MODE == 'camus_full_2CH':
        path1 = os.path.join(cfg.DATASETS.DATADIR, '2CH_ED_augmented', 'images',
                             'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        path2 = os.path.join(cfg.DATASETS.DATADIR, '2CH_ES_augmented', 'images',
                             'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        path1_label = os.path.join(cfg.DATASETS.DATADIR, '2CH_ED_augmented', 'sector_annotations',
                                   'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        path2_label = os.path.join(cfg.DATASETS.DATADIR, '2CH_ES_augmented', 'sector_annotations',
                                   'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        print("The images are in directories:", path1, path2)
        all_files = _list_image_files_recursively(path1) + _list_image_files_recursively(path2)
        print("The labels are in directory:", path1_label, path2_label)
        classes = _list_image_files_recursively(path1_label) + _list_image_files_recursively(path2_label)
        instances = None
    
    elif cfg.DATASETS.DATASET_MODE == 'camus_full_4CH':
        path1 = os.path.join(cfg.DATASETS.DATADIR, '4CH_ED_augmented', 'images',
                             'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        path2 = os.path.join(cfg.DATASETS.DATADIR, '4CH_ES_augmented', 'images',
                             'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        path1_label = os.path.join(cfg.DATASETS.DATADIR, '4CH_ED_augmented', 'sector_annotations',
                                   'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        path2_label = os.path.join(cfg.DATASETS.DATADIR, '4CH_ES_augmented', 'sector_annotations',
                                   'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        print("The images are in directories:", path1, path2)
        all_files = _list_image_files_recursively(path1) + _list_image_files_recursively(path2)
        print("The labels are in directory:", path1_label, path2_label)
        classes = _list_image_files_recursively(path1_label) + _list_image_files_recursively(path2_label)
        instances = None
    
    elif cfg.DATASETS.DATASET_MODE == 'camus_full_2CH_4CH':
        path1 = os.path.join(cfg.DATASETS.DATADIR, '2CH_ED_augmented', 'images',
                             'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        path2 = os.path.join(cfg.DATASETS.DATADIR, '2CH_ES_augmented', 'images',
                             'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        path3 = os.path.join(cfg.DATASETS.DATADIR, '4CH_ED_augmented', 'images',
                             'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        path4 = os.path.join(cfg.DATASETS.DATADIR, '4CH_ES_augmented', 'images',
                             'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        path1_label = os.path.join(cfg.DATASETS.DATADIR, '2CH_ED_augmented', 'sector_annotations',
                                   'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        path2_label = os.path.join(cfg.DATASETS.DATADIR, '2CH_ES_augmented', 'sector_annotations',
                                   'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        path3_label = os.path.join(cfg.DATASETS.DATADIR, '4CH_ED_augmented', 'sector_annotations',
                                   'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        path4_label = os.path.join(cfg.DATASETS.DATADIR, '4CH_ES_augmented', 'sector_annotations',
                                   'training' if cfg.TRAIN.IS_TRAIN else 'validation')
        print("The images are in directories:", path1, path2, path3, path4)
        all_files = _list_image_files_recursively(path1) + _list_image_files_recursively(
            path2) + _list_image_files_recursively(path3) + _list_image_files_recursively(path4)
        print("The labels are in directory:", path1_label, path2_label, path3_label, path4_label)
        classes = _list_image_files_recursively(path1_label) + _list_image_files_recursively(
            path2_label) + _list_image_files_recursively(path3_label) + _list_image_files_recursively(path4_label)
        instances = None
    
    elif cfg.DATASETS.DATASET_MODE == 'thyroid':
        # this should be the base path : "/home/data/farid/THYROID_MULTILABEL_2D_3D/imagesTrain/2D"
        all_files = []
        classes = []
        for i in range(0, 29, 2):
            folder = os.path.join(cfg.DATASETS.DATADIR, f"{str(i).zfill(2)}")
            images = sorted(glob.glob(os.path.join(folder, "images", "*.png")))
            labels = [os.path.join(folder, "labels", os.path.basename(image).replace("images", "labels")) for image in
                      images]

            for image, label in zip(images, labels):
                # if image or label is not there, then skip
                if not os.path.exists(image) or not os.path.exists(label):
                    continue
                all_files.append(image)
                classes.append(label)

        instances = None
        print("Number of samples in the dataset: ", len(all_files))
        print("Number of labels in the dataset: ", len(classes))

    elif cfg.DATASETS.DATASET_MODE == 'celeba':
        # The edge is computed by the instances.
        # However, the edge get from the labels and the instances are the same on CelebA.
        # You can take either as instance input
        all_files = _list_image_files_recursively(
            os.path.join(cfg.DATASETS.DATADIR, 'train' if cfg.TRAIN.IS_TRAIN else 'test', 'images'))
        classes = _list_image_files_recursively(
            os.path.join(cfg.DATASETS.DATADIR, 'train' if cfg.TRAIN.IS_TRAIN else 'test', 'labels'))
        instances = _list_image_files_recursively(
            os.path.join(cfg.DATASETS.DATADIR, 'train' if cfg.TRAIN.IS_TRAIN else 'test', 'labels'))
    else:
        raise NotImplementedError('{} not implemented'.format(cfg.DATASETS.DATASET_MODE))

    dataset = ImageDataset(
        cfg.DATASETS.DATASET_MODE,
        cfg.TRAIN.IMG_SIZE,
        all_files,
        classes=classes,
        instances=instances,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=cfg.TRAIN.RANDOM_CROP,
        random_flip=cfg.TRAIN.RANDOM_FLIP,
        is_train=cfg.TRAIN.IS_TRAIN
    )

    if cfg.TRAIN.IS_TRAIN:
        batch_size = cfg.TRAIN.BATCH_SIZE
        if cfg.TRAIN.DETERMINISTIC:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=True
            )
        else:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=True
            )
    else:
        batch_size = cfg.TEST.BATCH_SIZE
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=True
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


def random_crop_and_resize_image_label(image, label, resize_to=None):
    """
    Randomly crops the image and label to 80% or 90% of their original size from sides or bottom,
    keeping the top part aligned, and then resizes them back to the original size or to a specified size.

    Args:
    - image (PIL.Image): The input image to augment.
    - label (PIL.Image): The corresponding label image to augment.
    - resize_to (tuple, optional): The size to which the image and label should be resized. If None, uses the original size.

    Returns:
    - PIL.Image: The augmented and resized image.
    - PIL.Image: The augmented and resized label with nearest neighbor interpolation.
    """
    original_width, original_height = image.size
    resize_to = resize_to if resize_to else (original_width, original_height)

    # Choose a random crop size: 80% or 90% of the original dimensions
    crop_size = random.uniform(0.8, 0.95)
    new_height = int(original_height * crop_size)

    # Randomly choose the bottom crop boundary if cropping is from the bottom
    bottom_crop = original_height - new_height

    # Randomly choose how much to crop from the left (the rest will be cropped from the right)
    crop_width = int(original_width * crop_size)
    left_crop = random.randint(0, original_width - crop_width)

    # Define the crop box
    crop_box = (left_crop, 0, left_crop + crop_width, new_height)

    # Crop and resize the image
    cropped_image = image.crop(crop_box)
    #resized_image = cropped_image.resize(resize_to, Image.ANTIALIAS)
    # ANTIALIAS is no longer available in PIL: module 'PIL.Image' has no attribute 'ANTIALIAS'
    resized_image = cropped_image.resize(resize_to, Image.Resampling.LANCZOS)
    # This change adapts the code to be compatible with Pillow 7.0.0 and later. 
    # The LANCZOS filter is an excellent choice for resizing when quality is a priority, 
    # as it typically provides better results for image downscaling.

    # Crop and resize the label with nearest neighbor interpolation
    cropped_label = label.crop(crop_box)
    resized_label = cropped_label.resize(resize_to, Image.NEAREST)

    return resized_image, resized_label


class ImageDataset(Dataset):
    def __init__(
            self,
            dataset_mode,
            resolution,
            image_paths,
            classes=None,
            instances=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=True,
            is_train=True
    ):
        super().__init__()
        self.is_train = is_train
        self.dataset_mode = dataset_mode
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.local_instances = None if instances is None else instances[shard:][::num_shards]
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

        out_dict = {}
        class_path = self.local_classes[idx]
        with bf.BlobFile(class_path, "rb") as f:
            pil_class = Image.open(f)
            pil_class.load()
        pil_class = pil_class.convert("L")

        if self.local_instances is not None:
            instance_path = self.local_instances[idx]  # DEBUG: from classes to instances, may affect CelebA
            with bf.BlobFile(instance_path, "rb") as f:
                pil_instance = Image.open(f)
                pil_instance.load()
            pil_instance = pil_instance.convert("L")
        else:
            pil_instance = None

        if self.dataset_mode == 'cityscapes':
            arr_image, arr_class, arr_instance = resize_arr([pil_image, pil_class, pil_instance], self.resolution)

        if self.dataset_mode == 'thyroid':
            
            if self.is_train:
                # if random > 0.5, then crop and resize the image and label
                if random.random() > 0.5 and self.dataset_mode != 'camus' not in self.dataset_mode:
                    pil_image, pil_class = random_crop_and_resize_image_label(pil_image, pil_class)
                    if pil_instance is not None:
                        pil_instance = pil_instance.resize((self.resolution, self.resolution), Image.NEAREST)
                # Random horizontal flip with a 50% chance
                if random.random() > 0.5:
                    pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
                    pil_class = pil_class.transpose(Image.FLIP_LEFT_RIGHT)

        if self.dataset_mode == 'camus' or self.dataset_mode == 'camus_full_2CH' or self.dataset_mode == 'camus_full_4CH' \
                or self.dataset_mode == 'camus_full_2CH_4CH' or self.dataset_mode == 'thyroid':
            arr_image, arr_class, arr_instance = resize_arr([pil_image, pil_class, pil_instance], self.resolution,
                                                            keep_aspect=False)

        if self.random_flip and random.random() < 0.5:
            arr_image = arr_image[:, ::-1].copy()
            arr_class = arr_class[:, ::-1].copy()
            arr_instance = arr_instance[:, ::-1].copy() if arr_instance is not None else None

        arr_image = arr_image.astype(np.float32) / 127.5 - 1

        out_dict['path'] = path
        out_dict['label_ori'] = arr_class.copy()

        if self.dataset_mode == 'ade20k':
            arr_class = arr_class - 1
            arr_class[arr_class == 255] = 150
        elif self.dataset_mode == 'coco':
            arr_class[arr_class == 255] = 182

        out_dict['label'] = arr_class[None,]

        if arr_instance is not None:
            out_dict['instance'] = arr_instance[None,]

        return np.transpose(arr_image, [2, 0, 1]), out_dict


def resize_arr(pil_list, image_size, keep_aspect=True):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if keep_aspect:
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    else:
        pil_image = pil_image.resize((image_size, image_size), resample=Image.BICUBIC)

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    return arr_image, arr_class, arr_instance


def center_crop_arr(pil_list, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    crop_y = (arr_image.shape[0] - image_size) // 2
    crop_x = (arr_image.shape[1] - image_size) // 2
    return arr_image[crop_y: crop_y + image_size, crop_x: crop_x + image_size], \
        arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size], \
        arr_instance[crop_y: crop_y + image_size, crop_x: crop_x + image_size] if arr_instance is not None else None


def random_crop_arr(pil_list, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    crop_y = random.randrange(arr_image.shape[0] - image_size + 1)
    crop_x = random.randrange(arr_image.shape[1] - image_size + 1)
    return arr_image[crop_y: crop_y + image_size, crop_x: crop_x + image_size], \
        arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size], \
        arr_instance[crop_y: crop_y + image_size, crop_x: crop_x + image_size] if arr_instance is not None else None
