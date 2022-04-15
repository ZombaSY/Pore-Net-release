import os
import platform

import torch
import torchvision.transforms.functional as tf
import random
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import ImageOps, ImageFilter
from models import utils
from multiprocessing import set_start_method


def is_image(src):
    ext = os.path.splitext(src)[1]
    return True if ext in ['.jpg', '.png', '.JPG', '.PNG'] else False


class ImageImageLoader(Dataset):

    def __init__(self,
                 train_x_path,
                 train_y_path,
                 is_val=False,
                 **kwargs):

        self.is_val = is_val
        self.args = kwargs['args']

        if hasattr(self.args, 'input_size'):
            h, w = self.args.input_size[0], self.args.input_size[1]
            self.size_1x = [int(h), int(w)]

        if hasattr(self.args, 'crop_size'):
            self.crop_factor = int(self.args.crop_size)

        self.image_mean = [0.512, 0.459, 0.353]
        self.image_std = [0.254, 0.226, 0.219]

        x_img_name = os.listdir(train_x_path)
        y_img_name = os.listdir(train_y_path)
        x_img_name = filter(is_image, x_img_name)
        y_img_name = filter(is_image, y_img_name)

        self.x_img_path = []
        self.y_img_path = []

        x_img_name = sorted(x_img_name)
        y_img_name = sorted(y_img_name)

        img_paths = zip(x_img_name, y_img_name)
        for item in img_paths:
            self.x_img_path.append(train_x_path + os.sep + item[0])
            self.y_img_path.append(train_y_path + os.sep + item[1])

        assert len(self.x_img_path) == len(self.y_img_path), 'Images in directory must have same file indices!!'

        self.len = len(x_img_name)

    def transform(self, image, target):
        # prevent exif rotate
        image = ImageOps.exif_transpose(image)
        target = ImageOps.exif_transpose(target)

        if hasattr(self.args, 'input_size'):
            image = tf.resize(image, self.size_1x)
            target = tf.resize(target, self.size_1x)

        if hasattr(self, 'offset') and self.is_val is not True:
            image_np = np.array(image)
            image_np = self.offset(image_np)
            image = Image.fromarray(image_np.astype(np.uint8))

        if hasattr(self.args, 'resolution'):
            image = utils.resize_img_by_resolution(image, maximum_resolution=self.args.resolution)
            target = utils.resize_img_by_resolution(target, maximum_resolution=self.args.resolution)

        if hasattr(self.args, 'crop_size') and not self.is_val:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_factor, self.crop_factor))
            image = tf.crop(image, i, j, h, w)
            target = tf.crop(target, i, j, h, w)

        if (random.random() > 0.5) and not self.is_val:
            image = tf.hflip(image)
            target = tf.hflip(target)

        if (random.random() < 0.3) and not self.is_val and self.args.transform_blur:
            image = image.filter(ImageFilter.GaussianBlur(radius=3))

        if (random.random() < 0.7) and not self.is_val and self.args.transform_jitter:
            transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
            image = transform(image)

        image = tf.to_tensor(image)
        target = torch.tensor(np.array(target))

        if not hasattr(self.args, 'input_space') or self.args.input_space == 'RGB':
            image = tf.normalize(image,
                                 mean=self.image_mean,
                                 std=self.image_std)
        target = target

        if not hasattr(self.args, 'input_space') or self.args.input_space == 'RGB':
            pass
        elif self.args.input_space == 'HSV':
            try:
                set_start_method('spawn')
                image = utils.ImageProcessing.rgb_to_hsv(image)
            except RuntimeError:
                pass

        return image, target

    def __getitem__(self, index):
        x_path = self.x_img_path[index]
        y_path = self.y_img_path[index]

        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path).convert('L')

        img_x, img_y = self.transform(img_x, img_y)

        return (img_x, x_path), (img_y, y_path)

    def __len__(self):
        return self.len


class ImageImageMaskLoader(Dataset):

    def __init__(self,
                 train_x_path,
                 train_y_path,
                 mask_path,
                 is_val=False,
                 **kwargs):

        self.is_val = is_val
        self.args = kwargs['args']

        if hasattr(self.args, 'input_size'):
            h, w = self.args.input_size[0], self.args.input_size[1]
            self.size_1x = [int(h), int(w)]

        if hasattr(self.args, 'crop_size'):
            self.crop_factor = int(self.args.crop_size)

        self.image_mean = [0.512, 0.459, 0.353]
        self.image_std = [0.254, 0.226, 0.219]

        x_img_name = os.listdir(train_x_path)
        y_img_name = os.listdir(train_y_path)
        mask_name = os.listdir(mask_path)

        self.x_img_path = []
        self.y_img_path = []
        self.mask_path = []

        x_img_name = sorted(x_img_name)
        y_img_name = sorted(y_img_name)
        mask_name = sorted(mask_name)

        img_paths = zip(x_img_name, y_img_name, mask_name)
        for item in img_paths:
            self.x_img_path.append(train_x_path + os.sep + item[0])
            self.y_img_path.append(train_y_path + os.sep + item[1])
            self.mask_path.append(mask_path + os.sep + item[2])

        assert len(self.x_img_path) == len(self.y_img_path), 'Images in directory must have same file indices!!'

        self.len = len(x_img_name)

    def transform(self, image, target, roi_mask):
        # prevent exif rotate
        image = ImageOps.exif_transpose(image)
        target = ImageOps.exif_transpose(target)

        if hasattr(self.args, 'input_size'):
            image = tf.resize(image, self.size_1x)
            target = tf.resize(target, self.size_1x)
            roi_mask = tf.resize(roi_mask, self.size_1x)

        if hasattr(self, 'offset') and self.is_val is not True:
            image_np = np.array(image)
            image_np = self.offset(image_np)
            image = Image.fromarray(image_np.astype(np.uint8))

        if hasattr(self.args, 'resolution'):
            image = utils.resize_img_by_resolution(image, maximum_resolution=self.args.resolution)
            target = utils.resize_img_by_resolution(target, maximum_resolution=self.args.resolution)
            roi_mask = utils.resize_img_by_resolution(roi_mask, maximum_resolution=self.args.resolution)

        if hasattr(self.args, 'crop_size') and not self.is_val:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_factor, self.crop_factor))
            image = tf.crop(image, i, j, h, w)
            target = tf.crop(target, i, j, h, w)
            roi_mask = tf.crop(roi_mask, i, j, h, w)

        if (random.random() > 0.5) and not self.is_val:
            image = tf.hflip(image)
            target = tf.hflip(target)
            roi_mask = tf.hflip(roi_mask)

        if (random.random() < 0.3) and not self.is_val and self.args.transform_blur:
            image = image.filter(ImageFilter.GaussianBlur(radius=3))

        if (random.random() < 0.7) and not self.is_val and self.args.transform_jitter:
            transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
            image = transform(image)

        image = tf.to_tensor(image)
        target = tf.to_tensor(target)
        roi_mask = tf.to_tensor(roi_mask)

        if not hasattr(self.args, 'input_space') or self.args.input_space == 'RGB':
            image = tf.normalize(image,
                                 mean=self.image_mean,
                                 std=self.image_std)

        target = target
        roi_mask = roi_mask

        if not hasattr(self.args, 'input_space') or self.args.input_space == 'RGB':
            pass
        elif self.args.input_space == 'HSV':
            try:
                set_start_method('spawn')
                image = utils.ImageProcessing.rgb_to_hsv(image)
            except RuntimeError:
                pass

        return image, target, roi_mask

    def __getitem__(self, index):
        x_path = self.x_img_path[index]
        y_path = self.y_img_path[index]
        mask_path = self.mask_path[index]

        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path).convert('L')
        img_mask = Image.open(mask_path).convert('L')

        img_x, img_y, img_mask = self.transform(img_x, img_y, img_mask)

        return (img_x, x_path), (img_y, img_mask)

    def __len__(self):
        return self.len


class SegLoader:

    def __init__(self,
                 dataset_path,
                 label_path,
                 batch_size=64,
                 num_workers=0,
                 pin_memory=True,
                 is_val=False,
                 mask_path=None,
                 **kwargs):

        if mask_path is None:
            self.image_loader = ImageImageLoader(dataset_path,
                                                 label_path,
                                                 is_val=is_val,
                                                 **kwargs)
            self.Loader = DataLoader(self.image_loader,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     shuffle=not is_val,
                                     pin_memory=pin_memory)
        else:
            self.image_loader = ImageImageMaskLoader(dataset_path,
                                                     label_path,
                                                     mask_path,
                                                     is_val=is_val,
                                                     **kwargs)
            self.Loader = DataLoader(self.image_loader,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     shuffle=not is_val,
                                     pin_memory=pin_memory)

    def __len__(self):
        return self.Loader.__len__()
