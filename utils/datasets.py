import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def limit_bboxes(x_min, y_min, x_max, y_max, img_w, img_h):
    x_min = np.clip(x_min, a_min=1, a_max=img_w - 1)
    y_min = np.clip(y_min, a_min=1, a_max=img_h - 1)
    x_max = np.clip(x_max, a_min=1, a_max=img_w - 1)
    y_max = np.clip(y_max, a_min=1, a_max=img_h - 1)
    return x_min, y_min, x_max, y_max


def yolo_to_pascal_format(yolo_bbox, img_h, img_w):
    x_central, y_central, w, h = yolo_bbox.T
    x_min = (x_central - w / 2) * img_w
    y_min = (y_central - h / 2) * img_h
    x_max = (x_central + w / 2) * img_w
    y_max = (y_central + h / 2) * img_h

    x_min, y_min, x_max, y_max = limit_bboxes(
        x_min, y_min, x_max, y_max, img_w=img_w, img_h=img_h
    )

    pascal_format_bbox = np.vstack((x_min, y_min, x_max, y_max)).T
    return pascal_format_bbox


def pascal_to_yolo_format(pascal_bbox, img_h, img_w):
    x_min, y_min, x_max, y_max = pascal_bbox.T

    x_min, y_min, x_max, y_max = limit_bboxes(
        x_min, y_min, x_max, y_max, img_w=img_w, img_h=img_h
    )

    x_central = (x_min + x_max) / 2 / img_w
    y_central = (y_min + y_max) / 2 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h

    yolo_format_bbox = np.vstack((x_central, y_central, w, h)).T
    return yolo_format_bbox


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(
        self,
        list_path: str,
        transform,
        img_size: int = 416,
        max_objects: Optional[int] = None,
        logger=None,
    ):

        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.transform = transform
        self.img_size = img_size
        self.logger = logger
        self.max_objects = max_objects

    def __len__(self) -> int:
        if self.max_objects is None:
            return len(self.img_files)

        return self.max_objects

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path).convert("RGB"))
        img_height, img_width, _ = img.shape

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        targets = None
        need_resize_image = True
        if os.path.exists(label_path):
            label_data = np.loadtxt(label_path).reshape(-1, 5)
            category_ids = label_data[:, 0]
            bboxes = label_data[:, 1:]
            width_arr = bboxes[:, 2]
            height_arr = bboxes[:, 3]

            """Check that we do not have bbox with zero height or width"""
            if all(width_arr > 0) and all(height_arr > 0):
                pascal_format_bboxes = yolo_to_pascal_format(
                    yolo_bbox=bboxes, img_h=img_height, img_w=img_width
                )

                augmented = self.transform(
                    image=img,
                    bboxes=pascal_format_bboxes.tolist(),
                    category_id=category_ids.tolist(),
                )

                img = transforms.ToTensor()(augmented["image"])
                need_resize_image = False

                targets = np.zeros((label_data.shape[0], 6))
                targets[:, 2:] = pascal_to_yolo_format(
                    pascal_bbox=np.array(augmented["bboxes"]),
                    img_h=img_height,
                    img_w=img_width,
                )
                targets[:, 1] = np.array(augmented["category_id"])
                targets = torch.from_numpy(targets).type_as(img)

        if need_resize_image:
            img = transforms.ToTensor()(img)
            img = resize(img, self.img_size)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))

        """Remove empty placeholder targets"""
        targets = [boxes for boxes in targets if boxes is not None]

        """Add sample index to targets"""
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        targets = torch.cat(targets, 0)
        imgs = torch.stack([img for img in imgs])
        return paths, imgs, targets
