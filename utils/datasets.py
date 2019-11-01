import glob
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import albumentations as albu
import cv2

from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFile
from typing import Optional
from utils.utils import yolo_to_pascal_format, pascal_to_yolo_format

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def get_transform(img_size: int, darknet_pretrained: bool = False):
    """Initializes and returns data transformation pipeline

    Args:
        img_size (int): image size
        darknet_pretrained (bool): if you use a pre-trained darknet model, you need to disable image normalization

    """

    normalize = albu.Normalize(p=1)
    if darknet_pretrained:
        normalize = albu.Normalize(p=0)

    train_transform = albu.Compose(
        [
            albu.RandomResizedCrop(
                scale=(0.9, 1.0), height=img_size, width=img_size, p=1,
            ),
            albu.ShiftScaleRotate(
                border_mode=cv2.BORDER_CONSTANT,
                rotate_limit=10,
                scale_limit=0,
                p=0.5,
                mask_value=255,
            ),
            albu.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            normalize,
        ],
        bbox_params=albu.BboxParams(format="pascal_voc", label_fields=["category_id"]),
    )

    test_transform = albu.Compose(
        [albu.Resize(height=img_size, width=img_size, p=1),
         normalize
         ],
        bbox_params=albu.BboxParams(format="pascal_voc", label_fields=["category_id"]),
    )

    return train_transform, test_transform


class ImageFolder(Dataset):
    def __init__(self, folder_path: str, img_size: int = 416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]

        img = transforms.ToTensor()(Image.open(img_path))
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        list_path: path to file that contains train image paths
        transform (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, resize, etc.)
        img_size: size of returned img
        num_samples: number of samples in epoch
    """

    def __init__(
        self,
        list_path: str,
        transform,
        img_size: int = 416,
        num_samples: Optional[int] = None,
    ):

        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels")
            .replace(".png", ".txt")
            .replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.transform = transform
        self.img_size = img_size
        self.max_objects = num_samples

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

            """Check that we don't have bbox with zero height or width"""
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

                augmented_bboxes = augmented["bboxes"]

                """Create `targets` array if we have bboxes in image after augmentations"""
                if len(augmented_bboxes) > 0:
                    targets = np.zeros((len(augmented_bboxes), 6))

                    targets[:, 2:] = pascal_to_yolo_format(
                        pascal_bbox=np.array(augmented_bboxes),
                        img_h=self.img_size,
                        img_w=self.img_size,
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
