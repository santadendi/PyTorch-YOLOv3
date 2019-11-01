from __future__ import division

import argparse
import tqdm
import numpy as np
import torch
import albumentations as albu

from models import Darknet
from utils.utils import (
    xywh2xyxy,
    non_max_suppression,
    get_batch_statistics,
    ap_per_class,
    load_classes,
)
from utils.datasets import ListDataset
from utils.parse_config import parse_data_config
from torch.utils.data import DataLoader


def evaluate(
    model,
    path: str,
    transform,
    iou_thres: float = 0.5,
    conf_thres: float = 0.5,
    nms_thres: float = 0.5,
    img_size: int = 416,
    batch_size: int = 8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    dataset = ListDataset(
        list_path=path, transform=transform, img_size=img_size, num_samples=None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(
        tqdm.tqdm(dataloader, desc="Detecting objects")
    ):
        labels += targets[:, 1].tolist()

        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = imgs.requires_grad_(False).to(device)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(
                prediction=outputs, conf_thres=conf_thres, nms_thres=nms_thres
            )

        sample_metrics += get_batch_statistics(
            outputs=outputs, targets=targets, iou_threshold=iou_thres
        )

    # Concatenate sample statistics
    if sample_metrics:
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*sample_metrics))
        ]
    else:
        true_positives, pred_scores, pred_labels = [np.zeros(1), np.zeros(1), np.zeros(1)]

    precision, recall, AP, f1, ap_class = ap_per_class(
        true_positives, pred_scores, pred_labels, labels
    )

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    test_transform = albu.Compose(
        [albu.Resize(height=opt.img_size, width=opt.img_size, p=1)],
        bbox_params=albu.BboxParams(format="pascal_voc", label_fields=["category_id"]),
    )

    # Initiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model=model,
        path=valid_path,
        transform=test_transform,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
