from __future__ import division

import os
import time
import datetime
import argparse
import torch
import albumentations as albu

from models import Darknet
from utils.logger import create_logger, Logger, tensorboard_log_train
from utils.utils import load_classes, weights_init_normal
from utils.datasets import ListDataset
from utils.parse_config import parse_data_config
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    for dir_name in ["output", "logs", "trained_models", "tensorboard_logs"]:
        os.makedirs(dir_name, exist_ok=True)

    tensorboard_logger = Logger("tensorboard_logs")
    date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_filename = date_str + "_train_logs.log"
    log_save_path = "logs/" + log_filename
    logger = create_logger(log_save_path=log_save_path)

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    test_transform = albu.Compose(
        [
            albu.Resize(height=opt.img_size, width=opt.img_size, p=1),
        ],
        bbox_params=albu.BboxParams(format="pascal_voc", label_fields=["category_id"]),
    )
    train_transform = albu.Compose(
        [
            albu.HorizontalFlip(p=0),
            albu.Resize(height=opt.img_size, width=opt.img_size, p=1),
        ],
        bbox_params=albu.BboxParams(format="pascal_voc", label_fields=["category_id"]),
    )
    dataset = ListDataset(
        list_path=train_path,
        transform=train_transform,
        img_size=opt.img_size,
        max_objects=None,
        logger=logger,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            """
            ----------------
               Log process
            ----------------
            """
            if batch_i % opt.print_every == 0:
                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----" % (
                    epoch,
                    opt.epochs,
                    batch_i,
                    len(dataloader),
                )

                metric_table = [
                    [
                        "Metrics",
                        *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))],
                    ]
                ]

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                    if tensorboard_logger:
                        tensorboard_log_train(
                            model=model,
                            loss=loss,
                            batches_done=batches_done,
                            logger=tensorboard_logger,
                        )

                metric_table_str = AsciiTable(metric_table).table
                log_str += f"\nTotal loss {loss.item()}"

                """Determine approximate time left for epoch"""
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(
                    seconds=epoch_batches_left
                    * (time.time() - start_time)
                    / (batch_i + 1)
                )
                log_str += f"\n---- ETA {time_left}\n"

                logger.info(log_str)
                print(metric_table_str)

        if epoch % opt.evaluation_interval == 0:
            logger.info("\n---- Evaluating Model ----")

            """Evaluate the model on the validation set"""
            precision, recall, AP, f1, ap_class = evaluate(
                model=model,
                path=valid_path,
                transform=test_transform,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            tensorboard_logger.list_of_scalars_summary(
                tag_value_pairs=evaluation_metrics, step=epoch
            )

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            logger.info(AsciiTable(ap_table).table)
            logger.info(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
