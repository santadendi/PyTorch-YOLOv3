from __future__ import division

import os
import time
import datetime
import argparse
import torch

from models import Darknet
from utils.logger import create_logger, Logger, tensorboard_log_train
from utils.utils import load_classes, weights_init_normal
from utils.datasets import ListDataset, get_transform
from utils.parse_config import parse_data_config, get_constants
from test import evaluate
from terminaltables import AsciiTable
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="size of each image batch"
    )
    parser.add_argument(
        "--gradient_accumulations",
        type=int,
        default=2,
        help="number of gradient accums before step",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="config/yolov3.cfg",
        help="path to model definition file",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="config/coco.data",
        help="path to data config file",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        help="if specified starts from checkpoint model",
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=4,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--img_size", type=int, default=416, help="size of each image dimension"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="interval between saving model weights",
    )
    parser.add_argument(
        "--evaluation_interval",
        type=int,
        default=1,
        help="interval evaluations on validation set",
    )
    parser.add_argument(
        "--print_every", type=int, default=100, help="metric printing interval"
    )
    parser.add_argument(
        "--compute_map", default=False, help="if True computes mAP every tenth batch"
    )
    parser.add_argument(
        "--multiscale_training", default=True, help="allow for multi-scale training"
    )
    opt = parser.parse_args()
    print(opt)

    for dir_name in ["output", "logs", "trained_models", "tensorboard_logs"]:
        os.makedirs(dir_name, exist_ok=True)

    tensorboard_logger = Logger("tensorboard_logs")
    date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_filename = date_str + "_train_logs.log"
    log_save_path = "logs/" + log_filename
    logger = create_logger(log_save_path=log_save_path)

    """Get data configuration"""
    data_config = parse_data_config(path=opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    trained_models_dir = "trained_models/" + date_str + "/"

    """Initiate model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    """If specified we start from checkpoint"""
    if opt.pretrained_weights:
        print("load pretrained weights")
        if opt.pretrained_weights.endswith(".pth"):
            """Load darknet weights"""
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            """Load checkpoint weights"""
            model.load_darknet_weights(opt.pretrained_weights)

    train_transform, test_transform = get_transform(img_size=opt.img_size)
    dataset = ListDataset(
        list_path=train_path,
        transform=train_transform,
        img_size=opt.img_size,
        num_samples=None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)
    metrics, formats = get_constants()

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device)
            targets = targets.requires_grad_(False).to(device)

            loss, outputs = model(imgs, targets)
            loss.backward()

            model.seen += imgs.size(0)

            if batches_done % opt.gradient_accumulations:
                """Accumulates gradient before each step"""
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

                """Log metrics at each YOLO layer"""
                for i, metric in enumerate(metrics):
                    row_metrics = [
                        formats[metric] % yolo.metrics.get(metric, 0)
                        for yolo in model.yolo_layers
                    ]
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
                batch_size=12,
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

            """Print class APs and mAP"""
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            logger.info(AsciiTable(ap_table).table)
            logger.info(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            if not os.path.isdir(trained_models_dir):
                os.mkdir(trained_models_dir)
            save_path = trained_models_dir + "/" + "yolov3_ckpt_{}.pth".format(epoch)
            torch.save(model.state_dict(), save_path)
