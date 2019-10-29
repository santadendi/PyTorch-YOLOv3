import tensorflow as tf
import logging
import sys


class Logger(object):
    def __init__(self, log_dir: str):
        """
        Create a summary writer logging to log_dir
        Args:
            log_dir (str): path to save the tensorboard log files
        """
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def list_of_scalars_summary(self, tag_value_pairs: [], step: int):
        """Log scalar variables."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        self.writer.add_summary(summary, step)


def create_logger(log_save_path: str):
    """
    Create a logger for message handling. Print logs to both the console and the file

    Args:
        log_save_path (str): path to save the log files

    """
    format = '%(asctime)s %(levelname)s %(message)s'
    name = 'main_logger'

    logger = logging.getLogger(name)
    formatter = logging.Formatter(format)

    file_handler = logging.FileHandler(log_save_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    logger.setLevel(logging.INFO)

    return logger


def tensorboard_log_train(model, loss, batches_done: int, logger: Logger) -> None:
    tensorboard_log = []
    for j, yolo in enumerate(model.yolo_layers):
        for name, metric in yolo.metrics.items():
            if name != "grid_size":
                tensorboard_log += [(f"{name}_{j + 1}", metric)]
    tensorboard_log += [("loss", loss.item())]
    logger.list_of_scalars_summary(tensorboard_log, batches_done)
