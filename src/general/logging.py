import json
import logging
import os
import warnings


def create_logger():
    my_logger = logging.getLogger()
    my_logger.setLevel(logging.INFO)
    return my_logger


def log_model_config(args, logger, log_dir):
    with open(
        os.path.join(log_dir, "config.json"),
        "w",
    ) as f:
        json.dump(vars(args), f)
    logger.info(f"Config stored at {log_dir}/config.json")


def mute_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
