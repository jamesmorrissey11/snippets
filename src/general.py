import argparse
import json
import os
import warnings
from datetime import datetime

import pypdf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="location of json data"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="where to store vector db"
    )
    parser.add_argument(
        "--config_dir", type=str, required=True, help="where to store the model config"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.model_dir = os.path.join(args.model_dir, timestamp)

    return args


def log_model_config(args, logger, log_dir):
    with open(
        os.path.join(log_dir, "config.json"),
        "w",
    ) as f:
        json.dump(vars(args), f)
    logger.info(f"Config stored at {log_dir}/config.json")


def pdf_to_pages(file):
    "extract text (pages) from pdf file"
    pages = []
    pdf = pypdf.PdfReader(file)
    for p in range(len(pdf.pages)):
        page = pdf.pages[p]
        text = page.extract_text()
        pages += [text]
    return pages


def mute_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)


def remove_patterns(text, patterns_to_remove):
    for pattern in patterns_to_remove:
        text = text.replace(pattern, "")
    return text
