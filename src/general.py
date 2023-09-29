import argparse
import json
import logging
import os
import warnings
from datetime import datetime

import boto3
import pypdf
from botocore.exceptions import ClientError


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


def write_documents_to_json(docs, json_path):
    split_json = [
        {"page_content": d.page_content, "metadata": d.metadata} for d in docs
    ]
    with open(json_path, "w") as f:
        json.dump(split_json, f)


def clone_repository(repo_dir, repo_url):
    try:
        os.system(f"git clone {repo_url} {repo_dir}")
    except Exception as e:
        print("Unable to clone repository")


def upload_dataset(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def upload_model(local_dir, bucket_name):
    s3_client = boto3.client("s3")
    model_version = local_dir.split("/")[-2]
    for root, _, files in os.walk(local_dir):
        for filename in files:
            path_to_local_file = os.path.join(root, filename)
            s3_path = f"models/{model_version}/{filename}"
            s3_client.upload_file(path_to_local_file, bucket_name, s3_path)


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
