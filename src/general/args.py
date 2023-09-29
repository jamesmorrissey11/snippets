import argparse
from datetime import datetime


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
