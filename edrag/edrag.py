import argparse
import json

from indexing import basic_indexing
from embed import embed
from evaluation import evaluate_all
from metrics import compute_all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/AICC_2023.json",
    )  # type: ignore
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    basic_indexing(config)
    embed(config)
    evaluate_all(config)
    compute_all_metrics(config)


if __name__ == "__main__":
    main()
