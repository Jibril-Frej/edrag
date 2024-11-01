import argparse
import json

from edrag.indexing import basic_indexing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/indexing.json",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    basic_indexing(config)


if __name__ == "__main__":
    main()
