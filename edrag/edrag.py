import logging

import hydra
from omegaconf import DictConfig

from indexing import basic_indexing
from embed import embed
from evaluation import evaluate_all
from metrics import compute_all_metrics


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs/",
    config_name="AICC_2023",
)
def main(config: DictConfig):

    log.info("Starting the eDRAG pipeline")
    basic_indexing(config)
    embed(config)
    evaluate_all(config)
    compute_all_metrics(config)


if __name__ == "__main__":
    main()
