import os
import logging
import json

from omegaconf import DictConfig, OmegaConf
import hydra

from hydra.utils import get_original_cwd, to_absolute_path


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs/", config_name="AICC_2023"
)
def my_app(cfg: DictConfig):
    log.info(os.getcwd())
    log.info(OmegaConf.to_yaml(cfg))

    print(f"Current working directory : {os.getcwd()}")
    print(f"Orig working directory    : {get_original_cwd()}")
    print(f"to_absolute_path('foo')   : {to_absolute_path('foo')}")
    print(f"to_absolute_path('/foo')  : {to_absolute_path('/foo')}")

    # Save a hello.txt file in hydra's output directory
    output_dir = HydraConfig.get().run.dir
    with open(os.path.join(output_dir, "hello.txt"), "w") as f:
        f.write("Hello Hydra!")
    # Read the index in cfg.Indexing.IndexFile
    with open(cfg.Indexing.IndexFile, "r") as f:
        index = json.load(f)


if __name__ == "__main__":
    my_app()
