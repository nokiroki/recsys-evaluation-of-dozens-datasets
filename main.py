"""Main module"""
import logging
from typing import Optional
import warnings

import numpy as np
np.float = float

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src import BaseRunner

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: Optional[DictConfig] = None) -> None:
    """Main function.

    Args:
        cfg (DictConfig): Hydra Config
    """
    runner: BaseRunner = instantiate(cfg["library"]["runner"])

    runner.run(cfg)

if __name__ == '__main__':
    main()
