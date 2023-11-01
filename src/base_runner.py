"""Base Module"""
from abc import ABC, abstractmethod

from omegaconf import DictConfig


class BaseRunner(ABC):
    """Abstract base runner."""

    @staticmethod
    @abstractmethod
    def run(cfg: DictConfig) -> None:
        """Model runner.

        Args:
            cfg (DictConfig): OmegaConf config with the parameters.
        """
        raise NotImplementedError
