"""Base class for dataset preparation"""
from abc import ABC, abstractmethod
from typing import Optional, Union
from pathlib import Path

from omegaconf import DictConfig

import pandas as pd


class BaseDataset(ABC):
    """Base class for the dataset preporation

    Can produce either single dense interaction dataset, or tuple with the
    users and items metainformation.

    Sparse matrix creation and usage of the features rely on the model implementation.
    """

    def __init__(
        self,
        saving_format: str = 'parquet'
    ) -> None:
        self._prepared_data: Optional[
            Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
        ] = None
        self._saving_format = saving_format

    @abstractmethod
    def prepare(self, properties: DictConfig) -> None:
        """Prepare data for the future work.

        Args:
            properties (dict): Mapping of the properties, unique for the each dataset.
            IMPORTANT. Case with the empty dict (deafult) need to be implemented

        Raises:
            NotImplementedError: Require implementation in the subclass.
        """
        raise NotImplementedError

    @property
    def prepared_data(self) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Get the prepared data. If no prepare was conducted, will use empty dict to initialize.

        Returns:
            Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
            return either [ratings, meta_user, meta_item] or only ratings
        
        Raises:
            ValueError: Require prepare method call.
        """
        if self._prepared_data is None:
            raise ValueError('Call prepare method first!')
        return self._prepared_data

    def _to_format(self, data: pd.DataFrame, path: Path, **params) -> None:
        if self._saving_format == 'csv':
            data.to_csv(path, **params)
        elif self._saving_format == 'parquet':
            data.to_parquet(path, **params)

    def _read_format(self, path: Path, **params) -> pd.DataFrame:
        if self._saving_format == 'csv':
            return pd.read_csv(path, **params)
        if self._saving_format == 'parquet':
            return pd.read_parquet(path, **params)
        else:
            raise ValueError("Not supported format")
