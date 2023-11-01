"""Classic Dataset Module"""
from typing import Optional, Tuple, Iterable
from pathlib import Path
import pickle
import logging

from omegaconf import DictConfig

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from .data_preporation import load_dataset
from .base_dataset import BaseDataset

logger = get_logger(name=__name__, log_level=logging.DEBUG)


class ClassicDataset(BaseDataset):
    """Classic Dataset that suits a plenty of cases
    """

    def prepare(self, properties: DictConfig) -> None:
        name: str = properties["name"]

        preproc_data_folder = Path('/'.join(("preproc_data", name)))
        if not preproc_data_folder.exists():
            logger.info('Dataset folder does not exist. Creating dataset.')
            preproc_data_folder.mkdir(parents=True)

        params_file = preproc_data_folder.joinpath("params.pcl")

        if params_file.exists():
            with params_file.open('rb') as file:
                properties_saved = pickle.load(file)
        else:
            properties_saved = None

        flag = False
        files = self._check_integrity(preproc_data_folder)
        if properties != properties_saved:
            logger.info("Parameters were changed. New preprocessing was started")
            flag = True
        elif files is not None:
            logger.info((''.join((
                "Following files are missing:\n",
                '\n'.join(files),
                "\nNew preprocessing was started"
            ))))

            flag = True

        if flag:
            with params_file.open('wb') as file:
                pickle.dump(properties, file)
            ratings = self._new_data_prepare(properties, preproc_data_folder)
            if properties['meta_user'] and properties['meta_item']:
                meta_user, meta_item = self._new_meta_prepare(properties, preproc_data_folder)
            else:
                meta_user, meta_item = None, None
            self._save_data((ratings, meta_user, meta_item), properties, preproc_data_folder)
        else:
            ratings, meta_user, meta_item = self._load_data(properties, preproc_data_folder)

        if meta_user is None and meta_item is None:
            self._prepared_data = ratings
        else:
            self._prepared_data = (ratings, meta_user, meta_item)


    def _check_integrity(self, preproc_data_folder: Path) -> Optional[Iterable[str]]:
        files_to_check = set((
            "item2order.pcl",
            "order2item.pcl",
            "user2order.pcl",
            "order2user.pcl",
            "params.pcl",
            f"ratings.{self._saving_format}"
        ))
        for file in preproc_data_folder.iterdir():
            if file.name in files_to_check:
                files_to_check.remove(file.name)
        if files_to_check:
            return tuple(files_to_check)



    def _load_data(
        self, properties: DictConfig, data_folder: Path
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        ratings = self._read_format(
            data_folder.joinpath(f"ratings.{self._saving_format}")
        )
        if (
            data_folder.joinpath(f"meta_user.{self._saving_format}").exists() \
            and properties['meta_user']
        ):
            meta_user = self._read_format(data_folder.joinpath(f"meta_user.{self._saving_format}"))
        else:
            meta_user = None
        if (
            data_folder.joinpath(f"meta_item.{self._saving_format}").exists() \
            and properties['meta_item']
        ):
            meta_item = self._read_format(data_folder.joinpath(f"meta_item.{self._saving_format}"))
        else:
            meta_item = None

        return ratings, meta_user, meta_item

    def _save_data(
            self,
            data: Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]],
            properties: DictConfig,
            data_folder: Path
    ) -> None:
        self._to_format(
            data[0], data_folder.joinpath(f"ratings.{self._saving_format}"), index=False
        )
        if data[1] and properties['meta_user']:
            self._to_format(
                data[1], data_folder.joinpath(f"meta_user.{self._saving_format}"), index=False
            )
        if data[2] and properties['meta_item']:
            self._to_format(
                data[2], data_folder.joinpath(f"meta_item.{self._saving_format}"), index=False
            )

    def _new_data_prepare(self, properties: DictConfig, data_folder: Path) -> pd.DataFrame:
        raw_file_path = Path('/'.join((properties['data_src'], properties['ratings_file'])))
        # Downloading file
        if not raw_file_path.exists():
            logger.info("Raw file does not exist, downloading...")
            load_dataset(properties)

        raw_file = self._read_format(raw_file_path)

        user_id: str = properties['user_column']
        item_id: str = properties['item_column']
        rating_col: str = properties['rating_column']
        positive_th: float = properties['positive_threshold']
        min_item_ratings: int = properties['min_item_ratings']
        min_user_ratings: int = properties['min_user_ratings']

        # Drop repeated ratings
        logger.info('Dropping duplicates')
        raw_file.drop_duplicates([user_id, item_id], inplace=True)

        # Create implicit binary ratings (relevant / not relevant)
        logger.info('Dropping not relevant elements')
        raw_file['implicit_rating'] = (raw_file[rating_col] >= positive_th).astype(np.int32)
        raw_file.drop(index=raw_file[raw_file['implicit_rating'] == 0].index, inplace=True)

        # Item filtering
        logger.info('Item filtering')
        raw_file = self._filter(raw_file, item_id, rating_col, min_item_ratings)
        # User filtering
        logger.info('User filtering')
        raw_file = self._filter(raw_file, user_id, rating_col, min_user_ratings)

        self._useritem_mapping(raw_file, user_id, item_id, data_folder)

        return raw_file

    # Generate the new ordered representations of the users and items
    def _useritem_mapping(
        self, data: pd.DataFrame, user_id: str, item_id: str, data_folder: Path
    ) -> None:
        user2order, order2user = {}, {}
        item2order, order2item = {}, {}

        for i, id_ in enumerate(data[user_id].unique()):
            user2order[id_] = i
            order2user[i] = id_
        for i, id_ in enumerate(data[item_id].unique()):
            item2order[id_] = i
            order2item[i] = id_

        with data_folder.joinpath("item2order.pcl").open('wb') as file_i2o, \
                data_folder.joinpath("user2order.pcl").open('wb') as file_u2o, \
                data_folder.joinpath("order2item.pcl").open('wb') as file_o2i, \
                data_folder.joinpath("order2user.pcl").open('wb') as file_o2u:

            pickle.dump(user2order, file_u2o)
            pickle.dump(item2order, file_i2o)
            pickle.dump(order2user, file_o2u)
            pickle.dump(order2item, file_o2i)

        data[user_id] = data[user_id].map(user2order)
        data[item_id] = data[item_id].map(item2order)

    # Filter users or items by some threshold
    def _filter(
        self, data: pd.DataFrame, col: str, rating_col: str, min_ratings: int
    ) -> pd.DataFrame:
        data = data.join(
            data.groupby(col).count()[rating_col].rename('count'), on=col, how='left'
        )
        data.drop(
            index=data[data['count'] < min_ratings].index, axis=1, inplace=True
        )
        data.drop(columns='count', axis=1, inplace=True)
        return data

    # No realization yet
    def _new_meta_prepare(
        self,
        properties: DictConfig,
        data_folder: Path
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        return None, None
