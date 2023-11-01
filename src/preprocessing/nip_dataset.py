"""Next Item Prediction Dataset Module"""
from typing import Optional, Tuple
import os
import pickle
import logging

from omegaconf import DictConfig

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from .base_dataset import BaseDataset

logger = get_logger(name=__name__, log_level=logging.DEBUG)


class NIPDataset(BaseDataset):
    """Next Item Prediction Dataset that suits only a NIP case."""

    def prepare(self, properties: DictConfig) -> None:
        name: str = properties["name"]
        if not os.path.exists("preproc_data"):
            logger.info("Creating data folder")
            os.mkdir("preproc_data")

        preproc_data_folder = os.path.join("preproc_data", name)
        if not os.path.exists(preproc_data_folder):
            logger.info("Dataset folder does not exist. Creating dataset.")
            os.mkdir(preproc_data_folder)

        params_file = os.path.join(preproc_data_folder, "nip_params.pcl")

        if os.path.exists(params_file):
            with open(params_file, "rb") as file:
                properties_saved = pickle.load(file)
        else:
            properties_saved = None
        logger.info("Parameters were changed. New preprocessing was started")
        with open(params_file, "wb") as file:
            pickle.dump(properties, file)
        nip_sequences = self._new_data_prepare(properties, preproc_data_folder)
        meta_user, meta_item = None, None

        if meta_user is None and meta_item is None:
            self._prepared_data = nip_sequences
        else:
            self._prepared_data = (nip_sequences, meta_user, meta_item)

    def _new_data_prepare(
        self, properties: DictConfig, data_folder: str
    ) -> pd.DataFrame:
        raw_file_path = os.path.join(properties["data_src"], properties["ratings_file"])
        if os.path.splitext(raw_file_path)[-1].lower() == ".parquet":
            raw_file = pd.read_parquet(raw_file_path)
        else:
            raw_file = pd.read_csv(raw_file_path)

        user_id: str = properties["user_column"]
        item_id: str = properties["item_column"]
        rating_col: str = properties["rating_column"]
        positive_th: float = properties["positive_threshold"]
        min_item_ratings: int = properties["min_item_ratings"]
        min_user_ratings: int = properties["min_user_ratings"]
        date: float = properties["date_column"]

        # Sort items by date
        logger.info("Sorting values")
        raw_file.sort_values(
            [user_id, date],
            ascending=[True, True],
            inplace=True,
        )

        # ToDO: drop it? ascending=False

        # Drop repeated events - only when type is unused
        logger.info("Dropping duplicates")
        raw_file.drop_duplicates([user_id, item_id], inplace=True)

        # Create implicit binary ratings (relevant / not relevant)
        logger.info("Dropping not relevant elements")
        raw_file["implicit_rating"] = (raw_file[rating_col] >= positive_th).astype(
            np.int32
        )
        raw_file.drop(
            index=raw_file[raw_file["implicit_rating"] == 0].index, inplace=True
        )

        # Item filtering
        logger.info("Item filtering")
        raw_file = self._filter(raw_file, item_id, rating_col, min_item_ratings)
        # User filtering
        logger.info("User filtering")
        raw_file = self._filter(raw_file, user_id, rating_col, min_user_ratings)

        self._useritem_mapping(raw_file, user_id, item_id, data_folder)

        return raw_file

    # Generate the new ordered representations of the users and items
    def _useritem_mapping(
        self, data: pd.DataFrame, user_id: str, item_id: str, data_folder: str
    ) -> None:
        user2order, order2user = {}, {}
        item2order, order2item = {}, {}

        for i, id_ in enumerate(data[user_id].unique()):
            user2order[id_] = i
            order2user[i] = id_
        for i, id_ in enumerate(data[item_id].unique()):
            item2order[id_] = i
            order2item[i] = id_

        with open(os.path.join(data_folder, "item2order.pcl"), "wb") as file_i2o, open(
            os.path.join(data_folder, "user2order.pcl"), "wb"
        ) as file_u2o, open(
            os.path.join(data_folder, "order2item.pcl"), "wb"
        ) as file_o2i, open(
            os.path.join(data_folder, "order2user.pcl"), "wb"
        ) as file_o2u:
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
            data.groupby(col).count()[rating_col].rename("count"), on=col, how="left"
        )
        data.drop(index=data[data["count"] < min_ratings].index, axis=1, inplace=True)
        data.drop(columns="count", axis=1, inplace=True)
        return data
