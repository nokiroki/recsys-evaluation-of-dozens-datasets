from typing import Union, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix, csr_matrix

from src.utils.logging import get_logger

logger = get_logger(name=__name__)


def pandas_to_sparse(
    dataset: pd.DataFrame,
    weighted: bool = False,
    shape: Optional[tuple[int, int]] = None,
    *,
    user_col: str = "userId",
    item_col: str = "itemId",
    rating_col: str = "rating",
    sparse_type: str = "csr"
) -> tuple[Union[coo_matrix, csr_matrix], Optional[Union[coo_matrix, csr_matrix]]]:
    if sparse_type == "csr":
        matrix = csr_matrix
    elif sparse_type == "coo":
        matrix = coo_matrix
    else:
        logger.warning("Unsupported type of sparse matrix. return csr as a default")
        matrix = csr_matrix

    interactions = matrix((
            np.where(dataset[rating_col] > 0, 1, 0),
            (dataset[user_col].to_numpy(), dataset[item_col].to_numpy()),
        ), shape=shape
    )

    if weighted:
        weights = interactions.copy()
        weights.data = dataset[rating_col].to_numpy()
        return interactions, weights
    else:
        return interactions, None


def pandas_to_recbole(
    dataset: pd.DataFrame,
    dataset_name: str,
    split_name: str = "train",
    *,
    data_dir: Path = Path("data/tmp/"),
    user_col: str = "userId",
    item_col: str = "itemId",
    rating_col: str = "rating",
    date_col: str = "datetime"
):
    dataset = dataset.drop(columns=["implicit_rating", date_col]).rename(
        columns={
            user_col: "user_id:token",
            item_col: "item_id:token",
            rating_col: "rating:float",
            date_col: "timestamp:float",
        }
    )

    data_dir = data_dir.joinpath(dataset_name)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(data_dir.joinpath(f"{dataset_name}.{split_name}.inter"), sep='\t', index=False)
