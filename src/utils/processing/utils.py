"""Universal data preporation module"""
from typing import Union, Optional, Any, Mapping, Tuple

import os
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

# from replay.data_preparator import DataPreparator
# from replay.splitters import DateSplitter, RandomSplitter
# from pyspark.sql import DataFrame

from optuna.trial import Trial
from omegaconf import DictConfig

from src.utils.logging import get_logger

logger = get_logger(name=__name__)


def create_sparse_matrix(
    data: pd.DataFrame,
    rating_col: str,
    user_col: str,
    item_col: str,
    weighted: bool = False,
    data_shape: tuple = None,
    sparse_type: str = "csr",
) -> Union[csr_matrix, Tuple[csr_matrix, csr_matrix]]:
    """
    Create a sparse matrix from the input DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        rating_col (str): The name of the column containing the ratings.
        user_col (str): The name of the column containing the user IDs.
        item_col (str): The name of the column containing the item IDs.
        weighted (bool, optional): Whether to create a weighted matrix based
        on ratings (default=False).
        data_shape (tuple): Desired shape of sparse matrix.
        sparse_type (str): type of the sparse matrix

    Returns:
        Union[csr_matrix, tuple[csr_matrix, csr_matrix]]: The sparse matrix or tuple of sparse
        matrix and weights.
    """
    if sparse_type == "csr":
        matrix = csr_matrix
    elif sparse_type == "coo":
        matrix = coo_matrix
    else:
        logger.warning("Unsupported type of sparse matrix. return csr as a default")
        matrix = csr_matrix

    interactions = matrix(
        (
            np.where(data[rating_col] > 0, 1, 0),
            (data[user_col].to_numpy(), data[item_col].to_numpy()),
        ),
        shape=data_shape,
    )

    if weighted:
        weights = interactions.copy()
        weights.data = data[rating_col].to_numpy()
        return interactions, weights
    else:
        return interactions, None

def train_test_split(
    dataset: pd.DataFrame,
    test_size: float,
    user_col: str = "userId",
    item_col: str = "itemId",
    date_col: str = "timestamp",
    *,
    splitting_type: str = "temporal",
    random_state: Optional[int] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if splitting_type == "temporal":
        dataset.sort_values(date_col, inplace=True)
        date_at_test_percentile = dataset[date_col].quantile(1 - test_size)

        train_set = dataset[dataset[date_col] <= date_at_test_percentile]
        test_set = dataset[dataset[date_col] > date_at_test_percentile]

    elif splitting_type == "random":
        random_state = np.random.RandomState(random_state)
        random_index = random_state.random_sample(len(dataset))
        train_index = random_index < 1 - test_size
        test_index = random_index >= 1 - test_size

        train_set = dataset[train_index]
        test_set = dataset[test_index]

    else:
        raise NotImplementedError(
            f"Splitting type {splitting_type} isn't implemented for this example"
        )

    # Filter out rows in test set that have users, items not present in train set
    train_users = set(train_set[user_col])
    train_items = set(train_set[item_col])
    test_set = test_set[
        test_set[user_col].isin(train_users) & test_set[item_col].isin(train_items)
    ]

    return train_set, test_set

def data_split(
    dataset: pd.DataFrame,
    data_conf: DictConfig,
    random_state: Optional[int] = None,
    return_format: str = "sparse",
    sparse_type: str = "csr",
    is_implicit: bool = False
) -> Union[
    Tuple[csr_matrix, csr_matrix, csr_matrix],
    Tuple[
        Tuple[csr_matrix, csr_matrix],
        Tuple[csr_matrix, csr_matrix],
        Tuple[csr_matrix, csr_matrix],
    ],
]:
    """
    Split the DataFrame into train, validation, and test sets based on the specified splitting type.
    Create sparse matrices for the train, validation, and test sets.

    Parameters:
        dataset (pd.DataFrame): The input DataFrame with 'date', 'rating', 'movieId_new',
            and 'userId_new' columns.
        data_conf (DictConfig): Configuration dictionary.
        random_state (int, optional): Random seed for reproducibility (default=None).
        return_format (str): format of return data: 'sparse' if csr matrixes,
            'recbole' if recbole dataloader, pandas as pd.DataFrame
        sparse_type (str): type of the sparse matrix

    Returns:
        if return_format is sparse:
            Union[tuple[csr_matrix, csr_matrix, csr_matrix], tuple[tuple[csr_matrix, csr_matrix],
            tuple[csr_matrix, csr_matrix], tuple[csr_matrix, csr_matrix]]]: Tuple of sparse
            matrices for train, validation, and test sets.
        if return_format is pandas:
            Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple of pandas DataFrames for
            train, validation, and test sets.
        if return_format is recbole:
            Union[AbstractDataLoader, AbstractDataLoader, AbstractDataLoader]:
            Tuple of recbole dataloaders DataFrames for train, validation, and test sets.

    Raises:
        ValueError: If the total fraction of train_size, val_size, and test_size is not equal to 1.
        NotImplementedError: If the specified splitting type is not implemented.

    Example:
        # Assuming you have a DataFrame called 'filtered_ratings'
        train_set, val_set, test_set = data_split(filtered_ratings, data_conf)
    """

    user_col = data_conf["user_column"]
    item_col = data_conf["item_column"]
    rating_col = data_conf["rating_column"]
    date_col = data_conf["date_column"]
    weighted = data_conf["weighted"]

    splitting_conf = data_conf["splitting"]
    train_size = splitting_conf["train_size"]
    val_size = splitting_conf["val_size"]
    test_size = splitting_conf["test_size"]
    splitting_type = splitting_conf["strategy"]

    if train_size + val_size + test_size != 1:
        raise ValueError("Expected total fraction equal to 1")

    # Create a sparse matrix using coo_matrix where the row is the userId,
    # the column is the itemId, and the value is the rating.

    if splitting_type == "temporal":
        dataset.sort_values(date_col, inplace=True)
        date_at_val_percentile = dataset[date_col].quantile(1 - val_size - test_size)
        date_at_test_percentile = dataset[date_col].quantile(1 - test_size)

        train_set = dataset[dataset[date_col] <= date_at_val_percentile]
        val_set = dataset[
            (dataset[date_col] > date_at_val_percentile)
            & (dataset[date_col] <= date_at_test_percentile)
        ]
        test_set = dataset[dataset[date_col] > date_at_test_percentile]

    elif splitting_type == "random":
        random_state = np.random.RandomState(random_state)
        random_index = random_state.random_sample(len(dataset))
        train_index = random_index < train_size
        val_index = (train_size <= random_index) & (
            random_index < train_size + val_size
        )
        test_index = random_index >= train_size + val_size

        train_set = dataset[train_index]
        val_set = dataset[val_index]
        test_set = dataset[test_index]

    else:
        raise NotImplementedError(
            f"Splitting type {splitting_type} isn't implemented for this example"
        )

    # Filter out rows in test set that have users, items not present in train and val sets
    train_users = set(train_set[user_col])
    train_items = set(train_set[item_col])
    val_set = val_set[
        val_set[user_col].isin(train_users) & val_set[item_col].isin(train_items)
    ]
    test_set = test_set[
        test_set[user_col].isin(train_users) & test_set[item_col].isin(train_items)
    ]

    if return_format == "pandas":
        return train_set, val_set, test_set

    elif return_format == "recbole":
        train_set = train_set.drop(columns=["implicit_rating", date_col]).rename(
            columns={
                user_col: "user_id:token",
                item_col: "item_id:token",
                rating_col: "rating:float",
                date_col: "timestamp:float",
            }
        )

        val_set = val_set.drop(columns=["implicit_rating", date_col]).rename(
            columns={
                user_col: "user_id:token",
                item_col: "item_id:token",
                rating_col: "rating:float",
                date_col: "timestamp:float",
            }
        )

        test_set = test_set.drop(columns=["implicit_rating", date_col]).rename(
            columns={
                user_col: "user_id:token",
                item_col: "item_id:token",
                rating_col: "rating:float",
                date_col: "timestamp:float",
            }
        )

        # create temp folder
        if not os.path.isdir(os.path.join("data", "tmp", data_conf["name"])):
            os.makedirs(os.path.join("data", "tmp", data_conf["name"]))

        # save temp files
        train_set.to_csv(
            os.path.join(
                "data", "tmp", data_conf["name"], data_conf["name"] + ".train.inter"
            ),
            sep="\t",
            index=False,
        )
        val_set.to_csv(
            os.path.join(
                "data", "tmp", data_conf["name"], data_conf["name"] + ".val.inter"
            ),
            sep="\t",
            index=False,
        )
        trainval_set = pd.concat([train_set, val_set], ignore_index=True)
        trainval_set.to_csv(
            os.path.join(
                "data", "tmp", data_conf["name"], data_conf["name"] + ".trainval.inter"
            ),
            sep="\t",
            index=False,
        )
        test_set.to_csv(
            os.path.join(
                "data", "tmp", data_conf["name"], data_conf["name"] + ".test.inter"
            ),
            sep="\t",
            index=False,
        )

    elif return_format == "sparse":
        data_matrix = coo_matrix(
            (
                dataset["implicit_rating"] if is_implicit else dataset[rating_col].to_numpy(),
                (dataset[user_col].to_numpy(), dataset[item_col].to_numpy()),
            )
        )
        data_shape = data_matrix.shape

        interactions_train, weights_train = create_sparse_matrix(
            train_set, rating_col, user_col, item_col, weighted, data_shape, sparse_type
        )
        interactions_val, weights_val = create_sparse_matrix(
            val_set, rating_col, user_col, item_col, weighted, data_shape, sparse_type
        )
        interactions_test, weights_test = create_sparse_matrix(
            test_set, rating_col, user_col, item_col, weighted, data_shape, sparse_type
        )
        return (
            (interactions_train, weights_train),
            (interactions_val, weights_val),
            (interactions_test, weights_test),
        )
    else:
        raise ValueError("return format should be pandas, sparse or recbole")


# def replay_data_split(
#     dataset: pd.DataFrame, data_conf: DictConfig, random_state: Optional[int] = None
# ) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
#     """Special function to realize the replay way of splitting.

#     Args:
#         dataset (pd.DataFrame): Dataset to split.
#         data_conf (DictConfig): OmegaConf config with the splitting parameters.
#         random_state (Optional[int], optional): Fixed random state

#     Raises:
#         ValueError: Raised if train_size + val_size + test_size is not equal to one.
#         NotImplementedError: Not implemented splitting size.

#     Returns:
#         Tuple[DataFrame, DataFrame, DataFrame, DataFrame]: tuple with [
#             train_val, train, val, test
#         ] datasets. Train_val here is required as concatenated train and val may not be the same.
#     """
#     user_col = data_conf["user_column"]
#     item_col = data_conf["item_column"]
#     rating_col = data_conf["rating_column"]
#     date_col = data_conf["date_column"]

#     splitting_conf = data_conf["splitting"]
#     train_size = splitting_conf["train_size"]
#     val_size = splitting_conf["val_size"]
#     test_size = splitting_conf["test_size"]
#     splitting_type = splitting_conf["strategy"]

#     if train_size + val_size + test_size != 1:
#         raise ValueError("Expected total fraction equal to 1")

#     # Hard code for nanoseconds to seconds transform
#     if dataset.iloc[0][date_col] > 1e11:
#         dataset[date_col] //= 1e9

#     dataset = DataPreparator().transform(
#         {
#             "user_id": user_col,
#             "item_id": item_col,
#             "relevance": rating_col,
#             "timestamp": date_col,
#         },
#         dataset,
#     )
#     dataset = dataset.withColumnRenamed("user_id", "user_idx")
#     dataset = dataset.withColumnRenamed("item_id", "item_idx")

#     params = {"drop_cold_items": True, "drop_cold_users": True}

#     if splitting_type == "random":
#         params["seed"] = random_state
#         train_test_splitter = RandomSplitter(test_size=test_size, **params)
#         train_val_splitter = RandomSplitter(test_size=val_size, **params)
#     elif splitting_type == "temporal":
#         train_test_splitter = DateSplitter(test_start=test_size, **params)
#         train_val_splitter = DateSplitter(test_start=val_size, **params)
#     else:
#         raise NotImplementedError(
#             f"Splitting type {splitting_type} isn't implemented for this example"
#         )

#     train_val, test = train_test_splitter.split(dataset)
#     train, val = train_val_splitter.split(train_val)

#     return train_val, train, val, test


def get_optimization_lists(params_vary: DictConfig, trial: Trial) -> Mapping[str, Any]:
    """Function for optuna optimization range lists generation.

    Args:
        params_vary (DictConfig): Dictionaries with the parameters to vary. Can be:
            float
            int
            choice
            const (parameters that are transfered to the original model as they are).
        trial (Trial): Optuna trial.

    Returns:
        Mapping[str, Any]: Initial model parameters.
    """
    initial_model_parameters = {}
    if "int" in params_vary:
        for param_int in params_vary["int"]:
            initial_model_parameters[param_int["name"]] = trial.suggest_int(**param_int)
    if "float" in params_vary:
        for param_float in params_vary["float"]:
            initial_model_parameters[param_float["name"]] = trial.suggest_float(
                **param_float
            )
    if "choice" in params_vary:
        for param_choice in params_vary["choice"]:
            initial_model_parameters[param_choice["name"]] = trial.suggest_categorical(
                **param_choice
            )
    if "const" in params_vary:
        initial_model_parameters.update(**params_vary["const"])

    return initial_model_parameters


def get_saving_path(path: Path) -> Path:
    """Save the model. If previously saved model exists,
    creating a new directory with higher ordering number

    Args:
        path (Path): Original path to the model.

    Returns:
        Path: New path with the name.
    """
    # Check if the directory exists
    if not path.exists():
        logger.warning("Directory does not exist. Creating...")
        path.mkdir(parents=True)
    # Check max model version and create new dir
    i = 0
    for model_path in path.glob("model_*"):
        i = max(int(model_path.stem.split("_")[1]) + 1, i)
    model_dir = path.joinpath(f"model_{i}")
    model_dir.mkdir()
    return model_dir


# pylint: disable=too-many-arguments
def save_results(
    metrics: Tuple[pd.DataFrame, str],
    top_items: Tuple[np.ndarray, str],
    result_folder: str,
    model_name: str,
    dataset_name: str,
    library_name: Optional[str] = None,
) -> None:
    """Save results with the particular parameters.

    Args:
        metrics (Tuple[pd.DataFrame, str]): Metrics table with the name.
        top_items (Tuple[np.ndarray, str]): Predicted top items with the name.
        ranks (Tuple[np.ndarray, str]): Predicted relevant ranks with the name.
        result_folder (str): Result folder to save.
        model_name (str): Name of the model.
        dataset_name (str): Name of the dataset.
        library_name (Optional[str], optional): Optional library name. Defaults to None.
    """
    logger.info("Saving models' results, top predicted items")

    path_to_model = [library_name] if library_name is not None else []
    path_to_model.extend([model_name, dataset_name])
    path_to_model = Path("/".join(path_to_model))

    result_metric_folder = Path("/".join((result_folder, "metrics"))).joinpath(
        path_to_model
    )
    result_items = Path("/".join((result_folder, "predicts"))).joinpath(
        path_to_model
    )

    if not result_metric_folder.exists():
        result_metric_folder.mkdir(parents=True)
    if not result_items.exists():
        result_items.mkdir(parents=True)

    metrics[0].to_csv(result_metric_folder.joinpath(f"{metrics[1]}.csv"))

    np.save(result_items.joinpath(f"{top_items[1]}.npy"), top_items[0])
