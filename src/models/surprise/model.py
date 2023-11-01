"""SupriceBench module"""
from copy import deepcopy
from functools import partial
import os
from pathlib import Path
import pickle
import time
from typing import Optional, Any, Mapping, List, Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import optuna
from optuna.trial import Trial
import pandas as pd
from scipy.sparse import coo_matrix
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from tqdm.auto import tqdm

from src.utils.logging import get_logger
from src.utils.metrics import normalized_discounted_cumulative_gain
from src.utils.processing import get_optimization_lists, get_saving_path

logger = get_logger(name=__name__)


class SurpriseBench:
    """Surprise Model Bench base class for model training, optimization, and evaluation."""

    def __init__(
        self,
        model: KNNBasic,
        model_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize the SurpriseBench instance.

        Args:
            model (KNNBasic):
            The KNNBasic model instance.
            model_params (Mapping[str, Any]): Model parameters.
        """
        self.model = model
        self.model_params = model_params
        self.learning_time: Optional[float] = None
        self.predict_time: Optional[float] = None
        self.max_rate = None

    @staticmethod
    def initialize_with_params(
        model_type: str, model_init_params: Mapping[Any, Any]
    ) -> "SurpriseBench":
        """
        Initialize SurpriseBench with model parameters.

        Args:
            model_type (str): Name of the model ("KNNBasic").
            model_init_params (Mapping[Any, Any]): Model initialization parameters.

        Returns:
            SurpriseBench: Initialized SurpriseBench instance.

        Raises:
            NotImplementedError: If the model name is not supported.
        """
        model = SurpriseBench.get_model(
            model_type,
            model_init_params,
        )

        if model is None:
            raise NotImplementedError(f"model {model_type} isn't implemented")

        return SurpriseBench(model, model_init_params)

    def build_reader_data(
        self,
        data: coo_matrix,
        split_type="train",
    ):
        """
        Transform coo_matrix input to surprise input dataset.

        Args:
            data (coo_matrix): Sparse matrix input data.
            split_type (str): split type (train or test) for return type.

        Returns:
            pd.DataFrame: input data as pd.DataFrame.
            Union[Dataset, List]: input data for surprise model.

        """
        tmp_data = data.tocoo()
        new_data = pd.DataFrame(
            np.column_stack((tmp_data.row, tmp_data.col, tmp_data.data)),
            columns=[
                self.model_params["user_column"],
                self.model_params["item_column"],
                self.model_params["rating_column"],
            ],
        )
        if self.max_rate is None:
            self.max_rate = max(new_data[self.model_params["item_column"]])

        reader = Reader(rating_scale=(1, min(self.max_rate, 5)))
        data = Dataset.load_from_df(
            new_data[
                [
                    self.model_params["user_column"],
                    self.model_params["item_column"],
                    self.model_params["rating_column"],
                ]
            ],
            reader,
        )

        if split_type == "train":
            output, _ = train_test_split(data, test_size=None, train_size=1.0)
        else:
            _, output = train_test_split(data, test_size=1.0)
        return new_data, output

    # pylint: disable=too-many-arguments
    @staticmethod
    def objective(
        trial: Trial,
        params_vary: DictConfig,
        model_params: DictConfig,
        k_opt: List[int],
        interactions_train: coo_matrix,
        interactions_val: coo_matrix,
    ) -> float:
        """
        Objective function for hyperparameter optimization using Optuna.

        Args:
            trial (Trial): Optuna trial object.
            params_vary (DictConfig): Hyperparameters to optimize.
            model_params (DictConfig): General hyperparameters of model.
            k_opt (int): Number of top-k items to evaluate.
            interactions_train (coo_matrix): Sparse training interactions matrix.
            interactions_val (coo_matrix): Sparse validation interactions matrix.

        Returns:
            float: Mean rank of the model evaluated on the validation set.
        """
        initial_model_parameters = get_optimization_lists(params_vary, trial)
        initial_model_parameters["sim_options"] = {}
        if "name" in initial_model_parameters:
            initial_model_parameters["sim_options"]["name"] = initial_model_parameters[
                "name"
            ]

        if "min_support" in initial_model_parameters:
            initial_model_parameters["sim_options"][
                "min_support"
            ] = initial_model_parameters["min_support"]
        initial_model_parameters["sim_options"] = dict(model_params["model"]['sim_options']) | dict(initial_model_parameters["sim_options"])
        model = SurpriseBench.initialize_with_params(
            model_params["name"], dict(model_params["model"]) | initial_model_parameters
        )
        model.fit(interactions_train)
        relevant_ranks = model.get_relevant_ranks(
            100,
            interactions_val,
        )
        metrics = []
        if isinstance(k_opt, int):
            return normalized_discounted_cumulative_gain(relevant_ranks, k_opt)

        for k in k_opt:
            metrics.append(normalized_discounted_cumulative_gain(relevant_ranks, k))
        return np.mean(metrics)

    @staticmethod
    def initialize_saved_model(path: Path) -> Optional["SurpriseBench"]:
        """Builder for the saved model.

        Args:
            path (Path): Path to the model.

        Returns:
            SurpriseBench: Loaded model.
        """
        if not path.exists():
            logger.error("Directory does not exist")
            return None
        if not os.listdir(path):
            logger.error("Directory is empty")
            return None
        model_params_path = path.joinpath("params.pcl")
        if model_params_path.exists():
            with open(model_params_path, "rb") as file:
                model_params = pickle.load(file)
        else:
            logger.warning("Attribute 'best params' not found")
            model_params = None

        try:
            model_path = path.joinpath("model.pcl")
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            return SurpriseBench(model, model_params)
        except OSError as error:
            logger.error(error.args[0])
            return None

    @staticmethod
    def initialize_with_optimization(
        optuna_params: DictConfig,
        model_params: DictConfig,
        interactions_train: coo_matrix,
        interactions_val: coo_matrix,
    ) -> "SurpriseBench":
        """Builder for the model with a optimization.

        Args:
            optuna_params (DictConfig): Mapping with the optuna additional parameters,
                including the samper and the pruner. Defined in the config.
            model_params (DictConfig): Default model parameters.
            interactions_train (coo_matrix): Sparse matrix with the user-item interactions.
            interactions_val (coo_matrix): Sparse matrix with user-item interactions for validation.

        Returns:
            SurpriseBench: Model with the optimized parameters.
        """
        study = optuna.create_study(
            direction="maximize",
            sampler=instantiate(optuna_params["sampler"]),
            pruner=instantiate(optuna_params["pruner"]),
        )
        study.optimize(
            partial(
                SurpriseBench.objective,
                params_vary=optuna_params["hyperparameters_vary"],
                k_opt=optuna_params["k_optimization"],
                interactions_train=interactions_train,
                model_params=model_params,
                interactions_val=interactions_val,
            ),
            n_trials=optuna_params["n_trials"],
        )
        best_params = study.best_params.copy()
        if "const" in optuna_params["hyperparameters_vary"]:
            best_params.update(optuna_params["hyperparameters_vary"]["const"])

        logger.info("Best parameters are - %s", best_params)
        return SurpriseBench.initialize_with_params(
            model_params["name"], best_params | dict(model_params["model"])
        )

    @staticmethod
    def get_model(model_type: str, model_init_params: Mapping[Any, Any]) -> KNNBasic:
        """Get model depending on "model_type" parameter. Available options:

            user_knn
        Args:
            model_type (str): Name of the model to initialize.
            model_init_params (Mapping[Any, Any]): Any parameters for the model initialization.

        Raises:
            ValueError: Error would be raised if the model with "model_type" name
                was not implemented yet.

        Returns:
            KNNBasic: Model instance.
        """
        if model_type == "user_knn":
            model = KNNBasic(**model_init_params)
        else:
            raise ValueError(f"model type {model_type} was not implemented")

        return model

    def fit(self, interactions: coo_matrix) -> None:
        """
        Fit the Surprise KNNBasic model to the training data.

        Args:
            interactions (coo_matrix): Training interactions matrix (user-item interactions).

        Returns:
            None
        """
        _, train = self.build_reader_data(interactions)

        start_time = time.time()
        self.model.fit(train)
        self.learning_time = time.time() - start_time

    def save_model(self, path: Path) -> None:
        """
        Save the Surprise KNNBasic model to a file.

        Args:
            path (str): Path to the directory where the model file should be saved.

        Returns:
            None
        """
        try:
            model_dir = get_saving_path(path)

            with open(model_dir.joinpath("model.pcl"), "wb") as model_file, open(
                model_dir.joinpath("params.pcl"), "wb"
            ) as params_file:
                pickle.dump(self.model, model_file)
                pickle.dump(self.model_params, params_file)
        except Exception as error:
            logger.error(error)
            logger.error(error.args[0])
            logger.error("Model has not been saved!")
            raise

    def get_relevant_ranks(
        self,
        k: int,
        test_interactions: coo_matrix,
        recommended_items: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get relevant ranks for the test interactions.

        Args:
            train_interactions (coo_matrix): Training interactions matrix (user-item interactions).
            k (int): The number of results to return.
            test_interactions (coo_matrix): Test interactions matrix (user-item interactions).
            recommended_items (Optional[np.ndarray])

        Returns:
            np.ndarray: Array of relevant ranks for each user in the test set.
        """
        if recommended_items is None:
            recs = self.recommend_k(k, test_interactions)
        else:
            recs = recommended_items.copy()

        test_df, _ = self.build_reader_data(
            test_interactions,
            split_type="test",
        )
        test_df = test_df.rename(
            columns={
                self.model_params["user_column"]: "user_idx",
                self.model_params["item_column"]: "item_idx",
                self.model_params["rating_column"]: "relevance",
            }
        )
        filler = test_df["item_idx"].max()

        for i, user in tqdm(
            enumerate(test_df["user_idx"].unique()),
            desc="Calculating ranks",
            total=test_df["user_idx"].nunique(),
        ):
            relevant_item = set(test_df[test_df["user_idx"] == user]["item_idx"].values)
            for j in range(recs.shape[1]):
                recs[i, j] = j if recs[i, j] in relevant_item else filler
            recs[i] = sorted(recs[i])
            recs[i, len(relevant_item) :] = -1

        return recs

    def recommend_k(
        self,
        k: int,
        test_interactions: Optional[coo_matrix] = None,
    ) -> np.ndarray:
        """
        Recommend top k items for users.

        Args:
            train_interactions (csr_matrix or coo_matrix):
                Sparse matrix of shape (users, number_items)
                representing the user-item interactions for training.
                Only used as general pipeline requirement.
            k (int): The number of results to return.
            test_interactions (csr_matrix or coo_matrix):
                Sparse matrix of shape (users, number_items)
                representing the user-item interactions for training.

        Returns:
            np.ndarray: 2-dimensional array with a row of item IDs for each user.
        """
        start_time = time.time()
        test_df, test_input = self.build_reader_data(
            test_interactions,
            split_type="test",
        )
        test_df = test_df.rename(
            columns={
                self.model_params["user_column"]: "user_idx",
                self.model_params["item_column"]: "item_idx",
                self.model_params["rating_column"]: "relevance",
            }
        )
        recs = pd.DataFrame(self.model.test(test_input)).rename(
            columns={
                "uid": "user_idx",
                "iid": "item_idx",
                "r_ui": "relevance",
            }
        )
        self.predict_time = time.time() - start_time
        logger.info("Finished predicting most similiar items")
        # Sort users
        recs.sort_values(["user_idx", "relevance"], inplace=True)

        tmp_recs = (
            recs.groupby("user_idx")
            .agg({"relevance": lambda l: np.unique(l.tolist()).tolist()[:k]})
            .reset_index()
        )

        for item, user in enumerate(np.unique(tmp_recs["user_idx"]), 0):
            new_relevance = tmp_recs.loc[tmp_recs["user_idx"] == user][
                "relevance"
            ].values[0]
            if item == 0:
                output_df = deepcopy(
                    recs.loc[
                        (recs["user_idx"] == user)
                        & (recs["relevance"].isin(new_relevance))
                    ]
                )
            else:
                output_df = pd.concat(
                    [
                        output_df,
                        deepcopy(
                            recs.loc[
                                (recs["user_idx"] == user)
                                & (recs["relevance"].isin(new_relevance))
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

        recs = (
            test_df.sort_values("user_idx")
            .drop_duplicates(subset=["user_idx"])
            .drop("item_idx", axis=1)
            .merge(recs, on="user_idx", how="left")["item_idx"]
        )

        last_recs = output_df.groupby("user_idx").agg(
            {"item_idx": lambda l: l.tolist()[:k]}
        )

        filler = recs.agg({"item_idx": "max"})[0] + 1

        recommend = np.ones((recs.shape[0], k)) * filler
        for i, rec in tqdm(
            enumerate(last_recs.values),
            desc="Filling items array",
            total=len(last_recs.values),
        ):
            if not isinstance(rec, float):
                recommend[i][: len(*rec)] = np.array(*rec)

        return recommend

    def get_items_and_ranks(
        self, k: int, test_interactions: coo_matrix
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrap two methods to get ranks and items.

        Args:
            train_interactions (csr_matrix or coo_matrix):
                Sparse matrix of shape (users, number_items)
                representing the user-item interactions for training.
            k (int): The number of results to return.
            test_interactions (csr_matrix or coo_matrix):
                Sparse matrix of shape (users, number_items)
                representing the user-item interactions for training.

        Returns:
            np.ndarray: list of k recommend items.
            np.ndarray: list of recommend items ranks.
        """
        items = self.recommend_k(k, test_interactions)
        ranks = self.get_relevant_ranks(k, test_interactions, items)
        return items, ranks
