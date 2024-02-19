"""Implicit ALS and BPR models embedding."""
import os
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Optional, Mapping, Any, List, Union

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

import optuna
from optuna.trial import Trial

from hydra.utils import instantiate
from omegaconf import DictConfig

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

from src.utils.logging import get_logger
from src.utils.metrics import normalized_discounted_cumulative_gain
from src.utils.processing import (
    get_optimization_lists,
    get_saving_path,
    pandas_to_aggregate,
)

logger = get_logger(name=__name__)

DTYPE = np.float32


class ImplicitBench:
    """
    Implicit Model Bench base class for model training, optimization, and evaluation.
    """

    def __init__(self, model, model_params: Optional[Mapping[str, Any]] = None) -> None:
        """
        Initialize the ImplicitAlsBench instance.

        Args:
            model (AlternatingLeastSquares or BayesianPersonalizedRanking):
            The ALS or BPR model instance.
            model_params (Mapping[str, Any]): Model parameters.
        """
        self.model = model
        self.model_params = model_params
        self.learning_time: Optional[float] = None
        self.predict_time: Optional[float] = None
        self.train_interactions: Optional[csr_matrix] = None

    @staticmethod
    def initialize_with_params(
        model_name: str, model_init_params: Mapping[Any, Any]
    ) -> "ImplicitBench":
        """
        Initialize ImplicitAlsBench with model parameters.

        Args:
            model_name (str): Name of the model ("als" or "bpr").
            model_init_params (Mapping[Any, Any]): Model initialization parameters.

        Returns:
            ImplicitAlsBench: Initialized ImplicitAlsBench instance.

        Raises:
            NotImplementedError: If the model name is not supported.
        """
        if model_name.lower() == "als":
            model = AlternatingLeastSquares(**model_init_params)
        elif model_name.lower() == "bpr":
            model = BayesianPersonalizedRanking(**model_init_params)
        else:
            raise NotImplementedError(f"model {model_name} isn't implemented")
        return ImplicitBench(model, model_init_params)

    @staticmethod
    def initialize_saved_model(path: str) -> Optional["ImplicitBench"]:
        """
        Initialize ImplicitAlsBench with a saved model.

        Args:
            path (str): Path to the saved model.

        Returns:
            Optional[ImplicitAlsBench]:
            Initialized ImplicitAlsBench instance or None if the model file doesn't exist.
        """
        # Create paths for model and params
        model_path = path.joinpath("model.pcl")
        params_path = path.joinpath("params.pcl")
        # Check if the directory exists
        if not path.exists():
            logger.error("Directory does not exist!")
            return None
        if not os.listdir(path):
            logger.error("Directory is empty!")
            return None
        # Check if the directory is empty
        if not model_path.exists() or not params_path.exists():
            logger.error(
                "Structure is not rights. Check if 'model.pcl'"
                "and 'params.pcl' are presented"
            )
            return None

        with model_path.open("rb") as model_file, params_path.open("rb") as params_file:
            return ImplicitBench(pickle.load(model_file), pickle.load(params_file))

    @staticmethod
    def initialize_with_optimization(
        model_name: str,
        optuna_params: DictConfig,
        interactions_train: Union[coo_matrix, csr_matrix],
        weights_train: Union[coo_matrix, csr_matrix],
        interactions_val: Union[coo_matrix, csr_matrix],
    ) -> "ImplicitBench":
        """
        Initialize ImplicitBench with hyperparameter optimization using Optuna.

        Args:
            model_name (str): Name of the model ("als" or "bpr").
            optuna_params (DictConfig): Optuna hyperparameter optimization parameters.
            interactions_train (coo_matrix): Sparse training interactions matrix.
            weights_train (coo_matrix): Sparse sample weight matrix for training interactions.
            interactions_val (coo_matrix): Sparse validation interactions matrix.

        Returns:
            ImplicitBench: Initialized ImplicitBench instance.
        """
        study = optuna.create_study(
            direction="maximize",
            sampler=instantiate(optuna_params["sampler"]),
            pruner=instantiate(optuna_params["pruner"]),
        )
        study.optimize(
            partial(
                ImplicitBench.objective,
                model_name=model_name,
                params_vary=optuna_params["hyperparameters_vary"],
                k_opt=optuna_params["k_optimization"],
                interactions_train=interactions_train,
                weights_train=weights_train,
                interactions_val=interactions_val,
            ),
            n_trials=optuna_params["n_trials"],
        )
        best_params = study.best_params.copy()
        if "const" in optuna_params["hyperparameters_vary"]:
            best_params.update(optuna_params["hyperparameters_vary"]["const"])

        logger.info("Best parameters are - %s", best_params)
        return ImplicitBench.initialize_with_params(model_name, best_params)

    # pylint: disable=too-many-arguments
    @staticmethod
    def objective(
        trial: Trial,
        model_name: str,
        params_vary: DictConfig,
        k_opt: List[int],
        interactions_train: Union[coo_matrix, csr_matrix],
        weights_train: Union[coo_matrix, csr_matrix],
        interactions_val: pd.DataFrame,
    ) -> float:
        """
        Objective function for hyperparameter optimization using Optuna.

        Args:
            trial (Trial): Optuna trial object.
            model_name (str): Name of the model ("als" or "bpr").
            params_vary (DictConfig): Hyperparameters to optimize.
            k_opt (int): Number of top-k items to evaluate.
            interactions_train (coo_matrix): Sparse training interactions matrix.
            weights_train (coo_matrix): Sparse sample weight matrix for training interactions.
            interactions_val (pd.DataFrame): Sparse validation interactions matrix.

        Returns:
            float: Mean metrics of the model evaluated on the validation set.
        """
        initial_model_parameters = get_optimization_lists(params_vary, trial)
        model = ImplicitBench.initialize_with_params(
            model_name, initial_model_parameters
        )
        model.fit(interactions_train, weights_train)
        val_userids = np.sort(interactions_val.userId.unique())
        top_100_items = model.recommend_k(
            userids=val_userids, k=100, train_interactions=interactions_train
        )
        metrics = []
        for k in k_opt:
            metrics.append(
                normalized_discounted_cumulative_gain(
                    top_100_items, pandas_to_aggregate(interactions_val), k
                )
            )
        return np.mean(metrics)

    def _process_weight(self, interactions, weights) -> coo_matrix:
        """
        Process the weights matrix.

        This method allows you to feed interactions and weights separately
        to models from implicit libraries. If weights is None, interactions used.
        If weights is not None, than shape and DTYPE is checked to match
        the interactions matrix.and weights returned.

        Args:
            interactions (coo_matrix): Sparse interactions matrix.
            weights (Optional[coo_matrix]): Sparse sample weight matrix.

        Returns:
            coo_matrix: Processed weight matrix.

        Raises:
            ValueError: If the shape and order of the weights matrix do not match
                the interactions matrix.
        """
        if not isinstance(interactions, coo_matrix):
            if not isinstance(interactions, csr_matrix):
                raise ValueError("interactions must be a COO matrix.")
            interactions = interactions.tocoo()

        if weights is not None:
            if not isinstance(weights, coo_matrix):
                if not isinstance(weights, csr_matrix):
                    raise ValueError("Sample_weight must be a COO matrix.")
                weights = weights.tocoo()

            if weights.shape != interactions.shape:
                raise ValueError(
                    "Sample weight and interactions matrices must be the same shape"
                )

            if not (
                np.array_equal(interactions.row, weights.row)
                and np.array_equal(interactions.col, weights.col)
            ):
                raise ValueError(
                    "Sample weight and interaction matrix "
                    "entries must be in the same order"
                )

            if weights.data.dtype != DTYPE:
                weight_data = weights.data.astype(DTYPE)
            else:
                weight_data = weights.data
        else:
            if np.array_equiv(interactions.data, 1.0):
                # Re-use interactions data if they are all ones
                weight_data = interactions.data
            else:
                # Otherwise allocate a new array of ones
                weight_data = np.ones_like(interactions.data, dtype=DTYPE)
        return coo_matrix((weight_data, (interactions.row, interactions.col)))

    def fit(
        self,
        interactions: Union[coo_matrix, csr_matrix],
        weights: Optional[Union[coo_matrix, csr_matrix]] = None,
        show_progress: bool = True,
        callback=None,
    ) -> None:
        """
        Fit the Implicit ALS or BPR model to the training data.

        Args:
            interactions (coo_matrix): Training interactions matrix (user-item interactions).
            weights (coo_matrix, optional): Weight matrix for training interactions.
            show_progress (bool, optional): Whether to show the progress during model training.
            callback (function, optional): Callback function to be executed during training.

        Returns:
            None
        """
        # We need this in the COO format.
        interactions = interactions.tocoo()

        if interactions.dtype != DTYPE:
            interactions.data = interactions.data.astype(DTYPE)

        weight_data = self._process_weight(interactions, weights)
        start_time = time.time()
        self.model.fit(
            user_items=weight_data.tocsr(),
            show_progress=show_progress,
            callback=callback,
        )
        self.train_interactions = interactions.tocsr()
        self.learning_time = time.time() - start_time

    def save_model(self, path: Path) -> None:
        """
        Save the Implicit ALS model to a file.

        Args:
            path (str): Path to the directory where the model file should be saved.

        Returns:
            None
        """
        model_dir = get_saving_path(path)

        with open(model_dir.joinpath("model.pcl"), "wb") as file:
            pickle.dump(self.model, file)
        with open(model_dir.joinpath("params.pcl"), "wb") as file:
            pickle.dump(self.model_params, file)

    def recommend_k(
        self,
        k: int,
        userids: np.ndarray,
        train_interactions: Union[coo_matrix, csr_matrix, None] = None,
        filter_already_liked_items: bool = True,
        filter_items=None,
        recalculate_user: bool = False,
    ) -> np.ndarray:
        """
        Recommend top k items for users.

        Args:
            k (int): Number of items to recommend.
            userids (np.ndarray): Array of user IDs to generate recommendations for.
            train_interactions (csr_matrix or coo_matrix, optional): User-item interaction matrix.
                Defaults to the instance's train_interactions attribute if None.
            filter_already_liked_items (bool, optional): When True, don't return items present in
                the training set that were rated by the specified user.
            filter_items (array_like, optional): List of extra item IDs to filter out from
                the output.
            recalculate_user (bool, optional): When True, recalculates
                factors for a batch of users.

        Returns:
            np.ndarray: 2-dimensional array with a row of item IDs for each user.
        """
        if k <= 0 or not isinstance(k, int):
            raise ValueError("'k' must be a positive integer.")

        if train_interactions is None:
            train_interactions = self.train_interactions

        if not isinstance(train_interactions, csr_matrix):
            train_interactions = train_interactions.tocsr()

        start_time = time.time()
        
        ids, _ = self.model.recommend(
            userid=userids,
            user_items=train_interactions[userids],
            N=k,
            filter_already_liked_items=filter_already_liked_items,
            filter_items=filter_items,
            recalculate_user=recalculate_user,
        )

        self.predict_time = time.time() - start_time

        return ids
