"""Implicit ALS and BPR models embedding."""
import os
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Optional, Mapping, Any, List, Union

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

import optuna
import implicit
from optuna.trial import Trial
from tqdm import tqdm

from hydra.utils import instantiate
from omegaconf import DictConfig

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

from src.utils.logging import get_logger
from src.utils.metrics import normalized_discounted_cumulative_gain
from src.utils.processing import get_optimization_lists, get_saving_path


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
        interactions_val: Union[coo_matrix, csr_matrix]
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
        interactions_val: Union[coo_matrix, csr_matrix],
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
            interactions_val (coo_matrix): Sparse validation interactions matrix.

        Returns:
            float: Mean rank of the model evaluated on the validation set.
        """
        initial_model_parameters = get_optimization_lists(params_vary, trial)
        model = ImplicitBench.initialize_with_params(
            model_name, initial_model_parameters
        )
        model.fit(interactions_train, weights_train)
        relevant_ranks = model.get_relevant_ranks(interactions_val, interactions_train)
        metrics = []
        for k in k_opt:
            metrics.append(normalized_discounted_cumulative_gain(relevant_ranks, k))
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
        interactions: coo_matrix,
        weights: Optional[coo_matrix] = None,
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
            callback=callback
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

    @staticmethod
    def _implicit_recommendation_to_rank(
        ids: np.ndarray, max_test_items: int, test_user_items: np.ndarray
    ) -> np.ndarray:
        """
        Convert implicit recommendations to rank.

        Args:
            ids (np.ndarray): Array representing the sorted relevant items.
            test_user_items (np.ndarray): Array representing the test user-item interactions.

        Returns:
            np.ndarray: Array of ranks for each user in the test set.
            Starts from 0 to max_test_items.
        """
        # Create an array of ranks using the argsort function
        ranks = np.argsort(ids, axis=1)

        # Make np.ndarrays of ranks, fill rows by -1 if len is less than max_test_items
        ranks[~test_user_items.astype(bool)] = -1
        ranks = np.sort(ranks, axis=1)[:, ::-1][:, :max_test_items]

        return ranks

    def get_relevant_ranks(
        self,
        test_interactions: coo_matrix,
        train_interactions: Optional[coo_matrix] = None
    ) -> np.ndarray:
        """
        Get relevant ranks for the test interactions.

        Args:
            train_interactions (coo_matrix): Training interactions matrix (user-item interactions).
            test_interactions (coo_matrix): Test interactions matrix (user-item interactions).

        Returns:
            np.ndarray: Array of relevant ranks for each user in the test set.
        """
        # Converting to CSR
        if train_interactions is not None:
            train_interactions = train_interactions.tocsr()
        else:
            train_interactions = self.train_interactions.tocsr()
        test_interactions = test_interactions.tocsr()

        # Extract number of users and items
        userids, items = train_interactions.shape

        # Get the maximum number of non-zero test set
        max_items = test_interactions.getnnz(axis=1).max()

        # Use batches
        batch_size = 1024
        start_idx = 0
        ranks = np.empty((0, max_items), dtype="int32")

        # get an array of userids that have at least one item in the test set
        to_generate = np.arange(userids, dtype="int32")
        to_generate = to_generate[np.ediff1d(test_interactions.indptr) > 0]

        start_time = time.time()

        with tqdm(
            (len(to_generate) + batch_size - 1) // batch_size,
            desc="Generating recommendations", unit="batch"
        ) as pbar:
            while start_idx < len(to_generate):
                batch = to_generate[start_idx : start_idx + batch_size]

                # make recommendations on cpu due to large datasets
                if isinstance(
                    self.model,
                    (implicit.gpu.als.AlternatingLeastSquares,
                    implicit.gpu.bpr.BayesianPersonalizedRanking)
                    ):
                    self.model = self.model.to_cpu()
                else:
                    pass

                ids_batch, _ = self.model.recommend(batch, train_interactions[batch], N=items)
                start_idx += batch_size

                ranks_batch = self._implicit_recommendation_to_rank(
                    ids_batch, max_items, test_interactions[batch].toarray()
                )
                # Concatenate the new array to the result_array
                ranks = np.concatenate((ranks, ranks_batch), axis=0)

                pbar.update(1)

        self.predict_time = time.time() - start_time

        return ranks

    def recommend_k(
        self,
        k: int,
        filter_already_liked_items: bool = True,
        filter_items=None,
        recalculate_user: bool = False,
    ) -> np.ndarray:
        """
        Recommend top k items for users.

        Args:
            k (int): The number of results to return.
            train_interactions (csr_matrix or coo_matrix):
                Sparse matrix of shape (users, number_items)
                representing the user-item interactions for training.
            filter_already_liked_items (bool, optional): When True, don't return items present in
                the training set that were rated by the specified user.
            filter_items (array_like, optional): List of extra item IDs to filter out from
                the output.
            recalculate_user (bool, optional): When True, recalculates
                factors for a batch of users.

        Returns:
            np.ndarray: 2-dimensional array with a row of item IDs for each user.
        """
        train_interactions = self.train_interactions[np.ediff1d(self.train_interactions.indptr) > 0]
        userids = np.arange(train_interactions.shape[0])
        ids, _ = self.model.recommend(
            userid=userids,
            user_items=train_interactions.tocsr(),
            N=k,
            filter_already_liked_items=filter_already_liked_items,
            filter_items=filter_items,
            recalculate_user=recalculate_user,
        )
        return ids
