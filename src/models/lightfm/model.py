"""LightFM benchmark module"""
from typing import Optional, Mapping, Any
from pathlib import Path
import multiprocessing as mp
import pickle
import time
import os
from functools import partial

from lightfm import LightFM

import numpy as np
import pandas as pd

import optuna
from optuna.trial import Trial

from tqdm.auto import tqdm

from hydra.utils import instantiate
from omegaconf import DictConfig

from scipy.sparse import coo_matrix

from src.utils.logging import get_logger
from src.utils.metrics import normalized_discounted_cumulative_gain
from src.utils.processing import get_optimization_lists, get_saving_path

logger = get_logger(name=__name__)


class LightFMBench:
    """Main class for the LightFM model benchmarking.
    LightFM - Hybrid filtering model for the recommendation systems.
    For the more detailed documenation please refer to the 
    https://making.lyst.com/lightfm/docs/home.html
    """

    def __init__(self, model: LightFM, model_params: Optional[Mapping[str, Any]] = None) -> None:
        self.model = model
        self.model_params = model_params
        self.learning_time: Optional[float] = None
        self.predict_time: Optional[float] = None

    @staticmethod
    def initialize_with_params(model_init_params: Mapping[str, Any]) -> "LightFMBench":
        """Builder for the instantiating a model from the predefined parameters.

        Args:
            model_init_params (Mapping[Any, Any]): Any model parameters for the model.
        Returns:
            LightFMBench: LightFMBench object initialized from the parameters.
        """
        model = LightFM(**model_init_params)

        return LightFMBench(model, model_init_params)

    @staticmethod
    def initialize_saved_model(path: Path) -> Optional["LightFMBench"]:
        """Builder for the saved model.

        Args:
            path (Path): Path to the model.

        Returns:
            LightFMBench: Loaded model.
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

        with model_path.open('rb') as model_file, params_path.open('rb') as params_file:
            return LightFMBench(pickle.load(model_file), pickle.load(params_file))

    @staticmethod
    def initialize_with_optimization(
        optuna_params: DictConfig,
        learning_params: DictConfig,
        interactions_train: coo_matrix,
        weights_train: coo_matrix,
        interactions_val: coo_matrix
    ) -> "LightFMBench":
        """Builder for the model with a optimization.

        Args:
            optuna_params (DictConfig): Mapping with the optuna additional parameters,
                including the samper and the pruner. Defined in the config.
            learning_params (DictConfig): Learning parameters for the LightFM model.
            interactions_train (coo_matrix): Sparse matrix with the user-item interactions.
            weights_train (coo_matrix): Sparse matrix with the interaction weights.
                If interactions are binary, this option will be ignored.
            interactions_val (coo_matrix): Sparse matrix with user-item interactions for validation.

        Returns:
            LightFMBench: Model with the optimized parameters.
        """
        study = optuna.create_study(
            direction="maximize",
            sampler=instantiate(optuna_params["sampler"]),
            pruner=instantiate(optuna_params["pruner"])
        )
        study.optimize(
            partial(
                LightFMBench.objective,
                params_vary=optuna_params["hyperparameters_vary"],
                k_opt=optuna_params["k_optimization"],
                interactions_train=interactions_train,
                weights_train=weights_train,
                interactions_val=interactions_val,
                num_threads_opt=learning_params["num_threads"],
                num_epochs_opt=learning_params["num_epochs"]
            ),
            n_trials=optuna_params["n_trials"],
        )
        best_params = study.best_params.copy()
        if "const" in optuna_params["hyperparameters_vary"]:
            best_params.update(optuna_params["hyperparameters_vary"]["const"])

        logger.info("Best parameters are - %s", best_params)
        return LightFMBench.initialize_with_params(best_params)

    # pylint: disable=too-many-arguments
    @staticmethod
    def objective(
        trial: Trial,
        params_vary: DictConfig,
        k_opt: list[int],
        interactions_train: coo_matrix,
        weights_train: coo_matrix,
        interactions_val: coo_matrix,
        num_threads_opt: int,
        num_epochs_opt: int
    ) -> float:
        """Objective for the optimization. 

        Args:
            trial (Trial): trial object for the partial initialization
            params_vary (DictConfig): Mapping with the possible range of values.
                To unpack "method get_optimization_lists" is used
            k_opt (List[int]): List with values for k on which optimization will be provided.
                Result is averaged after all.
            interactions_train (coo_matrix): Sparse matrix with the user-item interactions
            weights_train (coo_matrix): Sparse matrix with the interaction weights.
                If interactions are binary, this option will be ignored.
            interactions_val (coo_matrix): Sparse matrix with user-item interactions for validation.
            num_threads_opt (int): Number of threads for the learning.
            num_epochs_opt (int): Number of training epochs.

        Returns:
            float: Averaged result of the model fitting
        """
        initial_model_parameters = get_optimization_lists(params_vary, trial)
        model = LightFMBench.initialize_with_params(initial_model_parameters)
        model.fit(interactions_train, weights_train, num_threads_opt, num_epochs_opt)
        relevant_ranks = model.get_relevant_ranks(
            interactions_val, interactions_train, num_threads_opt
        )
        metrics = []
        for k in k_opt:
            metrics.append(normalized_discounted_cumulative_gain(relevant_ranks, k))
        return np.mean(metrics)

    def fit(
        self,
        interactions: coo_matrix,
        weights: Optional[coo_matrix],
        num_threads: int,
        num_epochs: int
    ) -> None:
        """Wrapper of the model's "fit" method. Time measuring is added

        Args:
            interactions (coo_matrix): User-item interactions to fit in the sparse format
            weights (Optional[coo_matrix]): Weights for the interactions.
            num_threads (int): Number of threads for the learning.
            num_epochs (int): Number of training epochs.
        """
        start_time = time.time()
        self.model.fit(
            interactions,
            sample_weight=weights,
            num_threads=num_threads,
            epochs=num_epochs,
            verbose=True
        )
        self.learning_time = time.time() - start_time

    def save_model(self, path: Path) -> None:
        """Saving the model and the best parameters in the serializable pickle file.

        Args:
            path (Path): Path to the directory.
        """
        model_dir = get_saving_path(path)

        with \
            open(model_dir.joinpath("model.pcl"), "wb") as model_file, \
            open(model_dir.joinpath("params.pcl"), "wb") as params_file \
        :
            pickle.dump(self.model, model_file)
            pickle.dump(self.model_params, params_file)

    def get_relevant_ranks(
        self,
        test_interactions: coo_matrix,
        train_interactions: coo_matrix,
        num_threads: int,
        mp_threads: int = 10
    ) -> np.ndarray:
        """Get relevant ranks for the test interations.

        Args:
            test_interactions (coo_matrix): Sparse matrix with the test user-item interactions.
            train_interactions (coo_matrix): Sparse matrix with the train user-item interactions.
            num_threads (int): Number of threads to allocate.
            mp_threads (int): Number of threads for rank matrix generating. Defaults to 1.

        Returns:
            np.ndarray: Rank matrix.
        """
        start_time = time.time()
        ranks = self.model.predict_rank(
            test_interactions,
            train_interactions=train_interactions,
            num_threads=num_threads
        )
        self.predict_time = time.time() - start_time

        ranks = ranks.tocoo()
        ranks = pd.DataFrame((ranks.row, ranks.col, ranks.data)).T.astype(np.int32)

        max_len = ranks.groupby(0).count()[1].max()

        ranks_relevant = np.ones((ranks[0].unique().shape[0], max_len)) * -1

        with mp.Pool(mp_threads) as p:
            for i, row in tqdm(p.imap(
                self._async_rank_generation,
                map(lambda tup: (*tup, ranks), enumerate(ranks[0].unique()))
            ), total=ranks[0].nunique()):
                # print(result)
                ranks_relevant[i, :len(row)] = row

        # for i, user_id in tqdm(enumerate(ranks[0].unique()), total=ranks[0].nunique()):
        #     row = ranks[ranks[0] == user_id][2].values
        #     ranks_relevant[i, :len(row)] = row
        return ranks_relevant
    
    @staticmethod
    def _async_rank_generation(args: tuple[int, int, pd.DataFrame]) -> tuple[int, np.ndarray]:
        return args[0], args[2][args[2][0] == args[1]][2].values

    def recommend_k(
        self,
        train_interactions: coo_matrix,
        k: int,
        user_max: int = -1,
        num_threads: int = 16
    ) -> np.ndarray:
        """Get top-k recommended items for each user.

        Args:
            train_interactions (coo_matrix): Sparse matrix with the test user-item interactions.
            k (int): Value for the maximum predicted items
            user_max (int, optional): Number of users to make predictions.
                If -1, will take all users. Defaults to -1.

        Returns:
            np.ndarray: Matrix with recommended items.
        """
        total_users, total_items = train_interactions.shape
        user_relevant_items = {}
        for user, item in zip(train_interactions.row, train_interactions.col):
            if user not in user_relevant_items:
                user_relevant_items[user] = set((item,))
            else:
                user_relevant_items[user].add(item)

        preds = np.zeros((total_users, k))
        user_max = total_users if user_max == -1 else user_max

        # with mp.Pool(mp_threads) as pool:
        #     for i, row in tqdm(pool.imap(
        #         self._async_items_generation,
        #         map(
        #             lambda i: (i, self.model, user_relevant_items, total_items),
        #             list(range(min(user_max, total_users)))
        #         )
        #     ), total=min(user_max, total_users)):
        #         preds[i] = row[:k]

        for i in tqdm(range(min(user_max, total_users))):
            preds_user = self.model.predict(
                np.ones(total_items) * i, np.arange(total_items), num_threads=16
            )
            min_pred = preds_user.min()
            if i in user_relevant_items:
                preds_user[list(user_relevant_items[i])] = min_pred
            preds_user = np.argsort(preds_user)[::-1]
            preds[i] = preds_user[:k]

        return preds
    
    @staticmethod
    def _async_items_generation(args: tuple[int, LightFM, dict, int]) -> tuple[int, np.ndarray]:
        i, model, user_relevant_items, total_items = args
        preds_user = model.predict(
            np.ones(total_items) * i, np.arange(total_items)
        )
        min_pred = preds_user.min()
        if i in user_relevant_items:
            preds_user[list(user_relevant_items[i])] = min_pred
        preds_user = np.argsort(preds_user)[::-1]
        return i, preds_user
