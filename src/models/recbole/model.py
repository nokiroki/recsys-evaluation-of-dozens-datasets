"""Recbole module"""
import os
import pickle
import time
from typing import Optional, Mapping, Any, List
from pathlib import Path
from functools import partial

import numpy as np
import torch
import optuna
from optuna.trial import Trial
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from recbole.utils import init_seed, get_model, get_trainer
from recbole.data.dataloader.general_dataloader import (
    TrainDataLoader,
    FullSortEvalDataLoader,
)
from recbole.utils.case_study import full_sort_topk

from src.utils.logging import get_logger
from src.utils.metrics import normalized_discounted_cumulative_gain
from src.utils.processing import get_optimization_lists, get_saving_path

logger = get_logger(name=__name__)


class RecboleBench:
    """
    Recole Bench base class for model training, optimization, and evaluation.
    """

    def __init__(
        self,
        model: Any,
        train_loader: TrainDataLoader,
        model_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize the RecboleBench instance.

        Args:
            model (EASE or MultiVAE, or other in recbole):
            The EASE or MultiVAE auroencoder model instance.
            train_loader (TrainDataLoader): train dataloader from recbole package.
            model_params (Mapping[str, Any]): Model parameters.
        """
        self.model = model
        self.model_params = model_params

        self.train_loader = train_loader
        self.train_dataset = self.train_loader.dataset
        self.config = self.train_dataset.config

        init_seed(self.config["seed"], self.config["reproducibility"])

        self.learning_time: Optional[float] = None
        self.predict_time: Optional[float] = None

    @staticmethod
    def initialize_with_params(
        train_loader: TrainDataLoader,
        model_params: Optional[Mapping[str, Any]] = None,
    ) -> "RecboleBench":
        """
        Initialize RecboleBench with model parameters.
        Args:
            train_loader (TrainDataLoader): train dataloader from recbole package.
            model_params (Mapping[str, Any]): Model parameters
        Returns:
            RecboleBench: Initialized RecboleBench instance.
        Raises:
            ValueError: If the model name is not the name of an existing model.
        """

        dataset = train_loader.dataset
        config = dataset.config

        if model_params is None:
            model_params = {}

        for param_name, param_value in model_params.items():
            config[param_name] = param_value

        init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
        model = get_model(config["model"])(config, dataset).to(config["device"])

        return RecboleBench(model, train_loader)

    @staticmethod
    def initialize_saved_model(
        path: Path, train_loader: TrainDataLoader
    ) -> Optional["RecboleBench"]:
        """
        Initialize RecboleBench with a saved model.
        Args:
            path (str): Path to the saved model.
            train_loader (TrainDataLoader): train dataloader from recbole package.
        Returns:
            Optional[RecboleBench]:
            Initialized RecboleBench instance or None if the model file doesn't exist.
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
            return RecboleBench(
                pickle.load(model_file), train_loader, pickle.load(params_file)
            )

    @staticmethod
    def initialize_with_optimization(
        optuna_params: DictConfig,
        train_loader: TrainDataLoader,
        valid_loader: FullSortEvalDataLoader,
    ) -> "RecboleBench":
        """
        Initialize RecboleBench with hyperparameter optimization using Optuna.

        Args:
            model_name (str): Name of the model ("ease" or "multivae").
            optuna_params (DictConfig): Optuna hyperparameter optimization parameters.
            train_loader (TrainDataLoader): train dataloader from recbole package.
            valid_loader (FullSortEvalDataLoader): valid dataloader from recbole package.
            save_path (str, optional): path to save best params

        Returns:
            RecboleBench: Initialized RecboleBench instance.
        """

        study = optuna.create_study(
            direction="maximize",
            sampler=instantiate(optuna_params["sampler"]),
            pruner=instantiate(optuna_params["pruner"]),
        )
        study.optimize(
            partial(
                RecboleBench.objective,
                params_vary=optuna_params["hyperparameters_vary"],
                k_opt=optuna_params["k_optimization"],
                train_loader=train_loader,
                valid_loader=valid_loader,
            ),
            n_trials=optuna_params["n_trials"],
        )
        best_params = study.best_params.copy()
        if "const" in optuna_params["hyperparameters_vary"]:
            best_params.update(optuna_params["hyperparameters_vary"]["const"])

        logger.info("Best parameters are - %s", best_params)
        return RecboleBench.initialize_with_params(train_loader, best_params)

    @staticmethod
    def objective(
        trial: Trial,
        params_vary: DictConfig,
        k_opt: List[int],
        train_loader: TrainDataLoader,
        valid_loader: FullSortEvalDataLoader,
    ) -> float:
        """
        Objective function for hyperparameter optimization using Optuna.

        Args:
            trial (Trial): Optuna trial object.
            params_vary (DictConfig): Hyperparameters to optimize.
            k_opt (int): Number of top-k items to evaluate.
            train_loader (TrainDataLoader): train dataloader from recbole package.
            valid_loader (FullSortEvalDataLoader): valid dataloader from recbole package.

        Returns:
            float: Mean rank of the model evaluated on the validation set.
        """

        initial_model_parameters = get_optimization_lists(params_vary, trial)
        model = RecboleBench.initialize_with_params(
            train_loader, initial_model_parameters
        )
        model.fit(train_loader, valid_loader)
        relevant_ranks = model.get_relevant_ranks(valid_loader)
        metrics = []
        for k in k_opt:
            metrics.append(normalized_discounted_cumulative_gain(relevant_ranks, k))
        return np.mean(metrics)

    def fit(
        self, train_loader: TrainDataLoader, valid_loader: FullSortEvalDataLoader = None
    ) -> None:
        """
        Fit the Recbole EASE or MultiVAE model to the training data.

        Args:
            train_loader (TrainDataLoader): train dataloader from recbole package.
            valid_loader (FullSortEvalDataLoader): valid dataloader from recbole package.
        Returns:
            None
        """
        start_time = time.time()
        # trainer loading and initialization
        trainer = get_trainer(self.config["MODEL_TYPE"], self.config["model"])(
            self.config, self.model
        )
        trainer.fit(
            train_loader,
            valid_loader,
            saved=False,
            show_progress=self.config["show_progress"],
        )
        learning_time = time.time() - start_time

        self.learning_time = learning_time

    def save_model(self, path: Path) -> None:
        """
        Save the Recbole model to a file.

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

    def get_relevant_ranks(
        self,
        test_loader: FullSortEvalDataLoader,
    ) -> np.ndarray:
        """
        Calculate rank for test data

        Args:
            test_loader (FullSortEvalDataLoader): test dataloader from recbole package.
            train_loader (TrainDataLoader): train dataloader from recbole package.


        Returns:
            ranks (np.ndarray): ranks for every item from the loader and
                for every user in uid_series

        """

        start_time = time.time()

        user_list = test_loader.uid_list.numpy()
        item_num = test_loader.dataset.item_num
        batch_size = 4096
        start_idx = 0

        topk = np.empty((0, item_num), dtype="int32")

        with tqdm(
            (len(user_list) + batch_size - 1) // batch_size,
            desc="Calculating item ranks",
            unit=" batch",
        ) as pbar:
            while start_idx < len(user_list):
                batch = user_list[start_idx : start_idx + batch_size]

                topk_batch = full_sort_topk(batch, self.model, test_loader, item_num)[1]

                start_idx += batch_size

                # Concatenate the new array to the result_array
                topk = np.concatenate((topk, topk_batch.cpu()), axis=0)

                pbar.update(1)

        test_items_per_user = test_loader.uid2positive_item[user_list]
        max_test_items = max(map(len, test_items_per_user))
        # Create an array of ranks using the argsort function
        ranks = torch.argsort(torch.tensor(topk), axis=1)

        result_tensor = torch.zeros_like(ranks, dtype=torch.bool)

        # creating mask
        for user, test_item in enumerate(test_items_per_user):
            result_tensor[user, test_item] = True

        # # Make tensor of ranks, fill rows by -1 if len is less than max_test_items
        ranks[~result_tensor] = -1
        ranks, _ = torch.sort(ranks, dim=1, descending=True)
        ranks = ranks[:, :max_test_items]

        self.predict_time = time.time() - start_time

        return ranks.cpu().numpy()

    def recommend_k(
        self, test_loader: FullSortEvalDataLoader, k: int = 100
    ) -> np.ndarray:
        """
        Recommend top k items for users.
        Args:
            test_loader (FullSortEvalDataLoader): test dataloader from recbole package.
            k (int, optional): number of items in recommendation
        Returns:
            np.ndarray: 2-dimensional array with a row of item IDs for each user.
        """
        user_list = test_loader.uid_list.clone().detach()
        
        batch_size = 4096
        start_idx = 0
        
        topk = np.empty((0, k), dtype="int32")

        
        with tqdm(
            (len(user_list) + batch_size - 1) // batch_size,
            desc=f"Generating top_{k} recommendations",
            unit=" batch",
        ) as pbar:
            while start_idx < len(user_list):
                batch = user_list[start_idx : start_idx + batch_size]

                topk_batch = full_sort_topk(batch, self.model, test_loader, k)[1]

                start_idx += batch_size

                # Concatenate the new array to the result_array
                topk = np.concatenate((topk, topk_batch.cpu()), axis=0)

                pbar.update(1)

        return topk
