"""SASRecAdpBench module."""
from collections import defaultdict
from copy import deepcopy
from functools import partial
import os
import pickle
import random
import time
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
)

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
import optuna
from optuna.trial import Trial
import pandas as pd
from scipy.sparse import coo_matrix
import torch
from tqdm import tqdm

from src.utils.logging import get_logger
from src.utils.metrics import normalized_discounted_cumulative_gain
from src.utils.processing import get_optimization_lists
from .components import SASRec
from .utils import WarpSampler

logger = get_logger(name=__name__)


class SASRecAdpBench:
    """SAS Model Bench (adaptated prediction) base class for model training, optimization, and evaluation."""

    def __init__(
        self,
        model: SASRec,
        model_params: Mapping[str, Any],
    ) -> None:
        """
        Initialize the SASRecAdpBench instance.

        Args:
            model (SASRec):
            The SASRec model instance.
            model_params (Mapping[str, Any]): Model parameters.
        """
        if model_params is None:
            model_params = {}
        device = model_params["device"] if "device" in model_params else "cpu"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.dev = self.device
        self.learning_time: Optional[float] = (
            None
            if "learning_time" not in model_params
            else model_params["learning_time"]
        )
        self.predict_time: Optional[float] = (
            None if "predict_time" not in model_params else model_params["predict_time"]
        )

        self.users = None if "users" not in model_params else model_params["users"]

        self.usernum = (
            None if "user_num" not in model_params else model_params["user_num"]
        )
        self.itemnum = (
            None if "item_num" not in model_params else model_params["item_num"]
        )
        self.num_batch = (
            None if "num_batch" not in model_params else model_params["num_batch"]
        )

        self.user_valid = (
            None if "user_valid" not in model_params else model_params["user_valid"]
        )
        self.user_train = (
            None if "user_train" not in model_params else model_params["user_train"]
        )

        self.val_step = (
            -1 if "val_step" not in model_params else model_params["val_step"]
        )

        self.batch_size = (
            64 if "batch_size" not in model_params else model_params["batch_size"]
        )
        self.maxlen = 50 if "maxlen" not in model_params else model_params["maxlen"]
        self.lr_value = (
            1e-3
            if "learning_rate" not in model_params
            else model_params["learning_rate"]
        )
        self.l2_emb = 0.0 if "l2_emb" not in model_params else model_params["l2_emb"]

        self.epoch_start_idx = (
            0
            if "epoch_start_idx" not in model_params
            else model_params["epoch_start_idx"]
        )


        self.user_column = (
            None if "user_column" not in model_params else model_params['user_column']
        )
        self.item_column = (
            None if "item_column" not in model_params else model_params['item_column']
        )
        self.inf_type = model_params['type']
        if self.inf_type not in ['nip', 'adaptated']:
            raise NotImplementedError(f"SASRec work only with NIP or Adaptated modes.")
        self.old_style = model_params['old_style']

        logger.info("Batch size is %d", self.batch_size)

    @staticmethod
    def initialize_with_params(model_init_params: DictConfig) -> "SASRecAdpBench":
        """
        Initialize SASRecAdpBench with model parameters.

        Args:
            model_name (str): Name of the model (SASRec).
            model_init_params (Mapping[Any, Any]): Model initialization parameters.

        Returns:
            SASRecAdpBench: Initialized SASRecAdpBench instance.

        Raises:
            NotImplementedError: If the model name is not supported.
        """
        stop_list_keys = [
            "batch_size",
            "learning_rate",
            "l2_emb",
            "epoch_start_idx",
            "user_column",
            "item_column",
            "type",
            "old_style",
        ]
        model_params = {
            k: v for k, v in model_init_params.items() if not k in stop_list_keys
        }
        model = SASRec(**DictConfig(model_params))

        return SASRecAdpBench(model, model_init_params)

    @staticmethod
    def initialize_saved_model(path: str) -> Optional["SASRecAdpBench"]:
        """
        Initialize SASRecAdpBench with a saved model.

        Args:
            path (str): Path to the saved model.

        Returns:
            Optional[SASRecAdpBench]:
            Initialized SASRecAdpBench instance or None if the model file doesn't exist.
        """
        try:
            with open(os.path.join(path, "model.pcl"), "rb") as file:
                file_data = pickle.load(file)
                return SASRecAdpBench(file_data[0], file_data[1])
        except OSError:
            logger.error("Model does not exists! Try initialize_with_params method")
            return None

    def _preproc(self, data: pd.DataFrame, train: bool = False, old_style: bool = False) -> Dict[int, List[int]]:
        if old_style:
            self.user_map = None
            return {k: list(table[self.item_column].values) for k, table in data.groupby(self.user_column)}
        if train:
            self.user_map = {x: i for i, x in enumerate(data[self.user_column].unique(), 1)}
        # change order
        new_data = data.replace({self.user_column: self.user_map})

        return {k: list(table[self.item_column].values) for k, table in new_data.groupby(self.user_column)}


    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        num_threads: int,
        num_epochs: int,
        val_step: int,
    ) -> None:
        """
        Fit the SASRec model to the training data.

        Args:
            train_data (pd.DataFrame): \
                Training data (user-date-item table).
            val_data (pd.DataFrame): \
                Validation data (user-date-item table).
            num_epochs (int): number of training epochs in fitting.
            val_step (int): Step of validation running.

        Returns:
            None
        """
        start_time = time.time()

        # move data from pd.DataFrame to dict[int, list[int]]
        self.user_train = self._preproc(train_data, train = True, old_style=self.old_style,)
        self.user_valid = self._preproc(val_data, old_style=self.old_style,)


        self.users = list(self.user_train.keys())

        self.usernum = max(self.users)

        self.num_batch = len(self.user_train) // self.batch_size

        self.val_step = val_step

        sampler = WarpSampler(
            self.user_train,
            self.usernum,
            self.itemnum,
            batch_size=self.batch_size,
            maxlen=self.maxlen,
            n_workers=5,
        )

        self._full_train(
            sampler,
            epochs=num_epochs,
            verbose=True,
        )
        self.learning_time = time.time() - start_time

    def _full_train(self, sampler: WarpSampler, epochs: int, verbose=True) -> None:
        """Train SasRec model.

        Args:
            sampler (WarpSampler): data structure to generate batch samples.
            epochs (int): number of training epochs.
            verbose (bool): parameter to verbose statistics.
        """
        self.model.train()
        # ce_criterion = torch.nn.CrossEntropyLoss()
        # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
        bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
        adam_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr_value, betas=(0.9, 0.98)
        )
        cur_best_val = -1.0
        for epoch in range(self.epoch_start_idx, epochs + 1):
            for step in range(self.num_batch):
                # Tuples to ndarray
                user_id, seq, pos, neg = sampler.next_batch()
                user_id, seq, pos, neg = (
                    np.array(user_id),
                    np.array(seq),
                    np.array(pos),
                    np.array(neg),
                )
                pos_logits, neg_logits = self.model(user_id, seq, pos, neg)
                pos_labels, neg_labels = torch.ones(
                    pos_logits.shape, device=self.device
                ), torch.zeros(neg_logits.shape, device=self.device)

                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in self.model.item_emb.parameters():
                    loss += self.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                if verbose and step % 200 == 0:
                    logger.info(
                        "loss in epoch %d iteration %d: % 2.3f "
                        % (epoch, step, loss.item())
                    )  # expected 0.4~0.6 after init few epochs

            if (epoch % self.val_step == 0 or epoch == epochs) and self.val_step != -1:
                self.model.eval()
                ndcg_value = self.evaluate_valid()
                if ndcg_value > cur_best_val:
                    logger.info("New best NDCG@10: %f", ndcg_value)
                    current_best_model = deepcopy(self.model)
                    cur_best_val = ndcg_value

                self.model.train()

        sampler.close()
        # change current model
        self.model = current_best_model
        logger.debug("Done")

    def evaluate_valid(self) -> float:
        """Calculate validation metric.

        Returns:
            float: metric on validation.
        """
        ndcg_value = 0.0
        valid_user = 0.0

        if self.usernum > 10000:
            users = random.sample(range(1, self.usernum + 1), 10000)
        else:
            users = range(1, self.usernum)

        for user_id in users:

            if user_id not in self.user_train or user_id not in self.user_valid:
                continue

            if len(self.user_train[user_id]) < 1 or len(self.user_valid[user_id]) < 1:
                continue

            seq = np.zeros([self.maxlen], dtype=np.int32)
            idx = self.maxlen - 1
            for i in reversed(self.user_train[user_id]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

            rated = set(self.user_train[user_id])
            rated.add(0)
            item_idx = [self.user_valid[user_id][0]]
            for _ in range(100):
                t_item = np.random.randint(1, self.itemnum + 1)
                while t_item in rated:
                    t_item = np.random.randint(1, self.itemnum + 1)
                item_idx.append(t_item)
            predictions = -self.model.predict(
                *[np.array(l) for l in [[user_id], [seq], item_idx]]
            )
            predictions = predictions[0]

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            if rank < 10:
                ndcg_value += 1 / np.log2(rank + 2)

        return ndcg_value / valid_user

    @staticmethod
    def initialize_with_optimization(
        optuna_params: DictConfig,
        params: DictConfig,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> "SASRecAdpBench":
        """
        Initialize SASRecAdpBench with hyperparameter optimization using Optuna.

        Args:
            model_name (str): Name of the model ("SASRec").
            optuna_params (DictConfig): Optuna hyperparameter optimization parameters.
            train_data (pd.DataFrame): Sparse training interactions matrix.
            weights_train (pd.DataFrame): Sparse sample weight matrix for training interactions.
            val_data (pd.DataFrame): Sparse validation interactions matrix.
            save_path (str, optional): Path to save the best parameters file.

        Returns:
            SASRecAdpBench: Initialized SASRecAdpBench instance.
        """

        params.optuna_optimizer.hyperparameters_vary.const["user_num"] = max(
            train_data[optuna_params.hyperparameters_vary.const['user_column']]
        )
        params.optuna_optimizer.hyperparameters_vary.const["item_num"] = max(
            train_data[optuna_params.hyperparameters_vary.const['item_column']]
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=instantiate(optuna_params["sampler"]),
            pruner=instantiate(optuna_params["pruner"]),
        )
        learning_params = params["learning"]
        study.optimize(
            partial(
                SASRecAdpBench.objective,
                params_vary=optuna_params["hyperparameters_vary"],
                k_opt=optuna_params["k_optimization"],
                train_data=train_data,
                val_data=val_data,
                num_threads_opt=learning_params["num_threads"],
                num_epochs_opt=learning_params["num_epochs"],
                val_step=learning_params["val_step"],
            ),
            n_trials=optuna_params["n_trials"],
        )
        best_params = study.best_params.copy()
        if "const" in optuna_params["hyperparameters_vary"]:
            best_params.update(optuna_params["hyperparameters_vary"]["const"])

        if save_path is not None:
            os.makedirs(
                save_path,
                exist_ok=True,
            )
            with open(os.path.join(save_path, "best_path.pcl"), "wb") as f:
                pickle.dump(best_params, f)
        logger.info("Best parameters are - %s", best_params)
        return SASRecAdpBench.initialize_with_params(best_params)

    @staticmethod
    def objective(
        trial: Trial,
        params_vary: DictConfig,
        k_opt: List[int],
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        num_threads_opt: int,
        num_epochs_opt: int,
        val_step: int,
    ) -> float:
        """
        Objective function for hyperparameter optimization using Optuna.

        Args:
            trial (Trial): Optuna trial object.
            model_name (str): Name of the model ("als" or "bpr").
            params_vary (DictConfig): Hyperparameters to optimize.
            k_opt (int): Number of top-k items to evaluate.
            train_data (pd.DataFrame): Train sorted data.
            val_data (pd.DataFrame): Validation sorted data.
            num_epochs_opt (int): number of epochs in optimizing trial.
            val_step (int): Step of validation running.

        Returns:
            float: Mean rank of the model evaluated on the validation set.
        """
        initial_model_parameters = get_optimization_lists(params_vary, trial)
        model = SASRecAdpBench.initialize_with_params(initial_model_parameters)
        model.fit(
            train_data,
            val_data,
            num_threads_opt,
            num_epochs_opt,
            val_step,
        )
        relevant_ranks = model.get_relevant_ranks(
            val_data,
            False,
            num_threads_opt,
        )

        metrics = []
        for k in k_opt:
            metrics.append(normalized_discounted_cumulative_gain(relevant_ranks, k))
        return np.mean(metrics)

    def get_relevant_ranks(
        self,
        test_data: pd.DataFrame,
        test: bool,
        num_threads: int = 0,
    ) -> np.ndarray:
        """
        Get relevant ranks for the test interactions.

        Args:
            test_data (pd.DataFrame): Test interactions matrix (user-item interactions).
            num_threads (int): Number of processes. Unused parameter.

        Returns:
            np.ndarray: Array of relevant ranks for each user in the test set.
        """
        self.model.eval()
        start_time = time.time()

        # concat train and/or val to create current time event
        if test:

            new_test = self._preproc(test_data, old_style=self.old_style,)

            train_val = {}

            max_size = 0

            for k, seq in new_test.items():
                if k not in self.user_train and k not in self.user_valid:
                    continue
                if k in self.user_train:
                    train_val[k] = self.user_train[k]
                if k in self.user_valid:
                    train_val[k] = train_val[k] + self.user_valid[k] if k in self.user_train else self.user_valid[k]
                
                if len(seq) > max_size:
                    max_size = len(seq)
        else:
            new_test = self.user_valid

            train_val = {}

            max_size = 0

            for k, seq in new_test.items():
                if k not in self.user_train:
                    continue
                if k in self.user_train:
                    train_val[k] = self.user_train[k]
                
                if len(seq) > max_size:
                    max_size = len(seq)            

        if self.inf_type == "nip":
            output = np.zeros((len(new_test), 1)) - 1
        if self.inf_type == "adaptated":
            output = np.zeros((len(new_test), max_size)) - 1



        for i, (user_id, full_seq) in enumerate(train_val.items()):

            # drop users with short train/val item sequence
            if (
                len(full_seq) < 1
            ):
                continue

            # cut input sequence
            seq = full_seq[-self.maxlen:] if len(full_seq) >= self.maxlen else [0 for _ in range(self.maxlen - len(full_seq))] + full_seq

            rated = set(full_seq)
            rated.add(0)
            rated.update(train_val[user_id])
            if self.inf_type == "nip":
                rated.add(new_test[user_id][0])
                item_idx = [new_test[user_id][0]] \
                    + [one_item for one_item in np.arange(1, self.itemnum) \
                    if one_item not in rated]
            if self.inf_type == "adaptated":
                rated.update(new_test[user_id])
                item_idx = new_test[user_id] \
                    + [one_item for one_item in np.arange(1, self.itemnum) \
                    if one_item not in rated]


            predictions = -self.model.predict(
                *[np.array(l) for l in [[user_id], [seq], item_idx]]
            )
            predictions = predictions[0]  # - for 1st argsort DESC

            if self.inf_type == "nip":
                rank = predictions.argsort().argsort().detach().cpu()[0]
                
                output[i, 0] = rank
            if self.inf_type == "adaptated":

                rank = predictions.argsort().argsort().detach().cpu()[:len(new_test[user_id])]

                output[i, :len(rank)] = rank          

        self.predict_time = time.time() - start_time

        return output

    def recommend_k(
        self,
        test_data: pd.DataFrame,
        test: bool,
        k: int,
    ) -> np.ndarray:
        """
        Recommend top k items for users.

        Args:
            k (int): The number of results to return.
            test_data (pd.DataFrame):
                pd.DataFrame data representing full data in future.

        Returns:
            np.ndarray: 2-dimensional array with a row of item IDs for each user.
        """

        # return preds
        self.model.eval()
        start_time = time.time()

        # concat train and val to create current time event
        if test:

            new_test = self._preproc(test_data, old_style=self.old_style,)

            train_val = {}

            max_size = 0

            for i, seq in new_test.items():
                if i not in self.user_train and i not in self.user_valid:
                    continue
                if i in self.user_train:
                    train_val[i] = self.user_train[i]
                if i in self.user_valid:
                    train_val[i] = train_val[i] + self.user_valid[i] if i in self.user_train else self.user_valid[i]
                
                if len(seq) > max_size:
                    max_size = len(seq)
        else:
            new_test = self.user_valid

            train_val = {}

            max_size = 0

            for i, seq in new_test.items():
                if i not in self.user_train:
                    continue
                if i in self.user_train:
                    train_val[i] = self.user_train[i]
                
                if len(seq) > max_size:
                    max_size = len(seq)

        output = np.zeros((len(new_test), k)) - 1

        for i, (user_id, full_seq) in enumerate(train_val.items()):

            # drop users with short train/val item sequence
            if (
                len(full_seq) < 1
            ):
                continue

            # cut input sequence
            seq = full_seq[-self.maxlen:] if len(full_seq) >= self.maxlen else [0 for _ in range(self.maxlen - len(full_seq))] + full_seq

            rated = set(full_seq)
            rated.add(0)
            rated.update(train_val[user_id])
            if self.inf_type == "nip":
                rated.add(new_test[user_id][0])
                item_idx = [new_test[user_id][0]] \
                    + [one_item for one_item in np.arange(1, self.itemnum) \
                    if one_item not in rated]
            if self.inf_type == "adaptated":
                rated.update(new_test[user_id])
                item_idx = new_test[user_id] \
                    + [one_item for one_item in np.arange(1, self.itemnum) \
                    if one_item not in rated]

            predictions = -self.model.predict(
                *[np.array(l) for l in [[user_id], [seq], item_idx]]
            )
            predictions = predictions[0]  # - for 1st argsort DESC
            output[i, :k] = np.array(item_idx)[predictions.argsort(descending=True,).detach().cpu().numpy()][:k]

        self.predict_time = time.time() - start_time


        return output

    def save_model(self, path: str) -> None:
        """
        Save the MSRec SasRec model to a file.

        Args:
            path (str): Path to the directory where the model file should be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        logger.info("Saving: %s", os.path.join(path, "model.pcl"))

        with open(os.path.join(path, "model.pcl"), "wb") as file:
            pickle.dump(
                [
                    self.model,
                    {
                        "batch_size": self.batch_size,
                        "maxlen": self.maxlen,
                        "learning_rate": self.lr_value,
                        "l2_emb": self.l2_emb,
                        "epoch_start_idx": self.epoch_start_idx,
                        "learning_time": self.learning_time,
                        "predict_time": self.predict_time,
                        "users": self.users,
                        "usernum": self.usernum,
                        "itemnum": self.itemnum,
                        "num_batch": self.num_batch,
                        "user_valid": self.user_valid,
                        "user_train": self.user_train,
                        "val_step": self.val_step,
                        "user_transform": self.user_map,
                        "type": self.inf_type,
                    },
                ],
                file,
            )
