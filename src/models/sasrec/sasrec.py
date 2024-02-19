"""SASRecBench module."""
from collections import (
    defaultdict,
    OrderedDict,
)
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
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)

from src.utils.logging import get_logger
from src.utils.metrics import normalized_discounted_cumulative_gain
from src.utils.processing import (
    get_optimization_lists,
    pandas_to_aggregate,
)
from .components import (
    CausalLMDataset,
    CausalLMPredictionDataset,
    PaddingCollateFn,
    SASRec,
    SeqRec,
)
from .utils import add_time_idx

logger = get_logger(name=__name__)

class SASRecBench:
    """SAS Model Bench base class for model training, optimization, and evaluation."""

    def __init__(
            self,
            model: SASRec,
            model_params: Mapping[str, Any],
        ) -> None:
            """
            Initialize the SASRecBench instance.

            Args:
                model (SASRec):
                The SASRec model instance.
                model_params (Mapping[str, Any]): Model parameters.
            """

            self.model = model
            self.model_params = model_params
            device = model_params["device"] if "device" in model_params else "cpu"
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            self.learning_time: Optional[float] = None
            self.predict_time: Optional[float] = None

            self.trainer = None
            self.pl_model = None

    @staticmethod
    def initialize_with_params(model_init_params: DictConfig) -> "SASRecBench":
        """
        Initialize SASRec with model parameters.

        Args:
            model_name (str): Name of the model (SASRec).
            model_init_params (Mapping[Any, Any]): Model initialization parameters.

        Returns:
            SASRec: Initialized SASRec instance.

        Raises:
            NotImplementedError: If the model name is not supported.
        """
        model = SASRec(**DictConfig(model_init_params['sasrec_params']))

        return SASRecBench(model, model_init_params)

    @staticmethod
    def initialize_saved_model(path: str) -> Optional["SASRecBench"]:
        """
        Initialize SASRecBench with a saved model.

        Args:
            path (str): Path to the saved model.

        Returns:
            Optional[SASRecBench]:
            Initialized SASRecBench instance or None if the model file doesn't exist.
        """
        try:
            with open(os.path.join(path, "model.pcl"), "rb") as file:
                file_data = pickle.load(file)
                return SASRecBench(file_data[0], file_data[1])
        except OSError:
            logger.error("Model does not exists! Try initialize_with_params method")
            return None

    @staticmethod
    def initialize_with_optimization(
        optuna_params: DictConfig,
        params: DictConfig,
        data: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> "SASRecBench":
        """
        Initialize SASRecBench with hyperparameter optimization using Optuna.

        Args:
            model_name (str): Name of the model ("SASRec").
            optuna_params (DictConfig): Optuna hyperparameter optimization parameters.
            data (pd.DataFrame): Sparse training interactions matrix.
            save_path (str, optional): Path to save the best parameters file.

        Returns:
            SASRecBench: Initialized SASRecBench instance.
        """

        study = optuna.create_study(
            direction="maximize",
            sampler=instantiate(optuna_params["sampler"]),
            pruner=instantiate(optuna_params["pruner"]),
        )
        learning_params = params["learning"]
        study.optimize(
            partial(
                SASRecBench.objective,
                params_vary=optuna_params["hyperparameters_vary"],
                params_const=params,
                k_opt=optuna_params["k_optimization"],
                data=data,
                num_threads_opt=learning_params["num_threads"],
                num_epochs_opt=learning_params["num_epochs"],
            ),
            n_trials=optuna_params["n_trials"],
        )
        best_params = study.best_params.copy()
        if "const" in optuna_params["hyperparameters_vary"]:
            best_params.update(optuna_params["hyperparameters_vary"]["const"])
        params.model['sasrec_params'] = {k: v for k, v in best_params.items() if k != 'l2_emb'}
        params.model.seqrec_module['l2_emb'] = best_params['l2_emb']

        if save_path is not None:
            os.makedirs(
                save_path,
                exist_ok=True,
            )
            with open(os.path.join(save_path, "best_path.pcl"), "wb") as f:
                pickle.dump(params.model, f)
        logger.info("Best parameters are - %s", params.model['sasrec_params'])
        return SASRecBench.initialize_with_params(params.model)

    @staticmethod
    def objective(
        trial: Trial,
        params_vary: DictConfig,
        params_const: DictConfig,
        k_opt: List[int],
        data: pd.DataFrame,
        num_threads_opt: int,
        num_epochs_opt: int,
    ) -> float:
        """
        Objective function for hyperparameter optimization using Optuna.

        Args:
            trial (Trial): Optuna trial object.
            model_name (str): Name of the model ("als" or "bpr").
            params_vary (DictConfig): Hyperparameters to optimize.
            k_opt (int): Number of top-k items to evaluate.
            data (pd.DataFrame): Train sorted data.
            num_epochs_opt (int): number of epochs in optimizing trial.
            val_step (int): Step of validation running.

        Returns:
            float: Mean rank of the model evaluated on the validation set.
        """
        initial_model_parameters = get_optimization_lists(params_vary, trial)
        params_const.model['sasrec_params'] = {k: v for k, v in initial_model_parameters.items() if k != 'l2_emb'}
        params_const.model.seqrec_module['l2_emb'] = initial_model_parameters['l2_emb']
        model = SASRecBench.initialize_with_params(params_const['model'])
        model.fit(
            data,
            num_threads_opt,
            num_epochs_opt,
        )

        train_data, val_data, val_full, _ =  model._prepare_data(data)
        top_100_items = model.recommend_k(train_data, 100)
        metrics = []
        if isinstance(k_opt, int):
            k_opt = [k_opt]
        
        
        for k in k_opt:
            metrics.append(normalized_discounted_cumulative_gain(
                top_100_items, pandas_to_aggregate(val_data,), k)
            )
        return np.mean(metrics)

    @staticmethod
    def _prepare_data(data, user_col: str = "userId", item_col: str = "itemId",):

        data = add_time_idx(data, user_col=user_col, sort=False,)

        train = data[data.time_idx_reversed >= 1]
        validation = data[data.time_idx_reversed == 0]
        validation_full = data[data.time_idx_reversed >= 0]

        item_count = data[item_col].max()

        return train, validation, validation_full, item_count

    def _create_dataloaders(self, train, validation,):

        train_dataset = CausalLMDataset(train, **self.model_params['dataset'])
        eval_dataset = CausalLMPredictionDataset(
            validation, max_length=self.model_params.dataset.max_length, validation_mode=True,)

        train_loader = DataLoader(train_dataset, batch_size=self.model_params.dataloader.batch_size,
                                shuffle=True, num_workers=self.model_params.dataloader.num_workers,
                                collate_fn=PaddingCollateFn())
        eval_loader = DataLoader(eval_dataset, batch_size=self.model_params.dataloader.test_batch_size,
                                shuffle=False, num_workers=self.model_params.dataloader.num_workers,
                                collate_fn=PaddingCollateFn())
        
        return train_loader, eval_loader
    
    def _get_pl_trainer(self):
        if hasattr(self.model_params.dataset, 'num_negatives') and self.model_params.dataset.num_negatives:
            seqrec_module = SeqRecWithSampling(self.model, **self.model_params['seqrec_module'])
        else:
            seqrec_module = SeqRec(self.model, **self.model_params['seqrec_module'])
        early_stopping = EarlyStopping(monitor="val_ndcg", mode="max",
                                    patience=self.model_params.patience, verbose=False)
        model_summary = ModelSummary(max_depth=4)
        checkpoint = ModelCheckpoint(save_top_k=1, monitor="val_ndcg",
                                    mode="max", save_weights_only=True)
        progress_bar = TQDMProgressBar(refresh_rate=100)
        callbacks=[early_stopping, model_summary, checkpoint, progress_bar]

        trainer = pl.Trainer(callbacks=callbacks, gpus=1, enable_checkpointing=True,
                            **self.model_params['trainer_params'])
        
        return trainer, seqrec_module, checkpoint

    def fit(
        self,
        data: pd.DataFrame,
        num_threads: int,
        num_epochs: int,
    ) -> None:
        """
        Fit the SASRec model to the training data.

        Args:
            data (pd.DataFrame): \
                Training data (user-date-item table).
            num_epochs (int): number of training epochs in fitting.

        Returns:
            None
        """
        start_time = time.time()

        train_data, val_data, val_full, _ =  self._prepare_data(data)

        train_loader, eval_loader = self._create_dataloaders(train_data, val_full)

        trainer, pl_model, checkpoint = self._get_pl_trainer()


        trainer.fit(model=pl_model,
                train_dataloaders=train_loader,
                val_dataloaders=eval_loader)
        
        best_state_dict = torch.load(checkpoint.best_model_path)['state_dict']
        new_state_dict = OrderedDict()
        for key in list(best_state_dict.keys()):
            new_state_dict[key[6:]] = best_state_dict[key]

        self.model.load_state_dict(new_state_dict)

        self.trainer = trainer
        self.pl_model = pl_model

        self.learning_time = time.time() - start_time

    def recommend_k(
        self,
        data: pd.DataFrame,
        k: int,
    ) -> np.ndarray:
        """
        Recommend top k items for users.

        Args:
            k (int): The number of results to return.
            data (pd.DataFrame):
                pd.DataFrame data representing full data in future.

        Returns:
            np.ndarray: 2-dimensional array with a row of item IDs for each user.
        """

        # return preds
        self.model.eval()
        start_time = time.time()

        data = add_time_idx(data, sort=False,)

        predict_dataset = CausalLMPredictionDataset(data, max_length=self.model_params.dataset.max_length,)

        predict_loader = DataLoader(
            predict_dataset, shuffle=False,
            collate_fn=PaddingCollateFn(),
            batch_size=self.model_params.dataloader.test_batch_size,
            num_workers=self.model_params.dataloader.num_workers)
        
        self.pl_model.predict_top_k = max(self.model_params.top_k_metrics)
        preds = self.trainer.predict(model=self.pl_model, dataloaders=predict_loader)

        user_ids = np.hstack([pred['user_ids'] for pred in preds])

        preds = np.vstack([pred['preds'] for pred in preds])
        user_ids = np.repeat(user_ids[:, None], repeats=preds.shape[1], axis=1)

        recs = pd.DataFrame({'user_id': user_ids.flatten(),
                            'item_id': preds.flatten(),})

        predicts = []
        min_size = 100

        for x in range(max(recs['user_id']) + 1):
            if len(list(recs[recs['user_id'] == x]['item_id'])) == 0:
                predicts.append([-1 for i in range(100)])
            else:
                predicts.append(list(recs[recs['user_id'] == x]['item_id']))
            
        predicts = np.array(predicts)




        self.predict_time = time.time() - start_time

        self.model.train()

        return predicts

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
                        "model_params": self.model_params,
                    },
                ],
                file,
            )