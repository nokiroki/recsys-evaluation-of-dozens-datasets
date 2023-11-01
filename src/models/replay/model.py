"""ReplayBench module."""
from typing import Optional, Any, Mapping, Union
from functools import partial
from pathlib import Path
import pickle
import os
import time

from hydra.utils import instantiate
from omegaconf import DictConfig

import numpy as np

import optuna
from optuna.trial import Trial

from pyspark.sql import DataFrame
from replay.models import ItemKNN, SLIM
from replay.model_handler import save, load

from tqdm.auto import tqdm

from src.utils.logging import get_logger
from src.utils.processing import get_saving_path, get_optimization_lists
from src.utils.metrics import normalized_discounted_cumulative_gain as ndcg

logger = get_logger(name=__name__)


class RePlayBench:
    """Main class for the RePlay model benchmarking.
    Replay is a framework with variaty of the implemented methods.
    For the more detailed documenation please refer to the
    https://github.com/sb-ai-lab/RePlay/tree/main#quickstart
    """

    def __init__(
        self,
        model: Union[ItemKNN, SLIM],
        model_params: Optional[Mapping[str, Any]] = None
    ) -> None:
        self.model = model
        self.model_params = model_params

        self.learning_time: Optional[float] = None
        self.predict_time: Optional[float] = None

    @staticmethod
    def initialize_with_params(
        model_type: str,
        model_init_params: Mapping[Any, Any]
    ) -> "RePlayBench":
        """Builder for the instantiating a model from the predefined parameters.

        Args:
            model_type (str): Name of the model to initialize.
            model_init_params (Mapping[Any, Any]): Any parameters for the model initialization.

        Returns:
            RePlayBench: LightFMBench object initialized from the parameters.
        """
        model = RePlayBench.get_model(model_type, model_init_params)

        return RePlayBench(model, model_init_params)

    @staticmethod
    def initialize_saved_model(path: Path) -> Optional["RePlayBench"]:
        """Builder for the saved model.

        Args:
            path (Path): Path to the model.

        Returns:
            LightFMBench: Loaded model.
        """
        if not path.exists():
            logger.error("Directory does not exist")
            return None
        if not os.listdir(path):
            logger.error("Directory is empty")
            return None
        model_params_path = path.joinpath("params.pcl")
        if model_params_path.exists():
            with open(model_params_path, 'rb') as file:
                model_params = pickle.load(file)
        else:
            logger.warning("Attribute 'best params' not found")
            model_params = None

        try:
            model = load(path)
            return RePlayBench(model, model_params)
        except OSError as error:
            logger.error(error.args[0])
            return None


    @staticmethod
    def initialize_with_optimization(
        model_type: str,
        optuna_params: DictConfig,
        interactions_train: DataFrame,
        interactions_val: DataFrame
    ) -> "RePlayBench":
        study = optuna.create_study(
            direction="maximize",
            sampler=instantiate(optuna_params["sampler"]),
            pruner=instantiate(optuna_params["pruner"])
        )
        study.optimize(
            partial(
                RePlayBench.objective,
                params_vary=optuna_params["hyperparameters_vary"],
                model_type=model_type,
                k_opt=optuna_params["k_optimization"],
                interactions_train=interactions_train,
                interactions_val=interactions_val,
            ),
            n_trials=optuna_params["n_trials"],
        )
        best_params = study.best_params.copy()
        if "const" in optuna_params["hyperparameters_vary"]:
            best_params.update(optuna_params["hyperparameters_vary"]["const"])

        logger.info("Best parameters are - %s", best_params)
        return RePlayBench.initialize_with_params(model_type ,best_params)

    
    @staticmethod
    def objective(
        trial: Trial,
        params_vary: DictConfig,
        model_type: str,
        k_opt: list[int],
        interactions_train: DataFrame,
        interactions_val: DataFrame
    ) -> float:
        """Objective for the optimization. 

        Args:
            trial (Trial): trial object for the partial initialization
            params_vary (DictConfig): Mapping with the possible range of values.
                To unpack "method get_optimization_lists" is used
            k_opt (List[int]): List with values for k on which optimization will be provided.
                Result is averaged after all.
            interactions_train (coo_matrix): Sparse matrix with the user-item interactions
            interactions_val (coo_matrix): Sparse matrix with user-item interactions for validation.

        Returns:
            float: Averaged result of the model fitting
        """
        initial_model_parameters = get_optimization_lists(params_vary, trial)
        model = RePlayBench.initialize_with_params(model_type, initial_model_parameters)
        model.fit(interactions_train)
        relevant_ranks = model.get_relevant_ranks(interactions_train, max(k_opt), interactions_val)
        return np.mean([ndcg(relevant_ranks, k) for k in k_opt])


    @staticmethod
    def get_model(model_type: str, model_init_params: Mapping[Any, Any]) -> Union[ItemKNN, SLIM]:
        """Get model depending on "model_type" parameter. Available options:
            SLIM
            item_knn

        Args:
            model_type (str): Name of the model to initialize.
            model_init_params (Mapping[Any, Any]): Any parameters for the model initialization.

        Raises:
            ValueError: Error would be raised if the model with "model_type" name
                was not implemented yet.

        Returns:
            Union[ItemKNN, SLIM]: Model instance.
        """
        if model_type == "item_knn":
            model = ItemKNN(**model_init_params)
        elif model_type == "slim":
            model = SLIM(**model_init_params)
        else:
            raise ValueError(f"model type {model_type} was not implemented")

        return model

    def fit(self, interactions: DataFrame) -> None:
        """Wrapper of the model's "fit" method. Time measuring is added

        Args:
            interactions (DataFrame): Train interactions.
        """
        start_time = time.time()
        self.model.fit(interactions)
        self.learning_time = time.time() - start_time

    def save_model(self, path: Path) -> None:
        """Saving the model and the best parameters in the serializable pickle file.

        Args:
            path (Path): Path to the directory.

        Raises:
            Error: Raised if model was not saved
        """
        try:
            model_dir = get_saving_path(path)

            save(self.model, model_dir, overwrite=True)
            if self.model_params is not None:
                with open(model_dir.joinpath("params.pcl"), 'wb') as file:
                    pickle.dump(self.model_params, file)
            else:
                logger.warning("Attribute 'best_params' is missing and will not be saved")
        except Exception as error:
            logger.error(error.args[0])
            logger.error("Model has not been saved!")
            raise

    def get_relevant_ranks(
        self,
        train_interactions: DataFrame,
        k: int,
        test_interactions: DataFrame,
        recommended_items: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get relevant ranks for the test interations.

        Args:
            train_interactions (DataFrame): Sparse matrix with the train user-item interactions.
            k (int): Limits for the ranks to predict.
            test_interactions (DataFrame): Sparse matrix with the test user-item interactions.
            recommended_items (Optional[np.ndarray], optional): 
                If already calculated, can be used istead of inner calculations. Defaults to None.

        Returns:
            np.ndarray: Rank matrix
        """
        if recommended_items is None:
            recs = self.recommend_k(train_interactions, k, test_interactions)
        else:
            recs = recommended_items.copy()
        test_interactions = test_interactions.toPandas().sort_values('user_idx')
        filler = test_interactions['item_idx'].max()
        for i, user in tqdm(
            enumerate(test_interactions['user_idx'].unique()),
            desc='Calculating ranks',
            total=test_interactions['user_idx'].nunique()
        ):
            relevant_item = set(
                test_interactions[
                    test_interactions['user_idx'] == user
                ]['item_idx'].values
            )
            for j in range(recs.shape[1]):
                recs[i, j] = j if recs[i, j] in relevant_item else filler
            recs[i] = sorted(recs[i])
            recs[i, len(relevant_item):] = -1
        return recs

    def recommend_k(
        self, train_interactions: DataFrame, k: int, test_interactions: Optional[DataFrame] = None
    ) -> np.ndarray:
        """Get top-k recommended items for each user.

        Args:
            train_interactions (DataFrame): Sparse matrix with the train user-item interactions.
            k (int): Limits for the ranks to predict.
            test_interactions (DataFrame): Sparse matrix with the test user-item interactions.

        Returns:
            np.ndarray: Matrix with recommended items.
        """
        start_time = time.time()
        recs = self.model.predict(
            train_interactions,
            k,
            filter_seen_items=True,
            users=test_interactions.select('user_idx').distinct() \
                if test_interactions is not None else None
        ).toPandas()
        self.predict_time = time.time() - start_time
        logger.info('Finished predicting most similiar items')
        # Sort users
        recs.sort_values(
            ['user_idx', 'relevance'],
            inplace=True,
            ascending=[True, False]
        )

        recs = recs.groupby('user_idx').agg({'item_idx': lambda l: l.tolist()[:k]})

        # Add users which are not presented in the recs
        recs = test_interactions \
                .toPandas() \
                .sort_values('user_idx') \
                .drop_duplicates(subset=['user_idx']) \
                .drop('item_idx', axis=1) \
                .merge(recs, on='user_idx', how='left')['item_idx']

        # Filler for the empty values
        filler = test_interactions.agg({'item_idx': 'max'}).collect()[0]["max(item_idx)"] + 1

        recommend = np.ones((recs.shape[0], k), dtype=np.int32) * filler
        for i, rec in tqdm(
            enumerate(recs.values),
            desc="Filling items array",
            total=len(recs.values)
        ):
            if not isinstance(rec, float):
                recommend[i][:len(rec)] = np.array(rec)

        return recommend

    def get_items_and_ranks(
        self, train_interactions: DataFrame, k: int, test_interactions: DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate both predictions and ranks

        Args:
            train_interactions (DataFrame): Sparse matrix with the train user-item interactions.
            k (int): Limits for the ranks to predict.
            test_interactions (DataFrame): Sparse matrix with the test user-item interactions.

        Returns:
            tuple[np.ndarray, np.ndarray]: Item predictions and relevant ranks matricies.
        """
        items = self.recommend_k(train_interactions, k, test_interactions)
        ranks = self.get_relevant_ranks(train_interactions, k, test_interactions, items)
        return items, ranks
