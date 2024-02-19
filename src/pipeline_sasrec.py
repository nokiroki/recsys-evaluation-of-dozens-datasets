from pathlib import Path

from omegaconf import DictConfig

import numpy as np
import pandas as pd
import torch

from src.preprocessing import ClassicDataset
from src.models.sasrec import SASRecBench
from src.utils.metrics import run_all_metrics, coverage, novelty, diversity
from src.utils.processing import (
    data_split,
    save_results,
    train_test_split,
    pandas_to_sparse,
    pandas_to_aggregate,
    pandas_to_recbole,
)
from src.utils.logging import get_logger
from .base_runner import BaseRunner

from hydra import compose, initialize
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from src.preprocessing.data_preporation import load_dataset

logger = get_logger(name=__name__)


class SasRecRunner(BaseRunner):
    @staticmethod
    def run(cfg: DictConfig) -> None:
        # get configs
        cfg_data = cfg["dataset"]
        cfg_model = cfg["library"]

        # pre-process raw ratings
        dataset = ClassicDataset()
        dataset.prepare(cfg_data)

        dataset_folder = Path('/'.join(("preproc_data", cfg_data["name"])))

        seed_everything(cfg_model['seed'], workers=True,)

        data_train, data_test = SasRecRunner._load_or_split_data(
            cfg_data, dataset_folder, dataset
        )

        shape = SasRecRunner._get_shape(data_train,)

        cfg_model.model.sasrec_params['item_num'] = int(shape[1] - 1)

        train_interactions_sparse, _ = pandas_to_sparse(
            data_train, weighted=True, shape=shape, sparse_type="coo",
        )

        model_folder = Path(
            "/".join(
                (
                    "preproc_data",
                    cfg_data["name"],
                    cfg["library"]["name"],
                    cfg_model["name"],
                )
            )
        )

        if cfg_model["saved_model"]:
            bench = SASRecBench.initialize_saved_model(
                model_folder.joinpath(cfg_model["saved_model_name"])
            )
        else:
            bench = None

        if bench is None:
            if cfg_model["enable_optimization"]:
                cfg_model.optuna_optimizer.hyperparameters_vary.const['item_num'] = int(shape[1] - 1)
                bench = SASRecBench.initialize_with_optimization(
                    cfg_model["optuna_optimizer"],
                    cfg_model,
                    data_train,
                    model_folder,
                )
                was_optimized = True
            else:
                bench = SASRecBench.initialize_with_params(cfg_model["model"])
                was_optimized = False

            bench.fit(data_train, **cfg_model["learning"])
            bench.save_model(
                model_folder.joinpath(cfg_model["saved_model_name"])
            )

        top_100_items = bench.recommend_k(data_train, 100)
        metrics_df = SasRecRunner._calculate_metrics(top_100_items, data_test, train_interactions_sparse, bench, cfg_data)

        save_results(
            (metrics_df, f"results_wasOptimized_{was_optimized}"),
            (top_100_items, f"items_wasOptimized_{was_optimized}"),
            cfg["results_folder"],
            cfg_model["name"],
            cfg_data["name"],
        )

    @staticmethod
    def _get_shape(dataset):
        return (
            dataset["userId"].max() + 1,
            dataset["itemId"].max() + 1,
        )

    @staticmethod
    def _load_or_split_data(cfg_data, dataset_folder, dataset):
        if (
            dataset_folder.joinpath("train.parquet").exists() and
            dataset_folder.joinpath("test.parquet").exists()
        ):
            data_train = pd.read_parquet(dataset_folder.joinpath("train.parquet"))
            data_test = pd.read_parquet(dataset_folder.joinpath("test.parquet"))
        else:
            data_train, data_test = train_test_split(
                dataset.prepared_data,
                test_size=cfg_data["splitting"]["test_size"],
                splitting_type=cfg_data["splitting"]["strategy"],
            )
            data_train.to_parquet(dataset_folder.joinpath("train.parquet"))
            data_test.to_parquet(dataset_folder.joinpath("test.parquet"))
        return data_train, data_test

    @staticmethod
    def _calculate_metrics(top_100_items, data_test, train_interactions, model, cfg_data):
        metrics = run_all_metrics(top_100_items, pandas_to_aggregate(data_test,), [5, 10, 20, 100])
        coverage_metrics, novelty_metrics, diversity_metrics = [], [], []
        for k in (5, 10, 20, 100):
            coverage_metrics.append(coverage(top_100_items, (train_interactions.getnnz(axis=0) > 0).sum(), k))
            novelty_metrics.append(novelty(top_100_items.astype(np.int32), train_interactions.tocsc(), k))
            diversity_metrics.append(diversity(top_100_items.astype(np.int32), train_interactions.tocsc(), k))


        metrics_df = pd.DataFrame(metrics, index=[5, 10, 20, 100], columns=(
            "Precision@k", "Recall@K", "MAP@K", "nDCG@k", "MRR@k", "HitRate@k",
        ))
        metrics_df["Coverage@K"] = coverage_metrics
        metrics_df["Novelty@K"] = novelty_metrics
        metrics_df["Diversity@k"] = diversity_metrics

        metrics_df["Time_fit"] = model.learning_time
        metrics_df["Time_predict"] = model.predict_time

        return metrics_df
