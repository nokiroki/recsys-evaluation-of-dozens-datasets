"""LightFM module"""
from pathlib import Path

from omegaconf import DictConfig

import numpy as np
import pandas as pd

from src.models.lightfm import LightFMBench
from src.preprocessing import ClassicDataset
from src.utils.processing import (
    save_results,
    train_test_split,
    pandas_to_aggregate,
    pandas_to_sparse
)
from src.utils.metrics import run_all_metrics, coverage, novelty, diversity
from .base_runner import BaseRunner


class LightFMRunner(BaseRunner):
    """LightFM model runner."""

    @staticmethod
    def run(cfg: DictConfig) -> None:
        # load configs
        cfg_data = cfg["dataset"]
        cfg_model = cfg["library"]

        dataset = ClassicDataset()
        dataset.prepare(cfg_data)

        dataset_folder = Path('/'.join(("preproc_data", cfg_data["name"])))

        data_train, data_test = LightFMRunner._load_or_split_data(cfg_data, dataset_folder, dataset)
        shape = LightFMRunner._get_shape(data_train)

        train_interactions_sparse, train_weights_sparse = pandas_to_sparse(
            data_train, weighted=True, shape=shape, sparse_type="coo"
        )

        model_folder = dataset_folder.joinpath(cfg_model["name"])

        if cfg_model["saved_model"]:
            lightfm = LightFMBench.initialize_saved_model(
                model_folder.joinpath(cfg_model["saved_model_name"])
            )
        else:
            if cfg_model["enable_optimization"]:
                data_train_opt, data_val = train_test_split(
                    data_train,
                    test_size=cfg_data["splitting"]["val_size"],
                    splitting_type=cfg_data["splitting"]["strategy"]
                )
                train_opt_interactions_sparse, train_opt_weights_sparse = pandas_to_sparse(
                    data_train_opt,
                    weighted=True,
                    shape=LightFMRunner._get_shape(data_train_opt),
                    sparse_type="coo"
                )
                lightfm = LightFMBench.initialize_with_optimization(
                    cfg_model["optuna_optimizer"],
                    cfg_model["learning"],
                    train_opt_interactions_sparse,
                    train_opt_weights_sparse,
                    data_val
                )
                was_optimized = True
            else:
                lightfm = LightFMBench.initialize_with_params(cfg_model["model"])
                was_optimized = False
            lightfm.fit(train_interactions_sparse, train_weights_sparse, **cfg_model["learning"])
            lightfm.save_model(model_folder)

        top_100_items = lightfm.recommend_k(userids=np.sort(data_test["userId"].unique()), k=100)
        metrics_df = LightFMRunner._calculate_metrics(
            top_100_items, data_test, train_interactions_sparse, lightfm
        )

        save_results(
            (metrics_df, f"results_wasOptimized_{was_optimized}"),
            (top_100_items, f"items_wasOptimized_{was_optimized}"),
            cfg["results_folder"],
            cfg_model["name"],
            cfg_data["name"]
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
    def _calculate_metrics(top_100_items, data_test, train_interactions, model):
        metrics = run_all_metrics(top_100_items, pandas_to_aggregate(data_test), [5, 10, 20, 100])
        coverage_metrics, novelty_metrics, diversity_metrics = [], [], []
        for k in (5, 10, 20, 100):
            coverage_metrics.append(coverage(top_100_items, (train_interactions.getnnz(axis=0) > 0).sum(), k))
            novelty_metrics.append(novelty(top_100_items.astype(np.int32), train_interactions.tocsc(), k))
            diversity_metrics.append(diversity(top_100_items.astype(np.int32), train_interactions.tocsc(), k))


        metrics_df = pd.DataFrame(metrics, index=[5, 10, 20, 100], columns=(
            "Precision@k", "Recall@K", "MAP@K", "nDCG@k", "MRR@k", "HitRate@k"
        ))
        metrics_df["Coverage@K"] = coverage_metrics
        metrics_df["Novelty@K"] = novelty_metrics
        metrics_df["Diversity@k"] = diversity_metrics

        metrics_df["Time_fit"] = model.learning_time
        metrics_df["Time_predict"] = model.predict_time

        return metrics_df
