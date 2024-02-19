"""Most Popular module"""
from pathlib import Path
import os
from omegaconf import DictConfig
import pandas as pd
import numpy as np

from src.models.baselines import RandomBaseline
from src.preprocessing import ClassicDataset
from src.utils.logging import get_logger
from src.utils.processing import save_results, train_test_split, pandas_to_sparse
from src.utils.metrics import run_all_metrics, coverage, novelty, diversity
from src.utils.processing import pandas_to_aggregate
from .base_runner import BaseRunner

logger = get_logger(name=__name__)


class BaselineRandomRunner(BaseRunner):
    """Most popular baseline runner."""

    @staticmethod
    def run(cfg: DictConfig) -> None:
        # load configs
        cfg_data = cfg["dataset"]
        cfg_model = cfg["library"]

        dataset = ClassicDataset()
        dataset.prepare(cfg_data)

        # split data into samples
        dataset_folder = Path(os.path.join("preproc_data", cfg_data["name"]))
        dataset = ClassicDataset()
        dataset.prepare(cfg_data)
        data_train, data_test = BaselineRandomRunner._load_or_split_data(
            cfg_data, dataset_folder, dataset
        )
        shape = BaselineRandomRunner._get_shape(data_train)
        train_interactions_sparse, _ = pandas_to_sparse(
            data_train,
            weighted=True,
            shape=shape,
        )

        model = RandomBaseline()
        model.fit(train_interactions_sparse)

        test_userids = np.sort(data_test.userId.unique())
        top_100_items = model.recommend_k(k=100, userids=test_userids)

        metrics_df = BaselineRandomRunner._calculate_metrics(
            top_100_items, data_test, train_interactions_sparse, model
        )

        save_results(
            (metrics_df, "results"),
            (top_100_items, "items"),
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
            dataset_folder.joinpath("train.parquet").exists()
            and dataset_folder.joinpath("test.parquet").exists()
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
        metrics = run_all_metrics(
            top_100_items, pandas_to_aggregate(data_test), [5, 10, 20, 100]
        )
        coverage_metrics, novelty_metrics, diversity_metrics = [], [], []
        for k in (5, 10, 20, 100):
            coverage_metrics.append(
                coverage(top_100_items, (train_interactions.getnnz(axis=0) > 0).sum(), k)
            )
            novelty_metrics.append(
                novelty(top_100_items.astype(np.int32), train_interactions.tocsc(), k)
            )
            diversity_metrics.append(
                diversity(top_100_items.astype(np.int32), train_interactions.tocsc(), k)
            )

        metrics_df = pd.DataFrame(
            metrics,
            index=[5, 10, 20, 100],
            columns=(
                "Precision@k",
                "Recall@K",
                "MAP@K",
                "nDCG@k",
                "MRR@k",
                "HitRate@k",
            ),
        )
        metrics_df["Coverage@K"] = coverage_metrics
        metrics_df["Novelty@K"] = novelty_metrics
        metrics_df["Diversity@k"] = diversity_metrics

        metrics_df["Time_fit"] = model.learning_time
        metrics_df["Time_predict"] = model.predict_time

        return metrics_df
