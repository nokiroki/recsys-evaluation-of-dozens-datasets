"""Implicit module"""
from pathlib import Path

from omegaconf import DictConfig

import pandas as pd

from src.models.implicit import ImplicitBench
from src.preprocessing import ClassicDataset
from src.utils.logging import get_logger
from src.utils.processing import save_results, train_test_split, pandas_to_sparse
from src.utils.metrics import run_all_metrics, coverage
from .base_runner import BaseRunner

logger = get_logger(name=__name__)


class ImplicitRunner(BaseRunner):
    """Implict model runner."""

    @staticmethod
    def run(cfg: DictConfig) -> None:
        # load configs
        implicit_name: str = cfg['library']['name']
        cfg_data = cfg['dataset']
        cfg_model = cfg['library']['implicit_model']
        column_names = {
            "user_col": cfg_data["user_column"],
            "item_col": cfg_data["item_column"],
            "date_col": cfg_data["date_column"],
            "rating_col": cfg_data["rating_column"]
        }

        # split data into samples
        dataset_folder = Path('/'.join(("preproc_data", cfg_data["name"])))

        dataset = ClassicDataset()
        dataset.prepare(cfg_data)
        shape = (
            dataset.prepared_data[column_names["user_col"]].nunique(),
            dataset.prepared_data[column_names["item_col"]].nunique()
        )

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
                user_col=column_names["user_col"],
                item_col=column_names["item_col"],
                date_col=column_names["date_col"]
            )
            data_train.to_parquet(dataset_folder.joinpath("train.parquet"))
            data_test.to_parquet(dataset_folder.joinpath("test.parquet"))

        train_interactions_sparse, train_weights_sparse = pandas_to_sparse(
            data_train,
            weighted=True,
            shape=shape,
            user_col=column_names["user_col"],
            item_col=column_names["item_col"],
            rating_col=column_names["rating_col"]
        )
        test_interactions_sparse, _ = pandas_to_sparse(
            data_test,
            weighted=False,
            shape=shape,
            user_col=column_names["user_col"],
            item_col=column_names["item_col"],
            rating_col=column_names["rating_col"]
        )

        model_folder = dataset_folder.joinpath(implicit_name, cfg_model["name"])

        if cfg_model["saved_model"]:
            implicit = ImplicitBench.initialize_saved_model(
                model_folder.joinpath(cfg_model["saved_model_name"])
            )
        else:
            implicit = None
        if implicit is None:
            if cfg_model["enable_optimization"]:
                data_train_opt, data_val = train_test_split(
                    data_train,
                    test_size=cfg_data["splitting"]["val_size"],
                    splitting_type=cfg_data["splitting"]["strategy"],
                    user_col=column_names["user_col"],
                    item_col=column_names["item_col"],
                    date_col=column_names["date_col"]
                )
                train_opt_interactions_sparse, train_opt_weights_sparse = pandas_to_sparse(
                    data_train_opt,
                    weighted=True,
                    shape=shape,
                    user_col=column_names["user_col"],
                    item_col=column_names["item_col"],
                    rating_col=column_names["rating_col"]
                )
                val_interactions_sparse, _ = pandas_to_sparse(
                    data_val,
                    weighted=False,
                    shape=shape,
                    user_col=column_names["user_col"],
                    item_col=column_names["item_col"],
                    rating_col=column_names["rating_col"]
                )
                implicit = ImplicitBench.initialize_with_optimization(
                    cfg_model["name"],
                    cfg_model["optuna_optimizer"],
                    train_opt_interactions_sparse,
                    train_opt_weights_sparse,
                    val_interactions_sparse
                )
                was_optimized = True
            else:
                implicit = ImplicitBench.initialize_with_params(
                    cfg_model["name"], cfg_model["model"]
                )
                was_optimized = False
            implicit.fit(train_interactions_sparse, train_weights_sparse, **cfg_model['learning'])
            implicit.save_model(model_folder)

        ranks = implicit.get_relevant_ranks(test_interactions_sparse)
        top_100_items = implicit.recommend_k(k=100)

        metrics = run_all_metrics(ranks, [5, 10, 20, 100])
        coverage_metrics = []
        for k in (5, 10, 20, 100):
            coverage_metrics.append(coverage(top_100_items, train_interactions_sparse.shape[1], k))

        metrics_df = pd.DataFrame(metrics, index=[5, 10, 20, 100], columns=(
            'Precision@k', 'Recall@K', 'MAP@K', 'nDCG@k', 'MRR@k', 'HitRate@k'
        ))
        metrics_df['Coverage@K'] = coverage_metrics

        metrics_df['Time_fit'] = implicit.learning_time
        metrics_df['Time_predict'] = implicit.predict_time

        save_results(
            (metrics_df, f"results_wasOptimized_{was_optimized}"),
            (top_100_items, f"items_wasOptimized_{was_optimized}"),
            (ranks, f"ranks_wasOptimized_{was_optimized}"),
            cfg["results_folder"],
            cfg_model["name"],
            cfg_data["name"],
            implicit_name
        )
