"""RecBole module"""
from pathlib import Path
import os
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np

import shutil
import importlib

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

from src.models.recbole import RecboleBench, get_model_rec
from src.preprocessing import ClassicDataset
from src.utils.logging import get_logger
from src.utils.processing import (
    save_results,
    train_test_split,
    pandas_to_sparse,
    pandas_to_recbole,
)
from src.utils.metrics import run_all_metrics, coverage, novelty, diversity
from .base_runner import BaseRunner

logger = get_logger(name=__name__)


class RecboleRunner(BaseRunner):
    """RecBole model runner."""

    @staticmethod
    def run(cfg: DictConfig) -> None:
        # load configs
        recbole_name: str = cfg["library"]["name"]
        cfg_data = cfg["dataset"]
        cfg_model = cfg["library"]["recbole_model"]

        # split data into samples
        dataset_folder = Path(os.path.join("preproc_data", cfg_data["name"]))
        dataset = ClassicDataset()
        dataset.prepare(cfg_data)
        data_train, data_test = RecboleRunner._load_or_split_data(
            cfg_data, dataset_folder, dataset
        )

        shape = RecboleRunner._get_shape(data_train)

        train_interactions_sparse, _ = pandas_to_sparse(
            data_train, weighted=True, shape=shape, sparse_type="coo"
        )

        pandas_to_recbole(
            dataset=data_train, dataset_name=cfg_data["name"], split_name="train"
        )
        pandas_to_recbole(
            dataset=data_test, dataset_name=cfg_data["name"], split_name="test"
        )

        parameter_dict = OmegaConf.to_container(
            cfg_model["recbole_params"], resolve=True
        )
        parameter_dict.update(
            dict(
                data_path=os.path.join("data", "tmp"),
                dataset=cfg_data["name"],
            )
        )

        model = get_model_rec(cfg_model["name"])

        config = Config(
            model=model,
            dataset=cfg_data["name"],
            config_file_list=None,
            config_dict=parameter_dict,
        )

        init_seed(config["seed"], config["reproducibility"])

        dataset = create_dataset(config)
        train_set, _, test_set = data_preparation(config, dataset)

        model_folder = dataset_folder.joinpath(recbole_name, cfg_model["name"])

        if cfg_model["saved_model"]:
            bench = RecboleBench.initialize_saved_model(
                model_folder.joinpath(cfg_model["saved_model_name"]), train_set
            )
        else:
            bench = None

        if bench is None:
            if cfg_model["enable_optimization"]:
                data_train_opt, data_val = train_test_split(
                    data_train,
                    test_size=cfg_data["splitting"]["val_size"],
                    splitting_type=cfg_data["splitting"]["strategy"],
                )
                pandas_to_recbole(
                    dataset=data_train_opt,
                    dataset_name=cfg_data["name"],
                    split_name="train_opt",
                )
                pandas_to_recbole(
                    dataset=data_val, dataset_name=cfg_data["name"], split_name="valid"
                )
                config["benchmark_filename"] = ["train_opt", "valid", "valid"]
                dataset_opt = create_dataset(config)
                train_opt_loader, _, valid_loader = data_preparation(
                    config, dataset_opt
                )

                bench = RecboleBench.initialize_with_optimization(
                    cfg_model["optuna_optimizer"],
                    train_opt_loader,
                    valid_loader,
                )
                was_optimized = True
            else:
                bench = RecboleBench.initialize_with_params(
                    train_loader=train_set, model_params=cfg_model["model"]
                )
                was_optimized = False

        bench = RecboleBench.initialize_with_params(
            train_loader=train_set, model_params=bench.config.final_config_dict
        )
        bench.fit(train_set)
        bench.save_model(model_folder)

        top_100_items = bench.recommend_k(test_set, k=100)
        metrics_df = RecboleRunner._calculate_metrics(
            top_100_items, test_set, train_interactions_sparse, dataset, bench
        )

        save_results(
            (metrics_df, f"results_wasOptimized_{was_optimized}"),
            (top_100_items, f"items_wasOptimized_{was_optimized}"),
            cfg["results_folder"],
            cfg_model["name"],
            cfg_data["name"],
            recbole_name,
        )

        shutil.rmtree(os.path.join(parameter_dict["data_path"], cfg_data["name"]))

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
    def _calculate_metrics(
        top_100_items, data_test, train_interactions, dataset, recbole
    ):
        recbole_test_lists = np.array(
            [
                tensor.tolist()
                for tensor in data_test.uid2positive_item
                if tensor is not None
            ],
            dtype=object,
        )
        metrics = run_all_metrics(top_100_items, recbole_test_lists, [5, 10, 20, 100])
        coverage_metrics, novelty_metrics, diversity_metrics = [], [], []
        top_100_items = dataset.id2token(dataset.iid_field, top_100_items).astype(int)
        for k in (5, 10, 20, 100):
            coverage_metrics.append(
                coverage(top_100_items, (train_interactions.getnnz(axis=0) > 0).sum(), k)
            )
            novelty_metrics.append(
                novelty(top_100_items, train_interactions.tocsc(), k)
            )
            diversity_metrics.append(
                diversity(top_100_items, train_interactions.tocsc(), k)
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

        metrics_df["Time_fit"] = recbole.learning_time
        metrics_df["Time_predict"] = recbole.predict_time

        return metrics_df
