"""RecBole module"""
from pathlib import Path
import os

from omegaconf import DictConfig

import omegaconf
import pandas as pd
import shutil

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

from src.models.recbole import RecboleBench
from src.preprocessing import ClassicDataset
from src.utils.logging import get_logger
from src.utils.processing import data_split, save_results
from src.utils.metrics import run_all_metrics, coverage
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

        dataset = ClassicDataset()
        dataset.prepare(cfg_data)

        data_split(
            dataset.prepared_data,
            cfg_data,
            cfg_model,
            return_format="recbole",
        )

        parameter_dict = omegaconf.OmegaConf.to_container(cfg_model["recbole_params"])
        parameter_dict["data_path"] = os.path.join("data", "tmp")
        parameter_dict["dataset"] = cfg_data["name"]

        config = Config(
            model=cfg_model["name"],
            dataset=cfg_data["name"],
            config_file_list=None,
            config_dict=parameter_dict,
        )

        init_seed(config["seed"], config["reproducibility"])

        dataset = create_dataset(config)

        train_set, valid_set, test_set = data_preparation(config, dataset)

        model_folder = Path(
            "/".join(
                ("preproc_data", cfg_data["name"], recbole_name, cfg_model["name"])
            )
        )

        if cfg_model["saved_model"]:
            bench = RecboleBench.initialize_saved_model(
                model_folder.joinpath(cfg_model["saved_model_name"]), train_set
            )
        else:
            bench = None

        if bench is None:
            if cfg_model["enable_optimization"]:
                bench = RecboleBench.initialize_with_optimization(
                    cfg_model["optuna_optimizer"],
                    train_set,
                    valid_set,
                )
                was_optimized = True
            else:
                bench = RecboleBench.initialize_with_params(
                    train_loader=train_set,
                    model_params=cfg_model['model']
                )
                was_optimized = False

        # load trainval to make refit
        config["benchmark_filename"] = ["trainval", "val", "test"]
        dataset = create_dataset(config)
        trainval_set, valid_set, test_set = data_preparation(config, dataset)
        bench = RecboleBench.initialize_with_params(
            train_loader=trainval_set, model_params=bench.config.final_config_dict
        )
        bench.fit(trainval_set)
        bench.save_model(model_folder)

        ranks = bench.get_relevant_ranks(test_set)
        top_100_items = bench.recommend_k(test_set, 100)

        metrics = run_all_metrics(ranks, [5, 10, 20, 100])
        coverage_metrics = []
        for k in (5, 10, 20, 100):
            coverage_metrics.append(
                coverage(top_100_items, test_set.dataset.item_num, k)
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

        metrics_df["Time_fit"] = bench.learning_time
        metrics_df["Time_predict"] = bench.predict_time

        save_results(
            (metrics_df, f"results_wasOptimized_{was_optimized}"),
            (top_100_items, f"items_wasOptimized_{was_optimized}"),
            (ranks, f"ranks_wasOptimized_{was_optimized}"),
            cfg["results_folder"],
            cfg_model["name"],
            cfg_data["name"],
            recbole_name,
        )

        # shutil.rmtree(os.path.join(parameter_dict["data_path"],cfg_data["name"]))
