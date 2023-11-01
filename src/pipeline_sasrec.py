from pathlib import Path

from omegaconf import DictConfig

import numpy as np
import pandas as pd
import torch

from src.preprocessing import ClassicDataset
from src.models.sasrec import SASRecAdpBench
from src.utils.metrics import run_all_metrics, coverage
from src.utils.processing import data_split, save_results
from src.utils.logging import get_logger
from .base_runner import BaseRunner

from hydra import compose, initialize
from omegaconf import OmegaConf

from src.preprocessing.data_preporation import load_dataset

logger = get_logger(name=__name__)


class SasRecRunner(BaseRunner):
    @staticmethod
    def run(cfg: DictConfig) -> None:
        # get configs
        cfg_data = cfg["dataset"]
        cfg_model = cfg["library"]["msrec_model"]

        # extract raw ratings
        if not Path(cfg_data["data_src"], cfg_data["ratings_file"]).exists():
            logger.info("Raw file does not exist, downloading...")
            load_dataset(cfg_data)

        item_sequences = pd.read_parquet(
            Path(cfg_data["data_src"], cfg_data["ratings_file"])
        )

        # pre-process raw ratings
        dataset = ClassicDataset()
        dataset.prepare(cfg_data)

        # add mandatory values
        cfg_model.model["user_num"] = max(
            dataset.prepared_data[cfg_data["user_column"]]
        )
        cfg_model.model["item_num"] = max(
            dataset.prepared_data[cfg_data["item_column"]]
        )

        # split data into samples
        train_set, \
        val_set, \
        test_set = data_split(
            dataset.prepared_data, cfg_data, return_format='pandas'
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
            sasrec_bench = SASRecAdpBench.initialize_saved_model(
                model_folder.joinpath(cfg_model["saved_model_name"])
            )
        else:
            sasrec_bench = None

        logger.debug(
            f"MIN DATA IDX: {min(dataset.prepared_data[cfg_data['item_column']])}"
        )

        if sasrec_bench is None:
            if cfg_model["enable_optimization"]:
                sasrec_bench = SASRecAdpBench.initialize_with_optimization(
                    cfg_model["optuna_optimizer"],
                    cfg_model,
                    train_set,
                    val_set,
                    model_folder,
                )
                was_optimized = True
            else:
                sasrec_bench = SASRecAdpBench.initialize_with_params(cfg_model["model"])
                was_optimized = False

            sasrec_bench.fit(train_set, val_set, **cfg_model["learning"])
            sasrec_bench.save_model(
                model_folder.joinpath(cfg_model["saved_model_name"])
            )

        ranks = sasrec_bench.get_relevant_ranks(
            test_set,
            test=True,
        )
        top_100_items = sasrec_bench.recommend_k(test_set, True, 100)

        metrics = run_all_metrics(ranks, [5, 10, 20, 100])
        coverage_metrics = []
        for k in (5, 10, 20, 100):
            coverage_metrics.append(
                coverage(
                    top_100_items,
                    len(np.unique(test_set[cfg_data["item_column"]])),
                    k,
                )
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

        metrics_df["Time_fit"] = sasrec_bench.learning_time
        metrics_df["Time_predict"] = sasrec_bench.predict_time

        save_results(
            (metrics_df, f"results_wasOptimized_{was_optimized}"),
            (top_100_items, f"items_wasOptimized_{was_optimized}"),
            (ranks, f"ranks_wasOptimized_{was_optimized}"),
            cfg["results_folder"],
            cfg_model["name"],
            cfg_data["name"]
        )
