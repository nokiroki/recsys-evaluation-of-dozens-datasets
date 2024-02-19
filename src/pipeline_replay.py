"""RecPlay module"""
from pathlib import Path
import os
from math import floor

import psutil

from omegaconf import DictConfig

import pandas as pd

from replay.utils import State
#from pyspark.sql import SparkSession

#from src.models.replay import RePlayBench
from src.preprocessing import ClassicDataset
from src.utils.logging import get_logger
#from src.utils.processing import replay_data_split, save_results
from src.utils.metrics import run_all_metrics, coverage
from .base_runner import BaseRunner

logger = get_logger(name=__name__)


class RePlayRunner(BaseRunner):
    """RecPlay model runner."""
    @staticmethod
    def run(cfg: DictConfig) -> None:
        # load configs
        replay_name: str = cfg["library"]["name"]
        cfg_data = cfg["dataset"]
        cfg_model = cfg["library"]["replay_model"]

        spark_memory = floor(psutil.virtual_memory().total / 1024**3 * 0.7)
        driver_memory = f"{spark_memory}g"

        shuffle_partitions = os.cpu_count() * 3
        user_home = os.environ["HOME"]

        session = (
            SparkSession.builder.config("spark.driver.memory", driver_memory)
                .config(
                    "spark.driver.extraJavaOptions",
                    "-Dio.netty.tryReflectionSetAccessible=true",
                )
                .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
                .config("spark.local.dir", os.path.join(user_home, "tmp"))
                .config("spark.driver.maxResultSize", "4g")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .config("spark.driver.host", "localhost")
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .config("spark.kryoserializer.buffer.max", "256m")
                # .config("spark.worker.cleanup.enabled", "true")
                # .config("spark.worker.cleanup.interval", "5")
                # .config("spark.worker.cleanup.appDataTtl", "5")
                .master("local[*]")
                .enableHiveSupport()
                .getOrCreate()
        )

        spark = State(session).session
        spark.sparkContext.setLogLevel("ERROR")

        dataset = ClassicDataset()
        dataset.prepare(cfg_data)

        interactions_train_val, \
        interactions_train, \
        interactions_val, \
        interactions_test = replay_data_split(dataset.prepared_data, cfg_data)

        model_folder = Path("/".join(
            ("preproc_data", cfg_data["name"], replay_name, cfg_model["name"])
        ))

        replay_model = None
        if cfg_model["saved_model"]:
            replay_model = RePlayBench.initialize_saved_model(
                model_folder.joinpath(cfg_model["saved_model_name"])
            )
        if replay_model is None:
            if cfg_model["enable_optimization"]:
                replay_model = RePlayBench.initialize_with_optimization(
                    cfg_model["name"],
                    cfg_model["optuna_optimizer"],
                    interactions_train,
                    interactions_val
                )
                was_optimized = True
            else:
                replay_model = RePlayBench.initialize_with_params(
                    cfg_model["name"], cfg_model["model"]
                )
                was_optimized = False

            replay_model.fit(interactions_train_val)
            replay_model.save_model(model_folder)

        items, ranks = replay_model.get_items_and_ranks(
            interactions_train_val, 100, interactions_test
        )

        metrics = run_all_metrics(ranks, [5, 10, 20, 100])
        coverage_metrics = []
        for k in (5, 10, 20, 100):
            coverage_metrics.append(coverage(
                items,
                interactions_train_val.select("item_idx").distinct().count(),
                k
            ))

        metrics_df = pd.DataFrame(metrics, index=[5, 10, 20, 100], columns=(
            "Precision@k", "Recall@K", "MAP@K", "nDCG@k", "MRR@k", "HitRate@k"
        ))
        metrics_df["Coverage@K"] = coverage_metrics
        metrics_df["Time_fit"] = replay_model.learning_time
        metrics_df["Time_predict"] = replay_model.predict_time

        save_results(
            (metrics_df, f"results_wasOptimized_{was_optimized}"),
            (items, f"items_wasOptimized_{was_optimized}"),
            (ranks, f"ranks_wasOptimized_{was_optimized}"),
            cfg["results_folder"],
            cfg_model["name"],
            cfg_data["name"],
            replay_name
        )

        spark.stop()
