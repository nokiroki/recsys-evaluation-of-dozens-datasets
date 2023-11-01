"""Implicit module"""
from pathlib import Path

from omegaconf import DictConfig

import pandas as pd
import numpy as np

from src.models.implicit import ImplicitBench
from src.preprocessing import ClassicDataset
from src.utils.logging import get_logger
from src.utils.processing import data_split, save_results
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

        dataset = ClassicDataset()
        dataset.prepare(cfg_data)

        # split data into samples
        (interactions_train, weights_train), \
        (interactions_val, weights_val), \
        (interactions_test, _) = data_split(
            dataset.prepared_data, cfg_data
        )

        interactions_train_val = interactions_train + interactions_val
        if weights_train is not None and weights_val is not None:
            weights_train_val = weights_train + weights_val
        else:
            weights_train_val = None

        model_folder = Path('/'.join((
            'preproc_data', cfg_data['name'], implicit_name, cfg_model['name'])
        ))

        if cfg_model['saved_model']:
            implicit = ImplicitBench.initialize_saved_model(
                model_folder.joinpath(cfg_model['saved_model_name'])
            )
        else:
            implicit = None
        if implicit is None:
            if cfg_model['enable_optimization']:
                implicit = ImplicitBench.initialize_with_optimization(
                    cfg_model['name'],
                    cfg_model['optuna_optimizer'],
                    interactions_train,
                    weights_train,
                    interactions_val
                )
                was_optimized = True
            else:
                implicit = ImplicitBench.initialize_with_params(
                    cfg_model['name'],
                    cfg_model['model']
                )
                was_optimized = False
            implicit.fit(interactions_train_val, weights_train_val, **cfg_model['learning'])
            implicit.save_model(model_folder)

        ranks = implicit.get_relevant_ranks(interactions_test, interactions_train_val)
        top_100_items = implicit.recommend_k(
            interactions_train_val[np.ediff1d(interactions_train_val.indptr) > 0],
            100
            )

        metrics = run_all_metrics(ranks, [5, 10, 20, 100])
        coverage_metrics = []
        for k in (5, 10, 20, 100):
            coverage_metrics.append(coverage(top_100_items, interactions_train_val.shape[1], k))

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
