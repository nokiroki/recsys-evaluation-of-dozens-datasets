"""LightFM module"""
from pathlib import Path

from omegaconf import DictConfig

import pandas as pd

from src.models.lightfm import LightFMBench
from src.preprocessing import ClassicDataset
from src.utils.processing import data_split, save_results
from src.utils.metrics import run_all_metrics, coverage
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

        # split data into samples
        (interactions_train, weights_train), \
        (interactions_val, weights_val), \
        (interactions_test, _) = data_split(
            dataset.prepared_data, cfg_data, sparse_type='coo'
        )

        interactions_train_val = (interactions_train + interactions_val).tocoo()
        if weights_train is not None and weights_val is not None:
            weights_train_val = (weights_train + weights_val).tocoo()
        else:
            weights_train_val = None

        model_folder = Path('/'.join(('preproc_data', cfg_data['name'], cfg_model['name'])))

        if cfg_model['saved_model']:
            lightfm = LightFMBench.initialize_saved_model(
                model_folder.joinpath(cfg_model['saved_model_name'])
            )
        else:
            lightfm = None
        if lightfm is None:
            if cfg_model['enable_optimization']:
                lightfm = LightFMBench.initialize_with_optimization(
                    cfg_model['optuna_optimizer'],
                    cfg_model['learning'],
                    interactions_train,
                    weights_train,
                    interactions_val
                )
                was_optimized = True
            else:
                lightfm = LightFMBench.initialize_with_params(cfg_model['model'])
                was_optimized = False
            lightfm.fit(interactions_train_val, weights_train_val, **cfg_model['learning'])
            lightfm.save_model(model_folder)

        ranks = lightfm.get_relevant_ranks(
            interactions_test, 
            interactions_train_val, 
            cfg_model["learning"]["num_threads"],
            cfg_model["learning"]["mp_threads"]
        )
        top_100_items = lightfm.recommend_k(
            interactions_train_val, 100, cfg_model["learning"]["num_threads"]
        )

        metrics = run_all_metrics(ranks, [5, 10, 20, 100])
        coverage_metrics = []
        for k in (5, 10, 20, 100):
            coverage_metrics.append(coverage(top_100_items, interactions_train_val.shape[1], k))

        metrics_df = pd.DataFrame(metrics, index=[5, 10, 20, 100], columns=(
            'Precision@k', 'Recall@K', 'MAP@K', 'nDCG@k', 'MRR@k', 'HitRate@k'
        ))
        metrics_df['Coverage@K'] = coverage_metrics

        metrics_df['Time_fit'] = lightfm.learning_time
        metrics_df['Time_predict'] = lightfm.predict_time

        save_results(
            (metrics_df, f"results_wasOptimized_{was_optimized}"),
            (top_100_items, f"items_wasOptimized_{was_optimized}"),
            (ranks, f"ranks_wasOptimized_{was_optimized}"),
            cfg["results_folder"],
            cfg_model["name"],
            cfg_data["name"]
        )
