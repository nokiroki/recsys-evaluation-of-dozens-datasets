"""Surprise module"""
from pathlib import Path

from omegaconf import DictConfig

import pandas as pd

from src.models.surprise import SurpriseBench
from src.preprocessing import ClassicDataset
from src.utils.processing import data_split, save_results
from src.utils.metrics import run_all_metrics, coverage
from .base_runner import BaseRunner


class SurpriseRunner(BaseRunner):
    """RecPlay model runner."""

    @staticmethod
    def run(cfg: DictConfig) -> None:
        # load configs
        cfg_data = cfg['dataset']
        cfg_model = cfg['library']['surprise_model']

        dataset = ClassicDataset()
        dataset.prepare(cfg_data)

        # split data into samples
        (interactions_train, _), \
        (interactions_val, _), \
        (interactions_test, _) = data_split(
            dataset.prepared_data, cfg_data, sparse_type='coo'
        )

        interactions_train_val = (interactions_train + interactions_val).tocoo()


        model_folder = Path('/'.join(('preproc_data', cfg_data['name'], cfg_model['name'])))

        if cfg_model['saved_model']:
            surprise_model = SurpriseBench.initialize_saved_model(
                model_folder.joinpath(cfg_model['saved_model_name'])
            )
            was_optimized = False
        else:
            surprise_model = None
        if surprise_model is None:
            if cfg_model['enable_optimization']:
                surprise_model = SurpriseBench.initialize_with_optimization(
                    cfg_model['optuna_optimizer'],
                    cfg_model,
                    interactions_train,
                    interactions_val
                )
                was_optimized = True
            else:
                surprise_model = SurpriseBench.initialize_with_params(cfg_model['name'], \
                                                                      cfg_model['model'],)
                was_optimized = False
            surprise_model.fit(interactions_train_val)
            surprise_model.save_model(model_folder)

        items, ranks = surprise_model.get_items_and_ranks(100, interactions_test)

        metrics = run_all_metrics(ranks, [5, 10, 20, 100])
        coverage_metrics = []
        for k in (5, 10, 20, 100):
            coverage_metrics.append(coverage(items, interactions_train_val.shape[1], k,))

        metrics_df = pd.DataFrame(metrics, index=[5, 10, 20, 100], columns=(
            'Precision@k', 'Recall@K', 'MAP@K', 'nDCG@k', 'MRR@k', 'HitRate@k'
        ))
        metrics_df['Coverage@K'] = coverage_metrics

        metrics_df['Time_fit'] = surprise_model.learning_time
        metrics_df['Time_predict'] = surprise_model.predict_time

        save_results(
            (metrics_df, f"results_wasOptimized_{was_optimized}"),
            (items, f"items_wasOptimized_{was_optimized}"),
            (ranks, f"ranks_wasOptimized_{was_optimized}"),
            cfg["results_folder"],
            cfg_model["name"],
            cfg_data["name"]
        )
