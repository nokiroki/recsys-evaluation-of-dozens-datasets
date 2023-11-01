"""Most Popular module"""
from omegaconf import DictConfig

import pandas as pd

from src.models.baselines import MostPopularBaseline
from src.preprocessing import ClassicDataset
from src.utils.logging import get_logger
from src.utils.processing import data_split, save_results
from src.utils.metrics import run_all_metrics, coverage
from .base_runner import BaseRunner

logger = get_logger(name=__name__)


class BaselineMostpopRunner(BaseRunner):
    """Most popular baseline runner."""

    @staticmethod
    def run(cfg: DictConfig) -> None:
        # load configs
        cfg_data = cfg['dataset']
        cfg_model = cfg['library']

        dataset = ClassicDataset()
        dataset.prepare(cfg_data)

        # split data into samples
        (interactions_train, _), \
        (interactions_val, _), \
        (interactions_test, _) = data_split(dataset.prepared_data, cfg_data)

        interactions_train_val = interactions_train + interactions_val

        model = MostPopularBaseline()
        model.fit(interactions_train_val)

        ranks = model.get_relevant_ranks(interactions_test)
        top_100_items = model.recommend_k(interactions_train_val, 100)

        metrics = run_all_metrics(ranks, [5, 10, 20, 100])
        coverage_metrics = []
        for k in (5, 10, 20, 100):
            coverage_metrics.append(coverage(top_100_items, interactions_train_val.shape[1], k))

        metrics_df = pd.DataFrame(metrics, index=[5, 10, 20, 100], columns=(
            'Precision@k', 'Recall@K', 'MAP@K', 'nDCG@k', 'MRR@k', 'HitRate@k'
        ))
        metrics_df['Coverage@K'] = coverage_metrics

        save_results(
            (metrics_df, "results"),
            (top_100_items, "items"),
            (ranks, "ranks"),
            cfg["results_folder"],
            cfg_model["name"],
            cfg_data["name"]
        )
