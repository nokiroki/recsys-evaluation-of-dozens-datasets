import logging
from typing import Optional
import warnings
from pathlib import Path

import numpy as np

np.float = float

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils.aggregation import filter_data, aggregate_methods
from src.compare_methods import run_CD, run_dolan_more, bayes_scores


@hydra.main(version_base=None, config_path="config", config_name="aggregation")
def main(cfg: Optional[DictConfig] = None) -> None:
    key_metric: str = cfg["key_metric"]
    key_k: int = cfg["key_k"]
    metric_path: Path = Path(cfg["metric_path"])
    methods_with_subfolders: list = cfg["methods_with_subfolders"]
    drop_minor_datasets: bool = cfg["drop_minor_datasets"]
    drop_minor_methods: bool = cfg["drop_minor_methods"]
    filter_datasets: list = cfg["filter_datasets"]
    save_path: Path = Path(cfg["save_path"])

    aggregation_method: str = cfg["aggregation_method"]

    metrics_df = aggregate_methods(
        metric_path,
        filter_datasets=filter_datasets,
        methods_with_subfolders=methods_with_subfolders,
    )

    metrics_df = filter_data(
        metrics_df,
        metric_save=key_metric,
        k_save=key_k,
        drop_minor_datasets=drop_minor_datasets,
        drop_minor_methods=drop_minor_methods,
    )

    metrics_df.to_csv(save_path.joinpath(f"metrics_{key_metric}_{key_k}.csv"), index=False)

    if "cd" in aggregation_method:
        results = run_CD(metrics_df, save_image=True, image_path=save_path)
        results.to_csv(save_path.joinpath(f"results_cd_{key_metric}_{key_k}.csv"))
    if "dm" in aggregation_method:
        results = run_dolan_more(metrics_df, image_path=save_path, image_format="png")
        results.to_csv(save_path.joinpath(f"results_dolan_more_{key_metric}_{key_k}.csv"))
    if "bayes" in aggregation_method:
        results = bayes_scores(metrics_df)
        results.to_csv(save_path.joinpath(f"results_bayes_{key_metric}_{key_k}.csv"))


if __name__ == "__main__":
    main()
