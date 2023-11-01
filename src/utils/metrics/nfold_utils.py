from typing import Union, Optional
from pathlib import Path

import numpy as np

import scipy.sparse as sps

from src.utils.metrics import run_all_metrics_nfold
from src.utils.logging import get_logger

logger = get_logger(name=__name__)

def get_nfold_by_method(
    path: Path,
    save_path: Path,
    is_optimized: Optional[bool],
    train_interactions: sps.csc_matrix,
    k: Union[int, list[int]],
    n: int = 100,
    excluded_percentage: float = .2
) -> None:
    if is_optimized is None:
        items_opt = path.joinpath(f"items.npy")
        ranks_opt = path.joinpath(f"ranks.npy")
    else:
        items_opt = path.joinpath(f"items_wasOptimized_{str(is_optimized)}.npy")
        ranks_opt = path.joinpath(f"ranks_wasOptimized_{str(is_optimized)}.npy")
    if all((items_opt.exists(), ranks_opt.exists())):
        logger.info("Calculating nfolds for %s optimizer", is_optimized)
        run_all_metrics_nfold(
            np.load(ranks_opt).astype(np.int32),
            k,
            np.load(items_opt).astype(np.int32),
            train_interactions,
            n,
            excluded_percentage
        ).to_csv(save_path.joinpath(
            f"results_nfold_{n}_wasOptimized_{str(is_optimized)}.csv"
        ))
    else:
        logger.warning("%s optimizer results does not exist", is_optimized)
