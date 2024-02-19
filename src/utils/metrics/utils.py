"""Metrics for recommensation system evaluation"""
from typing import Union, Optional
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import comb
import scipy.sparse as sps

from tqdm.auto import tqdm

from src.utils.logging import get_logger

logger = get_logger(name=__name__)


def mean_precision(ranks: np.ndarray, k: int) -> float:
    """Precision@k metric. Calculates mean precision over all users.

    Args:
        ranks (np.ndarray): True relative items' ranks
        k (int): Number of the top predicted items

    Returns:
        float: Metric's value
    """
    ranks_good = (ranks < k) & (ranks != -1)
    k_shape = np.clip((ranks != -1).sum(1), a_min=None, a_max=k)
    return np.mean(ranks_good.sum(1) / k_shape)


def mean_recall(ranks: np.ndarray, k: int) -> float:
    """Recall@k. Calculates mean recall over all users.

    Args:
        ranks (np.ndarray): True relative items' ranks
        k (int): Number of the top predicted items

    Returns:
        float: Metric's value
    """
    relevants = (ranks != -1).sum(1)
    ranks = (ranks < k) & (ranks != -1)
    ranks = ranks.sum(1)

    return np.mean(ranks / relevants)


def mean_average_precison(ranks: np.ndarray, k: int) -> float:
    """MAP@k metric. Calculates mean precision with respect to the item position.

    Args:
        ranks (np.ndarray): True relative items' ranks
        k (int): Number of the top predicted items

    Returns:
        float: Metric's value
    """
    ranks = ranks.astype(np.float32)
    ranks[ranks == -1] = np.inf
    ranks.sort(axis=1)
    ranks[ranks == np.inf] = -2
    ranks = ranks.astype(np.int32)

    mask = (ranks < k) & (ranks != -2)
    k_shape = np.clip((ranks != -2).sum(1), a_min=None, a_max=k)
    order_array = np.repeat(
        np.arange(ranks.shape[1])[np.newaxis, :] + 1, ranks.shape[0], axis=0
    )
    aranks = order_array / (ranks + 1)

    return np.mean((aranks * mask).sum(1) / k_shape)


def normalized_discounted_cumulative_gain(ranks: np.ndarray, k: int) -> float:
    """nDCG@k metric. Calculate mean discounted cumulative gain over all users.

    Args:
        ranks (np.ndarray): True relative items' ranks
        k (int): Number of the top predicted items

    Returns:
        float: Metric's value
    """
    order_array = np.repeat(
        np.arange(ranks.shape[1])[np.newaxis, :] + 1, ranks.shape[0], axis=0
    )
    mask_ideal = (ranks != -1) & (order_array <= k)
    ideal_dcg = ((1 / np.log2(order_array + 1)) * mask_ideal).sum(1)

    mask = (ranks < k) & (ranks != -1)
    log = np.log2(ranks + 2)
    log[log == 0] = -1

    dcg = ((1 / log) * mask).sum(1)

    return np.mean(dcg / ideal_dcg)


def hit_rate(ranks: np.ndarray, k: int) -> float:
    """HitRate@k metric. Calculates mean number of hits over all users.

    Args:
        ranks (np.ndarray): True relative items' ranks
        k (int): Number of the top predicted items

    Returns:
        float: Metric's value
    """
    ranks = ranks.astype(np.float32)
    ranks[ranks == -1] = np.inf
    min_hit = ranks.min(1)
    min_hit = min_hit < k
    return min_hit.mean()


def coverage(ranks: np.ndarray, total_items: int, k: int) -> float:
    """Covarage@k metric. Calculate percent of the k-th unique predicted items
    over all other items.

    Args:
        ranks (np.ndarray): True relative items' rank
        total_items (int): Total amount of the items
        k (int): Number of the top predicted items

    Returns:
        float: Metric's value
    """
    num_unique = np.unique(ranks[:, :k][ranks[:, :k] >= 0]).shape[0]
    return num_unique / total_items


def mean_reciprocal_rank(ranks: np.ndarray, k: int) -> float:
    """MRR@k metric. Calculates mean reciprocal rank over all users

    Args:
        ranks (np.ndarray): True relative items' ranks
        k (int): Number of the top predicted items

    Returns:
        float: Metric's value
    """
    ranks = ranks.astype(np.float32)
    ranks[ranks == -1] = np.inf
    reciprocal_rank = ranks.min(1) + 1
    reciprocal_rank = 1 / reciprocal_rank
    reciprocal_rank[reciprocal_rank < 1 / k] = 0
    return reciprocal_rank.mean()


def cosin_similarity(
    interactions: pd.DataFrame, i: int, j: int, item_col: str, user_col: str
) -> float:
    """Additional method for calculating cosine similarity. Cached version

    Args:
        interactions (pd.DataFrame): Interactions on which we need to calculate the similarity.
        i (int): First column.
        j (int): Second column.
        item_col (str): Item column name.
        user_col (str): User column name.

    Returns:
        float: Cosin Similarity value
    """

    users_i = interactions[interactions[item_col] == i][user_col].unique()
    users_j = interactions[interactions[item_col] == j][user_col].unique()
    numerator = len(np.intersect1d(users_i, users_j))
    denominator = np.sqrt(len(users_i) * len(users_j))
    if denominator == 0:
        return 0
    return numerator / denominator


def diversity(
    predicted_items: np.ndarray,
    interactions: sps.csc_matrix,
    k: int,
    propogated_dict: Optional[dict[tuple[int, int], float]] = None
) -> Union[float, tuple[dict[tuple[int, int], float], float]]:
    predicted_items = predicted_items[:, :k].astype(np.int32)
    if propogated_dict is None:
        cosin_sim_history = {}
    else:
        cosin_sim_history = propogated_dict.copy()
    predicted_items.sort(1)
    total_cos_sim = 0
    for item_ids in tqdm(predicted_items, desc="Diversity calculation", leave=False):
        for i, j in combinations(item_ids, 2):
            if (i, j) in cosin_sim_history:
                total_cos_sim += cosin_sim_history[(i, j)]
            else:
                col1 = interactions[:, i]
                col2 = interactions[:, j]
                cos_sim = col1.T.dot(col2).sum() / np.sqrt(col1.sum() * col2.sum())
                cosin_sim_history[(i, j)] = cos_sim
                total_cos_sim += cos_sim

    answer = 1 - total_cos_sim / (predicted_items.shape[0] * comb(k, 2))

    if propogated_dict is None:
        return answer
    return answer, cosin_sim_history


def cached_divercity(
    predicted_items: np.ndarray,
    idx: np.ndarray,
    interactions: sps.csc_matrix,
    k: int,
    propogated_dict: Optional[dict[tuple[int, int], float]],
    userwise_dict: Optional[dict[int, float]] = None
) -> tuple[float, dict[tuple[int, int], float], dict[int, float]]:
    predicted_items = predicted_items[:, :k].astype(np.int32)
    cosin_sim_history = propogated_dict.copy()

    if userwise_dict is None:
        user_history = {}
        flag = False
    else:
        user_history = userwise_dict.copy()
        flag = True

    predicted_items.sort(1)
    total_cos_sim = 0
    for index, item_ids in tqdm(
        zip(idx, predicted_items),
        desc="Diversity calculation",
        leave=False,
        total=idx.shape[0]
    ):
        if flag and index in user_history:
            total_cos_sim += user_history[index]
        else:
            user_cos_sim = 0
            for i, j in combinations(item_ids, 2):
                if (i, j) in cosin_sim_history:
                    user_cos_sim += cosin_sim_history[(i, j)]
                else:
                    col1 = interactions[:, i]
                    col2 = interactions[:, j]
                    if (denominator := np.sqrt(col1.sum() * col2.sum())) == 0:
                        cos_sim = 0
                    else:
                        cos_sim = col1.T.dot(col2).sum() / denominator
                    cosin_sim_history[(i, j)] = cos_sim
                    user_cos_sim += cos_sim
            user_history[index] = user_cos_sim
            total_cos_sim += user_cos_sim

    answer = 1 - total_cos_sim / (predicted_items.shape[0] * comb(k, 2))

    return answer, cosin_sim_history, user_history


def novelty(
    predicted_items: np.ndarray,
    interactions: sps.csc_matrix,
    k: int
) -> float:
    """Novelty@k. Calculate the novelty of the prediction.
        For each item the novelty is the opposite of the popularity.

    Args:
        predicted_items (np.ndarray): predicted items. Can be cutted to the top-k.
        interactions (sps.csc_matrix): Train interactions on which fitting was provided.
        k (int): Value to specify amount of items to predict.

    Returns:
        float: Metric value
    """
    predicted_items = predicted_items[:, :k]
    novelties = np.array(interactions.sum(axis=0)).flatten() / interactions.shape[1]
    novelties = novelties[novelties.nonzero()[0]]
    novelties = -np.log2(novelties)
    unique_count = np.bincount(predicted_items.flatten())
    len_novelties = novelties.shape[0]
    len_unique_count = unique_count.shape[0]
    if len_novelties >= len_unique_count:
        m_r = np.zeros(len_novelties)
        m_r[:len_unique_count] = unique_count
    else:
        m_r = unique_count[:len_novelties]
    return (m_r * novelties).sum() / np.unique(predicted_items).shape[0]



def mean_ranks(ranks: np.ndarray) -> float:
    """MeanRanks. Calculates mean rank over all users.

    Args:
        ranks (np.ndarray): True relative items' ranks
        k (int): Number of the top predicted items

    Returns:
        float: Metric's value
    """
    return (np.where(ranks == -1, 0, ranks).sum(1) / (ranks != -1).sum(1)).mean()


def run_all_metrics(
    ranks: np.ndarray,
    k: Union[int, list[int]],
    predicted_items: Optional[np.ndarray] = None,
    train_interactions: Optional[sps.csc_matrix] = None,
    nfolds_cache: Optional[
        tuple[
            Optional[dict[tuple[int, int], float]], Optional[list[dict[int, float]]]
        ]
    ] = None,
    idx: Optional[np.ndarray] = None,
) -> Union[
        list[list[float]],
        tuple[list[list[float]], dict[tuple[int, int], float], list[dict[int, float]]]
    ]:
    """Run all of the above metrics 
    (except the Coverage, Diversity and Novelty
    as they are required more aatention and some other parameters)

    Args:
        ranks (np.ndarray): Relevant items' ranks.
        k (Union[int, list[int]]): Number of the top predicted items
        predicted_items (Optional[np.ndarray]):
            Top k predicted items to calculate to calculate novelty and diversity metrics.
            If None, these metrics will be ignored. Default is None.
        train_interactions (Optional[sps.csc_matrix]):
            train interaction matrix to calculate novelty and diversity metrics.
            If None, these metrics will be ignored. Default is None.
        item_col (Optional[str]): Item column to calculate the above metrics.
            Default is None.
        user_col (Optional[str]): User column to calculate the above metrics.
            Default is None.
        propogated_dict (Optional[dict[tuple[int, int], float]]): Propogate dictionary with
            users total cosine similarity. If not None, will use dictionary from the parameter and
            additionaly returns updated version. Default is None
    Returns:
        List of List[float]: A list of lists containing metrics values for each cutoff value.
    """
    all_metrics: list[list[float]] = []

    if isinstance(k, int):
        k = [k]

    for k_i in k:
        all_metrics.append(
            [
                mean_precision(ranks, k_i),
                mean_recall(ranks, k_i),
                mean_average_precison(ranks, k_i),
                normalized_discounted_cumulative_gain(ranks, k_i),
                mean_reciprocal_rank(ranks, k_i),
                hit_rate(ranks, k_i),
            ]
        )
    
    if predicted_items is None or train_interactions is None:
        logger.warning((
            "Some of the necessary parameters are absent! "
            "Calculating coverage, diversity and novelty is impossible and will not be performed."
        ))
        return all_metrics

    if nfolds_cache is None or idx is None:
        cosin_sim_history = {}
        for k_i, metric_list in zip(k, all_metrics):
            diversity_metric, cosin_sim_history = diversity(
                predicted_items, train_interactions, k_i, cosin_sim_history
            )
            metric_list.extend(
                [
                    coverage(ranks, train_interactions.shape[1], k_i),
                    diversity_metric,
                    novelty(predicted_items, train_interactions, k_i)
                ]
            )
        return all_metrics
    
    propogated_dict, userwise_dict = nfolds_cache
    cosin_sim_history = {} if propogated_dict is None else propogated_dict.copy()
    user_histories = [None] * len(k) if userwise_dict is None else userwise_dict.copy()

    for i, (k_i, metric_list) in enumerate(zip(k, all_metrics)):
        diversity_metric, cosin_sim_history, user_history = cached_divercity(
            predicted_items,
            idx,
            train_interactions,
            k_i,
            cosin_sim_history,
            user_histories[i]
        )
        user_histories[i] = user_history
        metric_list.extend(
            [
                coverage(ranks, train_interactions.shape[1], k_i),
                diversity_metric,
                novelty(predicted_items, train_interactions, k_i)
            ]
        )
    return all_metrics, cosin_sim_history, user_histories


def run_all_metrics_nfold(
    ranks: np.ndarray,
    k: Union[int, list[int]],
    predicted_items: Optional[np.ndarray] = None,
    train_interactions: Optional[sps.csc_matrix] = None,
    num_folds: int = 5,
    excluded_percentage: float = .2
) -> pd.DataFrame:
    """
    Calculate evaluation metrics for recommendation systems using n-fold cross-validation.

    Parameters:
        ranks (np.ndarray): True relative items' rank.
                        Use -1 for items not ranked by the user.
        k (int or List[int]): Cutoff value(s) for evaluating the top-k recommendations.
                            If int, use a single cutoff value.
                            If List[int], provide multiple values.
        predicted_items (Optional[np.ndarray]):
            Top k predicted items to calculate to calculate novelty and diversity metrics.
            If None, these metrics will be ignored. Default is None.
        train_interactions (Optional[pd.Dataframe]):
            train interaction matrix to calculate novelty and diversity metrics.
            If None, these metrics will be ignored. Default is None.
        item_col (Optional[str]): Item column to calculate the above metrics.
            Default is None.
        user_col (Optional[str]): User column to calculate the above metrics.
            Default is None.
        num_folds (int, optional): Number of folds for n-fold cross-validation. Default is 5.
        fold_method (str, optional): Method for splitting data into folds. 
                                    Options are "interaction-wise" and "random".
                                    Default is "random".
        excluded_percentage (float, optional): 
                    Percentage of users to exclude from each fold when using "random" fold method.
                                            Default is 0.2 (20%).

    Returns:
        pd.DataFrame: evaluation metrics all across folds.

    Raises:
        NotImplementedError: If the specified fold_method is not implemented.
    """
    if isinstance(k, int):
        k = [k]

    num_users = ranks.shape[0]
    ranks_list, predicts_list, ids_list = [], [], []
    for _ in range(num_folds):
        fold_indices = np.random.choice(
            num_users, size=int(num_users * (1 - excluded_percentage)), replace=True
        )
        ids_list.append(fold_indices)
        ranks_list.append(ranks[fold_indices])
        predicts_list.append(predicted_items[fold_indices])
    
    logger.info("Got %i folds for metrics calculation", num_folds)

    metrics_df = []
    cosin_sim_history = None
    user_histories = None
    for index, (fold_ranks, fold_predicts, fold_indexes) in tqdm(
        enumerate(zip(ranks_list, predicts_list, ids_list)),
        "Fold_wise calculation",
        total=len(ranks_list)
    ):
        results, cosin_sim_history, user_histories = run_all_metrics(
            fold_ranks,
            k,
            fold_predicts,
            train_interactions,
            (cosin_sim_history, user_histories),
            fold_indexes
        )
        one_df = pd.DataFrame(
            results,
            columns=(
                "Precision@k",
                "Recall@K",
                "MAP@K",
                "nDCG@k",
                "MRR@k",
                "HitRate@k",
                "Coverage@k",
                "Diversity@k",
                "Novelty@k"
            ),
        )
        one_df["Run_id"] = index
        one_df['k'] = k
        one_df.set_index(["Run_id", 'k'], inplace=True)
        metrics_df.append(one_df)

    return pd.concat(metrics_df)

def get_metrics_by_method(
    path: Path,
    save_path: Path,
    is_optimized: Optional[bool],
    train_interactions: sps.csc_matrix,
    k: Union[int, list[int]],
) -> None:
    if is_optimized is None:
        items_opt = path.joinpath(f"items.npy")
        ranks_opt = path.joinpath(f"ranks.npy")
    else:
        items_opt = path.joinpath(f"items_wasOptimized_{str(is_optimized)}.npy")
        ranks_opt = path.joinpath(f"ranks_wasOptimized_{str(is_optimized)}.npy")
    if all((items_opt.exists(), ranks_opt.exists())):
        logger.info("Calculating metrics for %s optimizer", is_optimized)
        run_all_metrics(
            np.load(ranks_opt).astype(np.int32),
            k,
            np.load(items_opt).astype(np.int32),
            train_interactions,
            None
        ).to_csv(save_path.joinpath(
            f"results_wasOptimized_{str(is_optimized)}.csv"
        ))
    else:
        logger.warning("%s optimizer results does not exist", is_optimized)