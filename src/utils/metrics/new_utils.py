from typing import Union
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.special import comb
import scipy.sparse as sps

from tqdm.auto import tqdm

def mean_precision(predicts: np.ndarray, gts_agg: pd.Series, k: int) -> float:
    scores = []
    for idx, gts_user in enumerate(gts_agg):
        predicts_user = predicts[idx, :k]
        scores.append(len(np.intersect1d(predicts_user, gts_user)) / min(k, len(gts_user)))
    return np.mean(scores)


def mean_recall(predicts: np.ndarray, gts_agg: pd.Series, k: int) -> float:
    scores = []
    for idx, gts_user in enumerate(gts_agg):
        predicts_user = predicts[idx, :k]
        scores.append(len(np.intersect1d(predicts_user[:k], gts_user)) / len(gts_user))
    return np.mean(scores)


def mean_average_precision(predicts: np.ndarray, gts_agg: pd.Series, k: int) -> float:
    scores = []
    for idx, gts_user in enumerate(gts_agg):
        predicts_user = predicts[idx, :k]
        right_preds = np.in1d(predicts_user[:k], gts_user)
        user_scores = np.where(right_preds, right_preds.cumsum(), 0)
        user_scores = user_scores / np.arange(1, k + 1)
        scores.append(user_scores.sum() / min(k, len(gts_user)))
    return np.mean(scores)


def normalized_discounted_cumulative_gain(
    predicts: np.ndarray, gts_agg: pd.Series, k: int
) -> float:
    scores = []
    dcg_range = 1 / np.log2(np.arange(2, k + 2))
    for idx, gts_user in enumerate(gts_agg):
        predicts_user = predicts[idx, :k]
        ideal_dcg = dcg_range[:min(k, len(gts_user))].sum()
        right_preds = np.in1d(predicts_user[:k], gts_user).astype(np.int32)
        dcg = np.sum(right_preds * dcg_range)
        scores.append(dcg / ideal_dcg)

    return np.mean(scores)


def hit_rate(predicts: np.ndarray, gts_agg: pd.Series, k: int) -> float:
    scores = []
    for idx, gts_user in enumerate(gts_agg):
        predicts_user = predicts[idx, :k]
        scores.append(int(np.sum(np.in1d(predicts_user[:k], gts_user)) > 0))
    return np.mean(scores)


def mean_reciprocal_rank(predicts: np.ndarray, gts_agg: pd.Series, k: int) -> float:
    scores = []
    for idx, gts_user in enumerate(gts_agg):
        predicts_user = predicts[idx, :k]
        ranks = np.arange(1, k + 1)[np.in1d(predicts_user[:k], gts_user)]
        if len(ranks) == 0:
            scores.append(0)
        else:
            scores.append(1 / ranks[0])
    return np.mean(scores)


def coverage(predicts: np.ndarray, total_items: int, k: int) -> float:
    num_unique = len(np.unique(predicts[:, :k]))
    return num_unique / total_items


def diversity(
    predicted_items: np.ndarray,
    interactions: sps.csc_matrix,
    k: int
) -> float:
    predicted_items = predicted_items[:, :k].astype(np.int32)
    cosin_sim_history = {}
    l2_norms = np.sqrt(interactions.power(2).sum(axis=0))
    total_cos_sim = 0.0
    for item_ids in tqdm(predicted_items, desc="Diversity calculation", leave=False):
        for i, j in combinations(item_ids, 2):
            if (i, j) not in cosin_sim_history:
                # Compute cosine similarity using precomputed L2 norms
                cos_sim = interactions[:, i].T.dot(interactions[:, j])[0, 0] / (l2_norms[0, i] * l2_norms[0, j])
                cos_sim = 0 if np.isnan(cos_sim) or np.isinf(cos_sim) else cos_sim
                cosin_sim_history[(i, j)] = cos_sim
            total_cos_sim += cosin_sim_history[(i, j)]
    
    # Calculate the average cosine similarity and subtract from 1 to get diversity
    num_pairs = comb(k, 2)
    diversity_score = 1 - total_cos_sim / (predicted_items.shape[0] * num_pairs)
    
    return diversity_score


def novelty(
    predicted_items: np.ndarray,
    interactions: sps.csc_matrix,
    k: int
) -> float:
    predicted_items = predicted_items[:, :k]
    popularity = np.array(interactions.sum(axis=0)).flatten() / interactions.shape[0]
    # Avoid division by zero in the log operation
    popularity = np.clip(popularity, 1e-12, None)
    item_novelty = -np.log2(popularity)
    # Calculate the novelty for the recommended items only
    recommended_item_novelties = item_novelty[predicted_items.flatten()]
    # Calculate the average novelty across all recommended items
    novelty_score = np.mean(recommended_item_novelties)
    return novelty_score

def run_all_metrics(predicts: np.ndarray, gts_agg: pd.Series, k: Union[int, list[int]]):
    all_metrics: list[list[float]] = []
    if isinstance(k, int):
        k = [k]
    for k_i in k:
        all_metrics.append(
            [
                mean_precision(predicts, gts_agg, k_i),
                mean_recall(predicts, gts_agg, k_i),
                mean_average_precision(predicts, gts_agg, k_i),
                normalized_discounted_cumulative_gain(predicts, gts_agg, k_i),
                mean_reciprocal_rank(predicts, gts_agg, k_i),
                hit_rate(predicts, gts_agg, k_i),
            ]
        )
    return all_metrics


if __name__ == "__main__":
    gts = pd.Series([
        [1, 2],
        [1, 2, 3, 4, 5],
        [1, 2, 3]
    ])

    predicts = np.array([
        [10, 2, 4, 5, 1],
        [1, 2, 10, 4, 8],
        [5, 8, 1, 10, 4]
    ])

    # gts = np.array([
    #     [1, 2, -1, -1, -1],
    #     [1, 2, 3, 4, 5],
    #     [1, 2, 3, -1, -1]
    # ])

    print(mean_precision(predicts, gts, 2))
    print(mean_recall(predicts, gts, 2))
    print(mean_average_precision(predicts, gts, 2))
    print(normalized_discounted_cumulative_gain(predicts, gts, 2))
    print(hit_rate(predicts, gts, 2))
    print(mean_reciprocal_rank(predicts, gts, 2))
