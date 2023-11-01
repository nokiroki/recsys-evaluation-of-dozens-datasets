"""Most popular baseline"""
from typing import Optional, Mapping, List
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from tqdm.auto import tqdm

from src.utils.logging import get_logger

logger = get_logger(name=__name__)


class MostPopularBaseline:
    """Main class for the Most Popular item prediction baseline
    Items are weighted with the interaction count.
    For the each top predicted items will be cleared of the seen ones.
    """

    def __init__(self) -> None:
        self.ranks: Optional[np.ndarray] = None
        self.train_items: Mapping[int, List[int]] = defaultdict(list)

    def fit(self, interactions: csr_matrix) -> None:
        """Calculates top items vector

        Args:
            interactions (csr_matrix): Sparse matrix with user-item interactions
        """
        interactions = interactions.tocoo()
        films_top_rating = interactions.sum(axis=0)
        films_top_rating = np.ravel(films_top_rating)
        for user, item in zip(interactions.row, interactions.col):
            self.train_items[user].append(item)

        self.ranks = -films_top_rating

    def get_relevant_ranks(self, test_interactions: csr_matrix) -> np.ndarray:
        """Get ranks for the relevant items
        Seen items will be skipped.

        Args:
            test_interactions (csr_matrix): Relevant items for each user in the sparse format

        Returns:
            np.ndarray: ndarray with ranks
        """
        test_interactions = test_interactions.tocoo()
        test_relevants = defaultdict(list)
        max_len = test_interactions.getnnz(axis=1).max()
        for user, item in tqdm(zip(test_interactions.row, test_interactions.col)):
            test_relevants[user].append(item)
            max_len = max(max_len, len(test_relevants[user]))
        ranks = np.full((len(test_relevants), max_len), -1, dtype=np.int32)
        for i, user in tqdm(enumerate(test_relevants), total=len(test_relevants)):
            good_ranks = self.ranks.copy()
            good_ranks[self.train_items[user]] = 0
            ranks[i, :len(test_relevants[user])] = (
                good_ranks
                .argsort()
                .argsort()[test_relevants[user]]
            )

        return ranks

    def recommend_k(self, train_interactions: csr_matrix, k: int) -> np.ndarray:
        """Get k predicted items for each user.
        Will not be same because of seen items pruning.

        Args:
            train_interactions (csr_matrix): Sparse matrix with the user-item train interactions
            k (int): amount of items to predict

        Returns:
            np.ndarray: ndarray with predicted items
        """
        k = min(k, self.ranks.shape[0])
        predicts = np.zeros((train_interactions.shape[0], k), dtype=np.int32)
        for user in tqdm(range(train_interactions.shape[0])):
            good_ranks = self.ranks.copy()
            good_ranks[self.train_items[user]] = 0
            predicts[user] = good_ranks.argsort()[:k]
        return predicts
