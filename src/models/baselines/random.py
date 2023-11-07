"""Random baseline"""
from typing import Optional, Mapping, List
from collections import defaultdict

import numpy as np
import numpy.random as rnd
from scipy.sparse import csr_matrix

from src.utils.logging import get_logger

logger = get_logger(name=__name__)


class RandomBaseline:
    """Main random baseline
    Random sampled predict items for each user.
    """

    def __init__(self) -> None:
        self.train_items: Mapping[int, List[int]] = defaultdict(list)
        self.max_items: Optional[int] = None

    def fit(self, interactions: csr_matrix) -> None:
        """Initialize the seen items matrix for the user.

        Args:
            interactions (csr_matrix): Sparse matrix with user-item interactions
        """
        interactions = interactions.tocoo()
        for user, item in zip(interactions.row, interactions.col):
            self.train_items[user].append(item)
        self.max_items = interactions.shape[1]

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
        for user, item in zip(test_interactions.row, test_interactions.col):
            test_relevants[user].append(item)
            max_len = max(max_len, len(test_relevants[user]))
        ranks = np.ones((len(test_relevants), max_len), dtype=np.int32) * -1
        for i, user in enumerate(test_relevants):
            temp_ranks = np.arange(self.max_items)
            rnd.shuffle(temp_ranks)
            ranks[i, :len(test_relevants[user])] = temp_ranks[test_relevants[user]]
        return ranks

    def recommend_k(self, test_interactions: csr_matrix, k: int) -> np.ndarray:
        """Get k predicted items for each user.
        Will not be same because of seen items pruning.

        Args:
            train_interactions (csr_matrix): Sparse matrix with the user-item train interactions
            k (int): amount of items to predict

        Returns:
            np.ndarray: ndarray with predicted items
        """
        k = min(k, test_interactions.shape[1])
        predicts = np.zeros((test_interactions.shape[0], k), dtype=np.int32)
        for user in range(test_interactions.shape[0]):
            temp_ranks = np.arange(self.max_items)
            rnd.shuffle(temp_ranks)
            predicts[user] = temp_ranks.argsort()[:k]
        return predicts
