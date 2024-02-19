"""Random baseline"""
from typing import Optional, Mapping, List
from collections import defaultdict
import time

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
        self.learning_time: Optional[float] = None
        self.predict_time: Optional[float] = None

    def fit(self, interactions: csr_matrix) -> None:
        """Initialize the seen items matrix for the user.

        Args:
            interactions (csr_matrix): Sparse matrix with user-item interactions
        """
        start_time = time.time()
        interactions = interactions.tocoo()
        for user, item in zip(interactions.row, interactions.col):
            self.train_items[user].append(item)
        self.max_items = interactions.shape[1]
        
        self.learning_time = time.time() - start_time

    def recommend_k(self, k: int, userids: np.ndarray) -> np.ndarray:
        """Recommend top k items for specified users.

        Args:
            k (int): Number of items to recommend.
            userids (np.ndarray): Array of user IDs to generate recommendations for.

        Returns:
            np.ndarray: Array with top k recommended items for each specified user.
        """
        k = min(k, self.max_items)  # Ensure k does not exceed the number of items

        # Initialize array to store recommendations
        recommendations = np.zeros((len(userids), k), dtype=np.int32)

        start_time = time.time()
        
        for idx, user in enumerate(userids):
            seen_items = set(self.train_items[user])
            available_items = np.setdiff1d(np.arange(self.max_items), list(seen_items))

            # Randomly select k items from available items
            num_recommendations = min(k, len(available_items))
            recommendations[idx, :num_recommendations] = rnd.choice(available_items, size=num_recommendations, replace=False)
        
        self.predict_time = time.time() - start_time
        
        return recommendations