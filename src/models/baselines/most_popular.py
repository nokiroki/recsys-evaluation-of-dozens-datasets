"""Most popular baseline"""
from typing import Optional, Mapping, List
from collections import defaultdict
import time

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
        self.item_popularity: Optional[np.ndarray] = None
        self.train_items: Mapping[int, List[int]] = defaultdict(list)
        self.learning_time: Optional[float] = None
        self.predict_time: Optional[float] = None

    def fit(self, interactions: csr_matrix) -> None:
        """Calculate the popularity of each item based on interaction counts.

        Args:
            interactions (csr_matrix): Sparse matrix with user-item interactions.
        """
        start_time = time.time()

        # Sum of interactions for each item
        self.item_popularity = np.array(interactions.sum(axis=0)).ravel()

        # Store the items each user has interacted with
        interactions = interactions.tocoo()
        for user, item in zip(interactions.row, interactions.col):
            self.train_items[user].append(item)

        self.learning_time = time.time() - start_time

    def recommend_k(self, k: int, userids: np.ndarray) -> np.ndarray:
        """Recommend top k items for specified users.

        Args:
            k (int): Number of items to recommend.
            userids (np.ndarray): Array of user IDs to generate recommendations for.

        Returns:
            np.ndarray: Array with top k recommended items for each specified user.
        """
        num_items = self.item_popularity.size
        k = min(k, num_items)  # Ensure k does not exceed the number of items

        # Initialize array to store recommendations
        recommendations = np.zeros((len(userids), k), dtype=np.int32)

        start_time = time.time()

        for idx, user in enumerate(userids):
            seen_items = set(self.train_items[user])
            item_indices = np.argsort(-self.item_popularity)  # Sort items by popularity

            # Filter out seen items and take top k
            top_k_items = [item for item in item_indices if item not in seen_items][:k]
            recommendations[idx, : len(top_k_items)] = top_k_items

        self.predict_time = time.time() - start_time

        return recommendations
