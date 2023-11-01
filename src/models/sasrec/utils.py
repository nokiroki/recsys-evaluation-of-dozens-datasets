"""Sampler for batch generation."""
import multiprocessing
from multiprocessing import Process, Queue

import numpy as np


def random_neq(
    l_bound: int,
    r_bound: int,
    s_items: set[int],
) -> int:
    """Generate negative item for chosen user.

    Args:
        l_bound (int): left bound of sampling.
        r_bound (int): right bound of samling.
        s_items (set[int]): set of user items.
    Return:
        int: random item which was not in the set.
    """
    output = np.random.randint(l_bound, r_bound)
    while output in s_items:
        output = np.random.randint(l_bound, r_bound)
    return output


def sample_function(
    user_train: dict[int, list[int]],
    usernum: int,
    itemnum: int,
    batch_size: int,
    maxlen: int,
    result_queue: multiprocessing.Queue,
    seed_param: int,
) -> (int, list[int], list[int], list[int]):
    """Generate batch sample.

    Args:
        user_data (dict[int, list[int]]): dict of user item sequences.
        usernum (int): number of unique users in data.
        itemnum (int): number of unique items in data.
        batch_size (int): batch size for model input.
        maxlen (int): length of item sequence.
        result_queue (multiprocessing.Queue): queue for batch data.
        seed_param (int): seed number for reproducibility.
    
    Return (int, list[int], list[int], list[int]): user, sequence of items, \
        positive examples and negative esamples.
    """


    def sample():
        user = np.random.choice([*user_train])
        while len(user_train[user]) <= 1:
            user = np.random.choice([*user_train])

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        item_seq = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, item_seq)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (user, seq, pos, neg)

    np.random.seed(seed_param)
    while True:
        one_batch = []
        for _ in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler:
    """Wrapper of data sampler for SasRec model."""

    def __init__(
        self,
        user_data: dict[int, list[int]],
        usernum: int,
        itemnum: int,
        batch_size: int = 64,
        maxlen: int = 10,
        n_workers: int = 1,
    ) -> None:
        """Init WarpSampler for SasRec model.

        Args:
            user_data (dict[int, list[int]]): dict of user item sequences.
            usernum (int): number of unique users in data.
            itemnum (int): number of unique items in data.
            batch_size (int): batch size for model input.
            maxlen (int): length of item sequence.
            n_worker (int): number of worker for Process.
        """
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for _ in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(
                        user_data,
                        usernum,
                        itemnum,
                        batch_size,
                        maxlen,
                        self.result_queue,
                        np.random.randint(2e9),
                    ),
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        """Sample next batch."""
        return self.result_queue.get()

    def close(self):
        """Close all processors."""
        for p_proc in self.processors:
            p_proc.terminate()
            p_proc.join()
