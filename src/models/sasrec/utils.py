"""Filter interactions."""
import multiprocessing
from multiprocessing import Process, Queue

import numpy as np


def add_time_idx(df, user_col='userId', timestamp_col='timestamp', sort=True):
    """Add time index to interactions dataframe."""

    if sort:
        df = df.sort_values([user_col, timestamp_col])

    df['time_idx'] = df.groupby(user_col).cumcount()
    df['time_idx_reversed'] = df.groupby(user_col).cumcount(ascending=False)

    return df