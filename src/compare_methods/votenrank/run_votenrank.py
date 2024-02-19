import pandas as pd
from . import Leaderboard

def run_printtable_votenrank(data: pd.DataFrame, draw_plot: bool = False, values = None) -> pd.DataFrame:

    if values is not None:
        pivot_df = data.pivot(index='Method', columns='Dataset', values=values)
    else:
        pivot_df = data.pivot(index='Method', columns='Dataset', values='Value')
    naive_lb = Leaderboard(pivot_df)
    res = naive_lb.rank_all()

    return res
