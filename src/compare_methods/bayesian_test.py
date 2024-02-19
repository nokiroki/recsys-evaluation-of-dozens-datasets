"""Bayesian test module"""
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from baycomp import HierarchicalTest, SignedRankTest

from src.utils.logging import get_logger

logger = get_logger(name=__name__)


def get_score_array(data: pd.DataFrame, method_name: str, bootstraped: bool = True) -> np.ndarray:
    """Scores for the method by all datasets.

    Args:
        data (pd.DataFrame): Standart type pandas dataframe.
        method_name (str): Method for which we return metrics.
        bootstraped (bool, optional): If we are working with bootstraped values. Defaults to True.

    Returns:
        np.ndarray: Numpy array with method values.
    """
    data_temp = data[data["Method"] == method_name].drop(columns=["Method"], axis=1)
    if not bootstraped:
        return data_temp["Value"].values
    return data_temp.pivot(index="Dataset", columns="Run_id", values="Value").values


def bayes_scores(
    data: pd.DataFrame,
    rope: float = .0,
    bootstraped: bool = False,
    nsamples: int = 10000
) -> pd.DataFrame:
    """Bayesian method for methods comparison. Calculate Hierarchical Test for bootstraped data
        end Signed Rank Test vice versa.

    Args:
        data (pd.DataFrame): Standart type pandas dataframe.
        rope (float, optional): Region Of Practical Equivalence. Defaults to .0.
        bootstraped (bool, optional): If we are working with bootstraped values. Defaults to False.
        nsamples (int, optional): Number of samples to use while calculating the probabilities.
            Defaults to 10000.

    Returns:
        pd.DataFrame: Dataframe with the results.
    """
    results = list()
    names = data["Method"].unique()
    for method_1, method_2 in combinations(names, 2):
        scores_1 = get_score_array(data, method_1, bootstraped)
        scores_2 = get_score_array(data, method_2, bootstraped)
        if len(scores_1) != len(scores_2):
            logger.warning(
                "Length are different. Probabilities won't be calculcated for pair %s and %s",
                method_1,
                method_2
            )
            continue

        test = HierarchicalTest if bootstraped else SignedRankTest
        answer = test.probs(scores_1, scores_2, rope, nsamples=nsamples)
        if rope > 0:
            p_more, p_rope, p_less = answer
        else:
            p_more, p_less = answer
            p_rope = .0
        
        results.append((method_1, method_2, p_more, p_less, p_rope))

    results = pd.DataFrame(results, columns=("clf_1", "clf_2", "p_more", "p_less", "p_rope"))
    return results


def binarize_bayes(results: pd.DataFrame, threshold: float = .9) -> pd.DataFrame:
    """ Calculates if one of the probalities is significantly more likely then the others.

    Args:
        results (pd.DataFrame): The results after bayesian tests.
        threshold (float, optional): Probability threshold. Defaults to .9.

    Returns:
        pd.DataFrame: New Dataframe with the binary column.
    """
    results_copy = results.copy()
    results_copy[["p_more", "p_less", "p_rope"]] = \
        results_copy[["p_more", "p_less", "p_rope"]] >= threshold
    results_copy["is_significance"] = \
        results_copy["p_more"] | results_copy["p_less"] | results_copy["p_rope"]
    return results_copy
