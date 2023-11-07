"""Dolan-More curves plotting."""
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from .colors import PRETTY_COLORS

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

__all__ = ["run_dolan_more"]


def calculate_curves(
    df_betas: pd.DataFrame,
    betas_space: np.ndarray,
) -> pd.Series:
    """Method for calculating betas space for each method (num of succeed tasks over the beta space)

    Args:
        df_betas (pd.DataFrame): Dataframe with the shape num_methods x num_datasets.
            Contains beta values for each tuple
        betas_space (np.ndarray): Linspace of betas values.
            Can be any range and any number of points.

    Returns:
        pd.Series: Dataframe with the shape num_methods x amount_betas.
    """
    n_datasets = df_betas.shape[1]

    df_curves = pd.DataFrame(index=df_betas.index, columns=betas_space)
    for beta in betas_space:
        df_curves[beta] = (df_betas <= beta).sum(axis=1) / n_datasets

    return df_curves


def calculate_auc(df_curves: pd.DataFrame) -> pd.DataFrame:
    """Calculating the Area Under Curve (AUC) for the Dolan-More curves.

    Args:
        df_curves (pd.DataFrame): Dataframe with the shape num_methods x amount_betas.

    Returns:
        pd.DataFrame: Dataframe with auc values.
    """
    curves = df_curves.T
    curves_shift = curves.shift(1).iloc[1:]
    curves = curves.iloc[1:]

    dm_auc = ((curves + curves_shift) / 2).sum()
    dm_auc /= dm_auc.sum()
    dm_auc = pd.DataFrame(dm_auc, columns=["score"])
    dm_auc = dm_auc.reset_index().rename(columns={"Method": "Model_name"})
    dm_auc["ranks"] = dm_auc["score"].rank(ascending=False)
    return dm_auc


# pylint: disable=too-many-arguments
def draw_dm_curves(
    df_curves: pd.DataFrame,
    dm_auc: pd.DataFrame,
    figsize: tuple[int, int] = (7, 5),
    dpi: int = 200,
    linewidth: float = 1.5,
    fontsize: int = 14,
    grid: bool = False,
    image_path: Path = None,
    image_name: str = "DM",
    image_format: str = "pdf"
) -> None:
    """Draw Dolan-More curves via matplotlib.

    Args:
        df_curves (pd.DataFrame): Curves to plot.
        dm_auc (pd.DataFrame): AUC values for the legend.
        figsize (tuple[int, int], optional): Size of the figure. Defaults to (7, 5).
        dpi (int, optional): DPI of the figure. Defaults to 200.
        linewidth (float, optional): Width of the lines to plot. Defaults to 1.5.
        fontsize (int, optional): Size of the font for X and Y axises. Defaults to 14.
        grid (bool, optional): Enable grid. Defaults to False.
        image_path (Path, optional): Path to save. Defaults to None.
        image_name (str, optional): Name of the file. Defaults to "DM".
        image_format (str, optional): Format to save. Supports the same formats as matplotlib.
            Defaults to "pdf".
    """

    plt.figure(figsize=figsize, dpi=dpi)

    for i, method_name in enumerate(dm_auc.sort_values("score", ascending=False)["Model_name"]):
        curve = df_curves.loc[method_name]
        rating = dm_auc[dm_auc["Model_name"] == method_name]["score"].values[0]
        plt.plot(
            curve.index,
            curve.values,
            label = f"{method_name} AUC={round(rating, 3)}",
            linewidth = linewidth,
            color=PRETTY_COLORS[i] if i < len(PRETTY_COLORS) else None
        )
    plt.legend(loc="lower right", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(r"$\beta$", fontsize=fontsize + 4)
    plt.ylabel(r"Fraction of datasets p($\beta$)", fontsize=fontsize + 4)
    if grid:
        plt.grid(linewidth=.5)
    plt.savefig(
        image_path.joinpath(image_name + f".{image_format}"),
        dpi="figure",
        bbox_inches="tight"
    )


# pylint: disable=too-many-arguments
def run_dolan_more(
    old_data: pd.DataFrame,
    max_beta: float = 3.,
    mode: str = "normal",
    step: int = 0.1,
    save_image: bool = True,
    figsize: tuple[int, int] = (7, 5),
    dpi: int = 200,
    linewidth: float = 1.5,
    fontsize: int = 14,
    grid: bool = False,
    image_path: Path = None,
    image_name: str = "DM",
    image_format: str = "pdf"
) -> pd.Series:
    """Run dolan-more total script for the plotting and calculating metrics.

    Args:
        old_data (pd.DataFrame): Aggregated metrics dataframe, unpivoted via pd.melt method.
        max_beta (float, optional): _description_. Defaults to 3.
        mode (str, optional): _description_. Defaults to "normal".
        step (int, optional): _description_. Defaults to 0.1.
        save_image (bool, optional): _description_. Defaults to True.
        figsize (tuple[int, int], optional): Size of the figure. Defaults to (7, 5).
        dpi (int, optional): DPI of the figure. Defaults to 200.
        linewidth (float, optional): Width of the lines to plot. Defaults to 1.5.
        fontsize (int, optional): Size of the font for X and Y axises. Defaults to 14.
        grid (bool, optional): Enable grid. Defaults to False.
        image_path (Path, optional): Path to save. Defaults to None.
        image_name (str, optional): Name of the file. Defaults to "DM".
        image_format (str, optional): Format to save. Supports the same formats as matplotlib.
            Defaults to "pdf".

    Raises:
        NotImplementedError: _description_

    Returns:
        pd.Series: _description_
    """
    if mode not in ("normal", "leave_best_out"):
        raise NotImplementedError(f"Method {mode} not implemented")

    data = old_data.pivot(index="Method", columns="Dataset", values="Value")
    # Max values of metrics for each dataset
    max_values_ds = data.max()
    # Betas dataset index - methods, columns - datasets
    df_betas = max_values_ds / data

    beta_space = np.linspace(1., max_beta, int((max_beta - 1) / step) + 1)

    df_curves = calculate_curves(df_betas, beta_space)
    dm_auc = calculate_auc(df_curves)

    if save_image:
        draw_dm_curves(
            df_curves=df_curves,
            dm_auc=dm_auc,
            figsize=figsize,
            dpi=dpi,
            linewidth=linewidth,
            fontsize=fontsize,
            grid=grid,
            image_path=image_path,
            image_name=image_name,
            image_format=image_format
        )

    if mode == "normal":
        return dm_auc

    new_ranks = {}
    j = 1
    while len(dm_auc["Model_name"].unique()) > 1:
        min_rank = dm_auc["ranks"].min()

        if len(dm_auc[dm_auc["ranks"]==min_rank]["Model_name"].tolist()) > 1:
            for x in dm_auc[dm_auc["ranks"] == min_rank]["Model_name"].tolist():
                new_ranks[x] = (j + len(dm_auc[dm_auc["ranks"] == min_rank]["Model_name"].tolist()) - 1) \
                / len(dm_auc[dm_auc["ranks"] == min_rank]["Model_name"].tolist())
            j += len(dm_auc[dm_auc["ranks"] == min_rank]["Model_name"].tolist())
        else:

            new_ranks[dm_auc[dm_auc["ranks"] == min_rank]["Model_name"].tolist()[0]] = j
            j += 1

        old_data = old_data[
            old_data["Method"].isin(dm_auc[dm_auc["ranks"] != min_rank]["Model_name"])
        ]

        data = old_data.pivot(
            index="Method", columns="Dataset", values="Value"
        ).reset_index(drop=True)

        # Max values of metrics for each dataset
        max_values_ds = data.max()
        # Betas dataset index - methods, columns - datasets
        df_betas =  max_values_ds / data
        df_curves = calculate_curves(df_betas, beta_space)
        dm_auc = calculate_auc(df_curves)


    new_ranks[old_data["Method"].tolist()[0]] = j
    return new_ranks
