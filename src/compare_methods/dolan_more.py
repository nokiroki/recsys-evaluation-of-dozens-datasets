from typing import List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'


def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    data_DM = data.pivot(index='Dataset', columns='Method', values='Value')
    data_DM = data_DM.reset_index(drop=True)
    return data_DM


def calculate_curves(
    df_betas: pd.DataFrame, 
    betas_space: np.ndarray,
) -> pd.Series:
    n_datasets = df_betas.shape[1]

    df_curves = pd.DataFrame(index=df_betas.index, columns=betas_space)
    for beta in betas_space:
        df_curves[beta] = (df_betas.T <= beta).sum() / n_datasets

    return df_curves


def draw_DM_curves(
    df_curves: pd.DataFrame, 
    DM_AUC: pd.DataFrame,
    baseline_methods: List[str] = ['random', 'most_popular'],
    linewidth: float = 1.5,
    figsize: Tuple[int] = (7, 5),
    dpi: int = 200,
    round_rating: int = 3,
    draw_plot: bool = False, 
    save_image: bool = False, 
    image_path: Path = None, 
    image_name: str ='DM',
    grid: bool = False,
    fontsize: int = 14,
    methods_map: dict[str, int] = None
) -> None:

    plt.figure(figsize=figsize, dpi=dpi)

    color_map = {0: "#1f77b4", 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd', 5: '#8c564b', 6: '#e377c2', 7: '#7f7f7f', 8: '#bcbd22', 9: '#17becf'}
    pretty_names = {"recbole_EASE": "EASE", "recbole_ItemKNN": "ItemKNN", \
        "recbole_MultiVAE": "MultiVAE", "lightfm": "LightFM", "implicit_als": "ALS", \
        "implicit_bpr": "BPR", "msrec_sasrec": "SASRec", "most_popular": "MostPop", "random": "Random", "method_a": "Method A",
        "recbole_SLIMElastic": "SLIM"}

    for method_name in DM_AUC.sort_values('score', ascending=False)['Model_name']:
        curve = df_curves.loc[method_name]
        # linestyle = '--' if method_name in baseline_methods else '-'
        linestyle = '-'
        rating = DM_AUC[DM_AUC['Model_name'] == method_name]['score'].values[0]
        if methods_map is not None:
            plt.plot(
                curve.index, 
                curve.values, 
                label = f'{pretty_names[method_name]} AUC={round(rating, round_rating)}', 
                linestyle = linestyle,
                linewidth = linewidth,
                color=color_map[methods_map[method_name]]
            )
        else:
            plt.plot(
                curve.index, 
                curve.values, 
                label = f'{pretty_names[method_name]} ({round(rating, round_rating)})', 
                linestyle = linestyle,
                linewidth = linewidth,
            )
    plt.legend(loc="lower right", fontsize=fontsize,)
    plt.xticks(fontsize=fontsize,)
    plt.yticks(fontsize=fontsize,)
    plt.xlabel(r"$\beta$", fontsize=fontsize + 4,)
    plt.ylabel(r"Fraction of datasets p($\beta$)", fontsize=fontsize + 4,)
    if grid:
        plt.grid(linewidth=0.5)
    if save_image:
        full_path = image_path.joinpath(image_name+'.pdf')
        plt.savefig(full_path, dpi="figure", bbox_inches="tight")


def calculate_AUC(df_curves: pd.DataFrame) -> pd.DataFrame:
    curves = df_curves.T
    curves_shift = curves.shift(1).iloc[1:]
    curves = curves.iloc[1:]

    DM_AUC = ((curves + curves_shift) / 2).sum()
    DM_AUC = DM_AUC / DM_AUC.sum()
    DM_AUC = pd.DataFrame(DM_AUC, columns=['score'])
    DM_AUC = DM_AUC.reset_index().rename(columns={'Method': 'Model_name'})
    DM_AUC['ranks'] = DM_AUC['score'].rank(ascending=False)
    return DM_AUC


def run_DM(
    old_data: pd.DataFrame, 
    max_beta: float = 3.0,
    mode: str = "normal",
    step: int = 0.1,
    draw_plot: bool = True,
    baseline_methods: List[str] = ['random', 'most_popular'],
    linewidth: float = 4,
    figsize: Tuple[int] = (12, 10),
    dpi: int = 200,
    round_rating: int = 3,
    save_image: bool = False, 
    image_path: Path = None, 
    image_name: str ='DM',
    grid: bool = False,
    fontsize: int = 14,
    methods_map: dict[str, int] = None
) -> pd.Series:
    assert mode in ["normal", "leave_best_out"]
    data = transform_data(old_data)

    max_values_ds = data.T.max() #max values of metrics for each dataset
    df_betas =  max_values_ds / data.T #betas dataset index - methods, columns - datasets

    n_steps = int((max_beta - 1) / step) + 1
    betas_space = np.linspace(1.0, max_beta, n_steps)

    df_curves = calculate_curves(df_betas, betas_space)
    DM_AUC = calculate_AUC(df_curves)
    

    if draw_plot or save_image:
        draw_DM_curves(
            df_curves = df_curves,
            DM_AUC = DM_AUC,
            baseline_methods = baseline_methods,
            linewidth = linewidth,
            figsize = figsize,
            dpi = dpi,
            round_rating = round_rating,
            draw_plot=draw_plot,
            save_image=save_image, 
            image_path=image_path, 
            image_name=image_name,
            grid=grid,
            fontsize=fontsize,
            methods_map=methods_map,
        )

    if mode == "normal":
        return DM_AUC

    new_ranks = dict()
    j = 1
    while len(DM_AUC['Model_name'].unique()) > 1:
        min_rank = DM_AUC['ranks'].min()

        if len(DM_AUC[DM_AUC['ranks']==min_rank]['Model_name'].tolist()) > 1:
            for x in DM_AUC[DM_AUC['ranks'] == min_rank]['Model_name'].tolist():
                new_ranks[x] = (j + len(DM_AUC[DM_AUC['ranks'] == min_rank]['Model_name'].tolist()) - 1) \
                / len(DM_AUC[DM_AUC['ranks'] == min_rank]['Model_name'].tolist())
            j += len(DM_AUC[DM_AUC['ranks'] == min_rank]['Model_name'].tolist())
        else:

            new_ranks[DM_AUC[DM_AUC['ranks'] == min_rank]['Model_name'].tolist()[0]] = j
            j += 1

        old_data = old_data[old_data['Method'].isin(DM_AUC[DM_AUC['ranks'] != min_rank]['Model_name'])]
        
        data = transform_data(old_data)

        max_values_ds = data.T.max() #max values of metrics for each dataset
        df_betas =  max_values_ds / data.T #betas dataset index - methods, columns - datasets
        df_curves = calculate_curves(df_betas, betas_space)
        DM_AUC = calculate_AUC(df_curves)
        
    
    new_ranks[old_data['Method'].tolist()[0]] = j
    return new_ranks