"""Critical Difference diagram plotting."""
# Author: Hassan Ismail Fawaz <hassan.ismail-fawaz@uha.fr>
#         Germain Forestier <germain.forestier@uha.fr>
#         Jonathan Weber <jonathan.weber@uha.fr>
#         Lhassane Idoumghar <lhassane.idoumghar@uha.fr>
#         Pierre-Alain Muller <pierre-alain.muller@uha.fr>
# License: GPL3
from typing import Sequence
import argparse
from pathlib import Path
import operator
import math

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import networkx

from .bayesian_test import bayes_scores, binarize_bayes

matplotlib.use("agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "Arial"

ALPHA = 0.05


class CriticalDifference:

    def __init__(self) -> None:
        pass

# inspired from orange3 https://docs.orange.biolab.si/3/data-mining-library/reference/evaluation.cd.html
def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, labels=False, bayes_probs=None, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        labels (bool, optional): if set to `True`, the calculated avg rank
        values will be displayed
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.22 + minnotsignificant

    fig = plt.figure(figsize=(width+3.7, height+2.7))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=2)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=16)

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=linewidth)
        if labels:
            text(textspace + 0.4, chei - 0.075, format(ssums[i], '.2f'), ha="right", va="center", size=14) #10 #.4f
        text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=16)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=linewidth)
        if labels:
            text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.2f'), ha="left", va="center", size=14)
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
             ha="left", va="center", size=16)

    # draw_lines(lines)
    start = cline + 0.2
    side = -0.02
    height = 0.1

    if bayes_probs is not None:
        cliques = form_cliques_bayes(bayes_probs, nnames)
        i = 1
        achieved_half = False
        #print(nnames) #methods
        for clq in cliques:
            if len(clq) == 1:
                continue
            #print(clq)
            min_idx = np.array(clq).min()
            max_idx = np.array(clq).max()
            if min_idx >= len(nnames) / 2 and achieved_half == False:
                start = cline + 0.25
                achieved_half = True
            line([(rankpos(ssums[min_idx]) - side, start - .08),
                (rankpos(ssums[max_idx]) + side, start - .08)],
                linewidth=linewidth_sign,
                color="grey",
                linestyle="dashed"
            )
            start += height

    # draw no significant lines
    # get the cliques
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    #print(nnames) #methods
    for clq in cliques:
        if len(clq) == 1:
            continue
        #print(clq)
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        line([(rankpos(ssums[min_idx]) - side, start -.05),
              (rankpos(ssums[max_idx]) + side, start - .05)],
             linewidth=linewidth_sign
        )
        start += height


def form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)

def form_cliques_bayes(significance_df: pd.DataFrame, nnames: Sequence[str]):
    g_data = np.zeros((len(nnames), len(nnames)), dtype=np.int64)
    for name_1, name_2 in significance_df[
        ~significance_df["is_significance"]
    ][["clf_1", "clf_2"]].values:
        if name_1 not in nnames or name_2 not in nnames:
            continue
        i = np.where(nnames == name_1)[0][0]
        j = np.where(nnames == name_2)[0][0]
        g_data[min(i, j), max(i, j)] = 1
    return networkx.find_cliques(networkx.Graph(g_data))


def draw_cd_diagram(
    df_perf=None,
    alpha=0.05,
    bayes_threshold=.8,
    title=None, 
    labels=False,
    show_plot: bool = False,
    save_image: bool = False, 
    image_path: Path = None, 
    image_name: str = 'CD',
):
    """
    Draws the critical difference diagram given the list of pairwise classifiers that are
    significant or not
    """
    p_values, average_ranks, _, _ = wilcoxon_holm(df_perf=df_perf, alpha=alpha)
    bayes_probs = bayesian_test(df_perf, bayes_threshold, nsamples=1000)
    
    graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
                cd=None, reverse=True, width=10, textspace=1.2, labels=labels, bayes_probs=bayes_probs)
    
    font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        }
    if title:
        plt.title(title,fontdict=font, y=0.9, x=0.5)
    
    if show_plot:
        plt.show()
    if save_image:
        full_path = image_path.joinpath(image_name + ".pdf")
        plt.savefig(full_path, dpi=200, )

    
    #plt.savefig('cd-diagram.png',bbox_inches='tight')

    return average_ranks

def bayesian_test(
    df_perf: pd.DataFrame, threshold: float = .9, **test_kwargs
) -> pd.DataFrame:
    scores_table = binarize_bayes(bayes_scores(df_perf, **test_kwargs), threshold)
    return scores_table.drop(columns="p_rope")

def wilcoxon_holm(df_perf=None, alpha=0.05):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    """
    #print(pd.unique(df_perf['Method']))
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['Method']).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['Method'])
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['Method'] == c]['Value'])
        for c in classifiers))[1]
    if friedman_p_value >= alpha:
        # then the null hypothesis over the entire classifiers cannot be rejected
        print('the null hypothesis over the entire classifiers cannot be rejected. Proceeding anyway')
    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(df_perf.loc[df_perf['Method'] == classifier_1]['Value']
                          , dtype=np.float64)
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(df_perf.loc[df_perf['Method'] == classifier_2]
                              ['Value'], dtype=np.float64)
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
            # appen to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            # stop
            break
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf['Method'].isin(classifiers)]. \
        sort_values(['Method', 'Dataset'])
    # get the rank data
    rank_data = np.array(sorted_df_perf['Value']).reshape(m, max_nb_datasets)

    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=
    np.unique(sorted_df_perf['Dataset']))

    # number of wins #ranks
    #print('Ranks of models')
    dfff = df_ranks.rank(ascending=False)
    #print(dfff)
    #print(dfff[dfff == 1.0].sum(axis=1))
    num_wins = dfff[dfff == 1.0].sum(axis=1)

    # average the ranks
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets, num_wins


def run_CD(
    data: pd.DataFrame, 
    alpha: float = 0.05,
    draw_plot: bool = True,
    save_image: bool = False, 
    image_path: Path = None, 
    image_name: str = 'CD',
) -> pd.DataFrame:

    mapping_dict = {
        'sasrec': 'SASRec',
        'recbole_ItemKNN': 'ItemKNN',
        'recbole_EASE': 'EASE',
        'recbole_MultiVAE': 'MultiVAE',
        'recbole_LightGCN': 'LightGCN',
        'recbole_LightGCL': 'LightGCL',
        'implicit_bpr': 'BPR',
        'implicit_als': 'ALS',
        'lightfm': 'LightFM',
        'recbole_SLIMElastic': 'SLIM',
        'most_popular': 'MostPop',
        'random': 'Random',
    }
    data.loc[:, 'Method'] = data['Method'].replace(mapping_dict)

    df_res = draw_cd_diagram(
        df_perf=data,
        alpha=alpha, 
        title='nDCG@10', 
        labels=True,
        show_plot = draw_plot,
        save_image = save_image, 
        image_path = image_path, 
        image_name = image_name,
    )

    df_res = pd.DataFrame(df_res, columns=['score'])
    df_res = df_res.reset_index().rename(columns={'index': 'Model_name'})
    df_res['ranks'] = df_res['score'].rank(ascending=True)
    return df_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--path',default = 'C:/Users/User/Desktop/SMILES/project/metrics_recsys.csv', help='Path to csv data file')
    parser.add_argument('--save_fig',default = None , help='Path to saving output cd diagram')
    args = parser.parse_args()
    path = args.path
    save_fig = args.save_fig

    df_perf = pd.read_csv(path, index_col=False)

    draw_cd_diagram(df_perf=df_perf, title='nDCG@k', labels=True)
    if save_fig!=None:
       plt.savefig(f'{save_fig}/CD_diagram.png')


    p_values, average_ranks, _, num_wins = wilcoxon_holm(df_perf=df_perf, alpha=ALPHA)
    print(average_ranks)
    #print(num_wins)
