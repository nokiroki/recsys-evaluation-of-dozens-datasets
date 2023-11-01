"""Script for Dolan-More curves."""
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig


def dolan_more_function(
    data: dict[str, dict[str, float]],
    max_values: dict[str, float],
    metric: str,
    beta: float,
) -> list[float]:
    """Compute Dolan More function.

    Args:
        data (dict[str, dict[str, float]]): input data points to compute DM function.
        max_values (dict[str, float]): current best metric values for each problem.
        metric (str): metric name to use.
        beta (float): beta value to measure metric values.

    Returns:
        list[float]: list of DM values for each point in data.
    """
    output = [data[i][metric] >= max_values[i] / beta for i in max_values.keys()]
    return sum(output) / len(max_values)


def get_max_values(
    data: dict[str, dict[str, dict[str, float]]],
    completed_tests: list[tuple[str]],
) -> list[float]:
    """Compute best metric values for each problem.

    Args:
        data (dict[str, dict[str, dict[str, float]]]): input data as dict.
        completed_tests (list[tuple[str]]): full metric values from each model.

    Returns:
        dict[str, list[float]]: list of best values for each dataset.
    """
    output = defaultdict(list)

    for x_key in data.keys():
        for case, metric in completed_tests:
            output[case].append(data[x_key][case][metric])

    for case, metric in completed_tests:
        output[case] = max(output[case])

    return output


def integrate_curve(tmp_res: list[float], beta_values: list[float]) -> float:
    """Integration of Dolan More curve.

    Args:
        tmp_res (list[float]): f(x) values to integrate.
        beta_values (list[float]): x values to integrate.

    Returns:
        float: integrated DM curve.
    """
    output = 0.0
    for i in range(1, len(tmp_res)):
        output += (
            (tmp_res[i] + tmp_res[i - 1]) / 2 * (beta_values[i] - beta_values[i - 1])
        )
    return output


@hydra.main(version_base=None, config_path="../config", config_name="curve_plotting")
def pp_plot(cfg: DictConfig) -> None:
    """Plot DM curve."""

    # flag to check if optuna was used
    opt_flag = ["True", "False"]
    pth_list = list(Path(cfg["data"]).glob("**/*.csv"))
    datasets = defaultdict(list)
    print(cfg["methods"])
    if cfg["methods"] == "all":
        methods_name = set()
    else:
        methods_name = set(cfg["methods"])
    for cur_path in pth_list:
        if len(cur_path.parts) > 5:

            datasets[cur_path.parts[3] + "_" + cur_path.parts[5]].append(cur_path.parts[4])
        else:
            datasets[cur_path.parts[2] + "_" + cur_path.parts[4]].append(cur_path.parts[3])
        if cfg["methods"] == "all":
            methods_name.add(cur_path.parts[2])

    # count the total amount of data
    new_data = {}
    for method in methods_name:
        for method_keys, _ in datasets.items():
            if not isinstance(method, str):
                for tmp in method:
                    if tmp in "_".join(method_keys.split("_")[:-3]):
                        new_data[method_keys] = datasets[method_keys]
            else:
                if method in "_".join(method_keys.split("_")[:-3]):
                    new_data[method_keys] = datasets[method_keys]
    

    print(new_data)

    tmp_datasets = set(new_data[list(new_data.keys())[0]])

    for data_key, _ in new_data.items():
        tmp_datasets.intersection_update(set(new_data[data_key]))

    path_to_metrics = Path("results", "metrics")

    # collect all available metrics
    results = dict()
    k = 10
    cur_metric: str = cfg["metric"]

    for optim_case in opt_flag:
        for method in methods_name:
            results[f"{method}_{optim_case}"] = defaultdict(dict)
            for case in tmp_datasets:
                if (
                    path_to_metrics
                    / Path(method, case, f"results_wasOptimized_{optim_case}.csv")
                ).is_file():
                    results[f"{method}_{optim_case}"][case] = dict()
                    cur_df = pd.read_csv(
                        path_to_metrics
                        / Path(method, case, f"results_wasOptimized_{optim_case}.csv")
                    )
                    results[f"{method}_{optim_case}"][case][cur_metric] = float(
                        cur_df.loc[cur_df[cur_df.columns[0]] == k][cur_metric]
                    )
                elif (
                    path_to_metrics / Path(method, case, "results.csv")
                ).is_file() and optim_case == "False":
                    results[f"{method}_{optim_case}"][case] = dict()
                    cur_df = pd.read_csv(
                        path_to_metrics / Path(method, case, "results.csv")
                    )
                    results[f"{method}_{optim_case}"][case][cur_metric] = float(
                        cur_df.loc[cur_df[cur_df.columns[0]] == k][cur_metric]
                    )
                else:
                    continue

    results = {k: v for k, v in results.items() if v}

    completed_tests = []
    for case in tmp_datasets:
        cur_res = []
        for optim_case in opt_flag:
            for method in methods_name:
                if f"{method}_{optim_case}" in results:
                    cur_res.append(set(results[f"{method}_{optim_case}"][case].keys()))
        fname_set = set.intersection(*cur_res)
        completed_tests.extend([(case, fname) for fname in fname_set])

    # set line parameters
    lines = [
        {"width": 4, "style": "-", "color": (0.7, 0.7, 0.7)},
        {"width": 4, "style": "-", "color": (0, 0, 0)},
        {"width": 2.5, "style": "-", "color": (1, 1, 0)},
        {"width": 1.5, "style": "-", "color": (0, 0, 0.4)},
        {"width": 4, "style": ":", "color": (1, 0, 1)},
        {"width": 3, "style": ":", "color": (0, 0.4, 0)},
        {"width": 3, "style": "--", "color": (0, 1, 1)},
        {"width": 2, "style": "--", "color": (0.4, 0, 0)},
        {"width": 4, "style": "-.", "color": (1, 0, 0)},
        {"width": 2, "style": "-.", "color": (0, 0.3, 0.3)},
        {"width": 4, "style": "--", "color": (0.5, 0.5, 0)},
    ]

    # search max values for D-M curves
    max_values = get_max_values(results, completed_tests)

    # fix pretty names for plot output
    pretty_names = dict()
    for x_key in results.keys():
        if "True" in x_key:
            if "most_popular" in x_key:
                pretty_names[x_key] = (
                    x_key.split("_")[0] + "_" + x_key.split("_")[1] + "_" + "optimized"
                )
            else:
                pretty_names[x_key] = x_key.split("_")[0] + "_" + "optimized"
        else:
            if "most_popular" in x_key:
                pretty_names[x_key] = x_key.split("_")[0] + "_" + x_key.split("_")[1]
            else:
                pretty_names[x_key] = x_key.split("_")[0]

    # plot results
    beta_values = np.linspace(1, 3, 10)
    integration_results = dict()
    for n_line, x_key in enumerate(results.keys()):
        tmp_res = [
            dolan_more_function(results[x_key], max_values, cur_metric, beta=i)
            for i in beta_values
        ]
        integration_results[pretty_names[x_key]] = integrate_curve(tmp_res, beta_values)
        plt.plot(
            beta_values,
            tmp_res,
            lines[n_line]["style"],
            color=lines[n_line]["color"],
            linewidth=lines[n_line]["width"],
            label=pretty_names[x_key],
        )
    output_path = Path(
        "results",
        "datasets_statistics",
    )
    plt.xlabel(r"$\beta = Q_{best}\, /\, Q$")
    plt.ylabel(r"fraction of datasets p($\beta$)")
    plt.legend()
    plt.savefig(
        output_path / "Dolan_More_curve.png",
        bbox_inches="tight",
    )

    # sort values and save it as .csv
    dm_values = {
        k: v for k, v in sorted(integration_results.items(), key=lambda item: item[1])
    }

    pd.DataFrame(dm_values.items(), columns=["Methods", cur_metric]).round(3).to_csv(
        output_path / "dm_metric.csv"
    )


if __name__ == "__main__":
    pp_plot()
