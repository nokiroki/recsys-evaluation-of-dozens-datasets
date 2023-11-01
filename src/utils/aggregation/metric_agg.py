from typing import Optional, Union
from pathlib import Path
from itertools import filterfalse

import pandas as pd


def read_transform_dataset(path: Path, multirun: bool = True) -> pd.DataFrame:
    id_cols = ["Run_id", 'k'] if multirun else ['k']
    dataframe = pd.read_csv(path, index_col=id_cols if multirun else 0)
    dataframe.index.names = id_cols
    dataframe.reset_index(inplace=True)
    dataframe.rename(
        columns={name: name.split('@')[0].lower() for name in dataframe.columns},
        inplace=True
    )

    return pd.melt(
        dataframe,
        id_vars=id_cols,
        value_vars=dataframe.columns.drop(id_cols),
        var_name="Metric",
        value_name="Value"
    )


def aggregate_dataset(
        path: Path,
        multirun: bool = False,
        skip_datasets: tuple[str] = (
            "sber_bank", "twitch", "yoochoose", "xwines", "yahoo", "tmall",
            "lfm", "amazon_books", "hotelrec", "otto"
        )
) -> pd.DataFrame:
    columns = ["Dataset", "Metric", 'k', "Value"]
    if multirun:
        columns.append("Run_id")
    res_df = pd.DataFrame(columns=columns)
    for dataset in filterfalse(lambda name: name.stem in skip_datasets, path.iterdir()):
        for filename in dataset.iterdir():
            if (
                multirun and "nfold" not in filename.stem or
                not multirun and "nfold" in filename.stem
            ):
                continue
            if multirun:
                raise NotImplementedError
            if filename.stem in ("results", "results_wasOptimized_True"):
                file_path = filename
                break
            if filename.stem == "results_wasOptimized_Fale":
                file_path = filename
        res_df_file = read_transform_dataset(file_path, multirun)
        res_df_file['Dataset'] = dataset.stem
        res_df = pd.concat([res_df, res_df_file], ignore_index=True)
    return res_df


def aggregate_methods(
        metric_path: Path,
        methods_with_subfolders: tuple[str] = ("implicit", "recbole", "replay", "msrec"),
        stop_methods: tuple[str] = ("replay", "user_knn"),
        multirun: bool = False,
) -> pd.DataFrame:
    columns = ["Method", "Dataset", "Metric", "k", "Value"]
    if multirun:
        columns.append("Run_id")
    metrics_df = pd.DataFrame(columns=columns)

    for method in filterfalse(lambda name: name.stem in stop_methods, metric_path.iterdir()):
        if method.stem in methods_with_subfolders:
            for sub_method in method.iterdir():
                res_df_method = aggregate_dataset(sub_method, multirun)
                res_df_method["Method"] = '_'.join(
                    (sub_method.absolute().parent.stem, sub_method.stem)
                )
                metrics_df = pd.concat([metrics_df, res_df_method])
        else:
            res_df_method = aggregate_dataset(method, multirun)
            res_df_method["Method"] = method.stem
            metrics_df = pd.concat([metrics_df, res_df_method])

    return metrics_df.reset_index(drop=True)


def filter_data(
    df: pd.DataFrame, 
    metric_save: Optional[Union[str, list[str]]] = None,
    k_save: Optional[Union[int, list[int]]] = None,
    drop_minor_methods: bool = False,
    drop_minor_datasets: bool = False
) -> pd.DataFrame:

    drop_columns = []

    if metric_save is not None:
        if not isinstance(metric_save, list):
            metric_save = [metric_save]
        print(list(map(lambda s: s.lower(), metric_save)))
        df = df[df["Metric"].isin(map(lambda s: s.lower(), metric_save))]
        drop_columns.append("Metric")
    
    if k_save is not None:
        if not isinstance(k_save, list):
            k_save = [k_save]
        df = df[df['k'].isin(k_save)]
        drop_columns.append('k')

    if drop_minor_methods:
        max_instances_method = df["Method"].value_counts()[0]
        df = df[df["Method"] == max_instances_method]

    if drop_minor_datasets:
        val_cnts = df["Dataset"].value_counts()
        max_instances_dataset = val_cnts[0]
        saved_datasets = val_cnts[val_cnts == max_instances_dataset].index
        df = df[df["Dataset"].isin(saved_datasets)]


    df = df.drop(columns=drop_columns)
    return df


if __name__ == "__main__":
    path = Path("amazon_cds/results_nfold_100_wasOptimized_True.csv")
    print(read_transform_dataset(path, True))
