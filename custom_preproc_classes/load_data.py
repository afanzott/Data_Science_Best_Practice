import pandas as pd

from .config.core import config


def data_loading(path_features, path_target) -> pd.DataFrame:
    """
    This function takes a path to the features .csv and the target .csv and creates a joined table after necessesary
    preprocessing steps.

    Args:
        path_features: str containing the path to the features data .csv
        path_target: str containing the path to the target data .csv
    """
    # load data
    data_features = pd.read_csv(path_features, sep=";", index_col=False)
    target = pd.read_csv(path_target, sep=";")

    # columns to rename
    data_features.rename(columns=config["var_to_rename"][0], inplace=True)

    # columns to cast as Int64
    column_as_Int64 = config["feat_to_int"]
    data_features[column_as_Int64] = data_features[column_as_Int64].astype(
        "int64", errors="ignore")

    # merge both tables wrt groups & index
    data_features = data_features.loc[~data_features.groups.isnull(), :]
    data = pd.merge(data_features, target, how="inner", on=["groups", "index"])

    # cast etherium as float64
    data[config["feat_to_numeric"]] = pd.to_numeric(
        data[config["feat_to_numeric"]], errors="coerce")
    data[config["feat_to_numeric"]] = pd.to_numeric(
        data[config["feat_to_numeric"]], errors="coerce")

    # cast all categorical variables as categorical
    data[config["cat_vars"]] = data[config["cat_vars"]].astype('O')
    data[config["cat_vars"]] = data[config["cat_vars"]].astype('O')

    return data


def data_loading_pred(data_file: str) -> pd.DataFrame:
    """
    This function takes a path to the data file .csv wanted for prediction and creates a pandas dataframe, which can be consumed
    by a prediction model.

    Args:
        data_file: path str to the file which should be predicted
    """
    # load data
    data = pd.read_csv(data_file, sep=";", index_col=False)
    # columns to rename
    data.rename(columns=config["var_to_rename"][0], inplace=True)

    # columns to cast as Int64
    column_as_Int64 = config["feat_to_int"]
    data[column_as_Int64] = data[column_as_Int64].astype(
        "int64", errors="ignore")

    # merge both tables wrt groups & index
    data = data.loc[~data.groups.isnull(), :]

    # cast etherium as float64
    data[config["feat_to_numeric"]] = pd.to_numeric(
        data[config["feat_to_numeric"]], errors="coerce")
    data[config["feat_to_numeric"]] = pd.to_numeric(
        data[config["feat_to_numeric"]], errors="coerce")

    # cast all categorical variables as categorical
    data[config["cat_vars"]] = data[config["cat_vars"]].astype('O')
    data[config["cat_vars"]] = data[config["cat_vars"]].astype('O')

    return data
