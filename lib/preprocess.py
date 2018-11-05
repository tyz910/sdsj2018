import copy
import datetime
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lib.features import select_features
from lib.util import Log, Config
from typing import Optional, List


@Log.timeit
def preprocess(df: pd.DataFrame, config: Config):
    non_negative_target_detect(df, config)
    drop_columns(df, config)
    fillna(df, config)
    to_int8(df, config)

    time_series_detect(df, config)
    feature_selection(df, config)

    subsample(df, config, max_size_mb=2 * 1024)
    transform(df, config)
    subsample(df, config, max_size_mb=2 * 1024)


@Log.timeit
def transform(df: pd.DataFrame, config: Config):
    transform_datetime(df, config)
    transform_categorical(df, config)
    scale(df, config)


@Log.timeit
def drop_columns(df: pd.DataFrame, config: Config):
    df.drop([c for c in ["is_test", "line_id"] if c in df], axis=1, inplace=True)
    drop_constant_columns(df, config)


@Log.timeit
def fillna(df: pd.DataFrame, config: Config):
    for c in [c for c in df if c.startswith("number_")]:
        df[c].fillna(-1, inplace=True)

    for c in [c for c in df if c.startswith("string_")]:
        df[c].fillna("", inplace=True)

    for c in [c for c in df if c.startswith("datetime_")]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)


@Log.timeit
def drop_constant_columns(df: pd.DataFrame, config: Config):
    if "constant_columns" not in config:
        config["constant_columns"] = [c for c in df if c.startswith("number_") and not (df[c] != df[c].iloc[0]).any()]
        Log.print("Constant columns: {}".format(config["constant_columns"]))

    if len(config["constant_columns"]) > 0:
        df.drop(config["constant_columns"], axis=1, inplace=True)


@Log.timeit
def transform_datetime(df: pd.DataFrame, config: Config):
    date_parts = ["year", "weekday", "month", "day", "hour"]

    if "date_columns" not in config:
        config["date_columns"] = {}

        for c in [c for c in df if c.startswith("datetime_")]:
            config["date_columns"][c] = []
            for part in date_parts:
                part_col = c + "_" + part
                df[part_col] = getattr(df[c].dt, part).astype(np.uint16 if part == "year" else np.uint8).values

                if not (df[part_col] != df[part_col].iloc[0]).any():
                    Log.print(part_col + " is constant")
                    df.drop(part_col, axis=1, inplace=True)
                else:
                    config["date_columns"][c].append(part)

            df.drop(c, axis=1, inplace=True)
    else:
        for c, parts in config["date_columns"].items():
            for part in parts:
                part_col = c + "_" + part
                df[part_col] = getattr(df[c].dt, part)
            df.drop(c, axis=1, inplace=True)


@Log.timeit
def transform_categorical(df: pd.DataFrame, config: Config):
    if "categorical_columns" not in config:
        config["categorical_columns"] = []

        # https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
        prior = config["categorical_prior"] = df["target"].mean()
        min_samples_leaf = 10
        smoothing = 5

        config["categorical_columns_string"] = {}
        for c in [c for c in df if c.startswith("string_")]:
            Log.print(c)
            config["categorical_columns"].append(c)

            averages = df[[c, "target"]].groupby(c)["target"].agg(["mean", "count"])
            smooth = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
            averages["target"] = prior * (1 - smooth) + averages["mean"] * smooth
            config["categorical_columns_string"][c] = averages["target"].to_dict()

        config["categorical_columns_id"] = {}
        for c in [c for c in df if c.startswith("id_")]:
            Log.print(c)
            config["categorical_columns"].append(c)

            if df[c].dtype == str or df[c].dtype == object:
                config["categorical_columns_id"][c] = {v: i for i, v in enumerate(df[c].unique())}

    for c, values in config["categorical_columns_string"].items():
        df.loc[:, c] = df[c].apply(lambda x: values[x] if x in values else config["categorical_prior"])

    for c, values in config["categorical_columns_id"].items():
        df.loc[:, c] = df[c].apply(lambda x: values[x] if x in values else -1)


@Log.timeit
def scale(df: pd.DataFrame, config: Config):
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    scale_columns = [c for c in df if c.startswith("number_") and df[c].dtype != np.int8 and c not in config["categorical_columns"]]

    if len(scale_columns) > 0:
        if "scaler" not in config:
            config["scaler"] = StandardScaler(copy=False)
            config["scaler"].fit(df[scale_columns])

        df[scale_columns] = config["scaler"].transform(df[scale_columns])


@Log.timeit
def to_int8(df: pd.DataFrame, config: Config):
    if "int8_columns" not in config:
        config["int8_columns"] = []
        vals = [-1, 0, 1]

        for c in [c for c in df if c.startswith("number_")]:
            if (~df[c].isin(vals)).any():
                continue
            config["int8_columns"].append(c)

        Log.print("Num columns: {}".format(len(config["int8_columns"])))

    if len(config["int8_columns"]) > 0:
        df.loc[:, config["int8_columns"]] = df.loc[:, config["int8_columns"]].astype(np.int8)


@Log.timeit
def subsample(df: pd.DataFrame, config: Config, max_size_mb: float=2.0):
    if config.is_train():
        df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if df_size_mb > max_size_mb:
            mem_per_row = df_size_mb / len(df)
            sample_rows = int(max_size_mb / mem_per_row)

            Log.print("Size limit exceeded: {:0.2f} Mb. Dataset rows: {}. Subsample to {} rows.".format(df_size_mb, len(df), sample_rows))
            _, df_drop = train_test_split(df, train_size=sample_rows, random_state=1)
            df.drop(df_drop.index, inplace=True)

            config["nrows"] = sample_rows
        else:
            config["nrows"] = len(df)


def shift_columns(df: pd.DataFrame, group: Optional[str]=None, number_columns: Optional[List[str]]=None):
    if number_columns is None:
        number_columns = [c for c in df if c.startswith("number_")]
    shift_columns = [c + "_shift" for c in number_columns]

    if group is not None:
        shifted = df.groupby([group])[number_columns].shift(-1)
    else:
        shifted = df[number_columns].shift(-1)

    df[shift_columns] = shifted.fillna(-1)


@Log.timeit
def time_series_detect(df: pd.DataFrame, config: Config):
    sample_size = 10000
    model_params = {
        "objective": "regression" if config["mode"] == "regression" else "binary",
        "metric": "rmse" if config["mode"] == "regression" else "auc",
        "learning_rate": 0.01,
        "verbosity": -1,
        "seed": 1,
        "max_depth": -1,
    }

    if config.is_train():
        datetime_columns = [c for c in df if c.startswith("datetime_")]
        id_columns = [c for c in df if c.startswith("id_")]

        sort_columns = []
        for dc in datetime_columns:
            sort_columns.append([dc])
            for ic in id_columns:
                sort_columns.append([ic, dc])
        else:
            for ic in id_columns:
                sort_columns.append([ic])

        scores = []
        config.limit_time_fraction(0.1)
        for sc in sort_columns:
            if config.is_time_fraction_limit():
                break

            Log.silent(True)
            df.sort_values(sc, inplace=True)

            config_sample = copy.deepcopy(config)
            df_sample = df.iloc[-sample_size:].copy() if len(df) > sample_size else df.copy()
            df_sample = df_sample[[c for c in df_sample if c.startswith("number_") or c == "target" or c in sc]]
            shift_columns(df_sample, group= sc[0] if len(sc) > 1 else None)
            transform(df_sample, config_sample)

            y = df_sample["target"]
            X = df_sample.drop("target", axis=1)
            X_train, X_test, y_train, y_test = ts_split(X, y, test_size=0.5)

            model_sorted = lgb.train(model_params, lgb.Dataset(X_train, label=y_train), 3000, lgb.Dataset(X_test, label=y_test),
                              early_stopping_rounds=100, verbose_eval=False)
            score_sorted = model_sorted.best_score["valid_0"][model_params["metric"]]

            sampled_columns = [c for c in X if "_shift" not in c]
            model_sampled = lgb.train(model_params, lgb.Dataset(X_train[sampled_columns], label=y_train), 3000, lgb.Dataset(X_test[sampled_columns], label=y_test),
                              early_stopping_rounds=100, verbose_eval=False)
            score_sampled = model_sampled.best_score["valid_0"][model_params["metric"]]

            if config.is_classification():
                score_sorted = -score_sorted
                score_sampled = -score_sampled

            Log.silent(False)
            Log.print("Sort: {}. Score sorted: {:0.4f}. Score sampled: {:0.4f}".format(sc, score_sorted, score_sampled))
            score_ratio = score_sampled / score_sorted if config.is_regression() else abs(score_sorted / score_sampled)
            if score_ratio >= 1.03:
                Log.print(score_ratio)
                scores.append((score_sorted, sc))

        if len(scores) > 0:
            scores = sorted(scores, key=lambda x: x[0])
            Log.print("Scores: {}".format(scores))
            config["sort_values"] = scores[0][1]
            df.sort_values(config["sort_values"], inplace=True)

            config_sample = copy.deepcopy(config)
            df_sample = df.iloc[-sample_size:].copy() if len(df) > sample_size else df.copy()
            shift_columns(df_sample, group=config["sort_values"][0] if len(config["sort_values"]) > 1 else None)
            transform(df_sample, config_sample)

            y = df_sample["target"]
            X = df_sample.drop("target", axis=1)

            model = lgb.train(model_params, lgb.Dataset(X, label=y), 1000)
            fi = pd.Series(model.feature_importance(importance_type="gain"), index=X.columns)
            fi = fi[fi > 0].sort_values()
            selected_columns = fi[fi >= fi.quantile(0.75)].index.tolist()

            selected_shift_columns = [c.replace("_shift", "") for c in selected_columns if "_shift" in c]
            if len(selected_shift_columns) > 0:
                Log.print("Shift columns: {}".format(selected_shift_columns))
                config["shift_columns"] = selected_shift_columns

    if "shift_columns" in config:
        shift_columns(df, group=config["sort_values"][0] if len(config["sort_values"]) > 1 else None, number_columns=config["shift_columns"])


@Log.timeit
def feature_selection(df: pd.DataFrame, config: Config):
    if config.is_train():
        df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if df_size_mb < 2 * 1024:
            return

        selected_columns = []
        config_sample = copy.deepcopy(config)
        config.limit_time_fraction(0.1)
        for i in range(20):
            if config.is_time_fraction_limit():
                break

            df_sample = df.sample(min(3000, len(df)), random_state=i).copy()
            transform(df_sample, config_sample)
            y = df_sample["target"]
            X = df_sample.drop("target", axis=1)

            if len(selected_columns) > 0:
                X = X.drop(selected_columns, axis=1)

            if len(X.columns) > 0:
                selected_columns += select_features(X, y, config["mode"])
            else:
                break

        Log.print("Selected columns: {}".format(selected_columns))

        drop_number_columns = [c for c in df if c.startswith("number_") and c not in selected_columns]
        if len(drop_number_columns) > 0:
            config["drop_number_columns"] = drop_number_columns

        config["date_columns"] = {}
        for c in [c for c in selected_columns if c.startswith("datetime_")]:
            d = c.split("_")
            date_col = d[0] + "_" + d[1]
            date_part = d[2]

            if date_col not in config["date_columns"]:
                config["date_columns"][date_col] = []

            config["date_columns"][date_col].append(date_part)

        drop_datetime_columns = [c for c in df if c.startswith("datetime_") and c not in config["date_columns"]]
        if len(drop_datetime_columns) > 0:
            config["drop_datetime_columns"] = drop_datetime_columns

    if "drop_number_columns" in config:
        Log.print("Drop number columns: {}".format(config["drop_number_columns"]))
        df.drop(config["drop_number_columns"], axis=1, inplace=True)

    if "drop_datetime_columns" in config:
        Log.print("Drop datetime columns: {}".format(config["drop_datetime_columns"]))
        df.drop(config["drop_datetime_columns"], axis=1, inplace=True)


@Log.timeit
def non_negative_target_detect(df: pd.DataFrame, config: Config):
    if config.is_train():
        config["non_negative_target"] = df["target"].lt(0).sum() == 0


def ts_split(X: pd.DataFrame, y: pd.Series, test_size: float) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    test_len = int(len(X) * test_size)
    return X[:-test_len], X[-test_len:], y[:-test_len], y[-test_len:]
