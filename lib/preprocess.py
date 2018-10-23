import copy
import datetime
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lib.features import select_features
from lib.util import timeit, log, Config


@timeit
def preprocess(df: pd.DataFrame, config: Config):
    feature_selection(df, config)
    preprocess_pipeline(df, config)


def preprocess_pipeline(df: pd.DataFrame, config: Config):
    if leak_detect(df, config):
        return

    drop_columns(df, config)
    fillna(df, config)
    to_int8(df, config)
    non_negative_target_detect(df, config)
    subsample(df, config, max_size_mb=2 * 1024)

    transform_datetime(df, config)
    transform_categorical(df, config)
    scale(df, config)
    subsample(df, config, max_size_mb=2 * 1024)


@timeit
def drop_columns(df: pd.DataFrame, config: Config):
    df.drop([c for c in ["is_test", "line_id"] if c in df], axis=1, inplace=True)
    drop_constant_columns(df, config)


@timeit
def fillna(df: pd.DataFrame, config: Config):
    for c in [c for c in df if c.startswith("number_")]:
        df[c].fillna(-1, inplace=True)

    for c in [c for c in df if c.startswith("string_")]:
        df[c].fillna("", inplace=True)

    for c in [c for c in df if c.startswith("datetime_")]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)


@timeit
def drop_constant_columns(df: pd.DataFrame, config: Config):
    if "constant_columns" not in config:
        config["constant_columns"] = [c for c in df if c.startswith("number_") and not (df[c] != df[c].iloc[0]).any()]
        log("Constant columns: " + ", ".join(config["constant_columns"]))

    if len(config["constant_columns"]) > 0:
        df.drop(config["constant_columns"], axis=1, inplace=True)


@timeit
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
                    log(part_col + " is constant")
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


@timeit
def transform_categorical(df: pd.DataFrame, config: Config):
    if "categorical_columns" not in config:
        # https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
        prior = config["categorical_prior"] = df["target"].mean()
        min_samples_leaf = int(0.01 * len(df))
        smoothing = 0.5 * min_samples_leaf

        config["categorical_columns"] = {}
        for c in [c for c in df if c.startswith("string_")]:
            averages = df[[c, "target"]].groupby(c)["target"].agg(["mean", "count"])
            smooth = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
            averages["target"] = prior * (1 - smooth) + averages["mean"] * smooth
            config["categorical_columns"][c] = averages["target"].to_dict()

        log(list(config["categorical_columns"].keys()))

    for c, values in config["categorical_columns"].items():
        df.loc[:, c] = df[c].apply(lambda x: values[x] if x in values else config["categorical_prior"])


@timeit
def scale(df: pd.DataFrame, config: Config):
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    scale_columns = [c for c in df if c.startswith("number_") and df[c].dtype != np.int8 and
                     c not in config["categorical_columns"]]

    if len(scale_columns) > 0:
        if "scaler" not in config:
            config["scaler"] = StandardScaler(copy=False)
            config["scaler"].fit(df[scale_columns])

        df[scale_columns] = config["scaler"].transform(df[scale_columns])


@timeit
def to_int8(df: pd.DataFrame, config: Config):
    if "int8_columns" not in config:
        config["int8_columns"] = []
        vals = [-1, 0, 1]

        for c in [c for c in df if c.startswith("number_")]:
            if (~df[c].isin(vals)).any():
                continue
            config["int8_columns"].append(c)

        log(config["int8_columns"])

    if len(config["int8_columns"]) > 0:
        df.loc[:, config["int8_columns"]] = df.loc[:, config["int8_columns"]].astype(np.int8)


@timeit
def subsample(df: pd.DataFrame, config: Config, max_size_mb: float=2.0):
    if config.is_train():
        df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if df_size_mb > max_size_mb:
            mem_per_row = df_size_mb / len(df)
            sample_rows = int(max_size_mb / mem_per_row)

            log("Size limit exceeded: {:0.2f} Mb. Dataset rows: {}. Subsample to {} rows.".format(df_size_mb, len(df), sample_rows))
            _, df_drop = train_test_split(df, train_size=sample_rows, random_state=1)
            df.drop(df_drop.index, inplace=True)

            config["nrows"] = sample_rows
        else:
            config["nrows"] = len(df)


@timeit
def non_negative_target_detect(df: pd.DataFrame, config: Config):
    if config.is_train():
        config["non_negative_target"] = df["target"].lt(0).sum() == 0


@timeit
def feature_selection(df: pd.DataFrame, config: Config):
    if config.is_train():
        df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if df_size_mb < 2 * 1024:
            return

        selected_columns = []
        config_sample = copy.deepcopy(config)
        for i in range(10):
            df_sample = df.sample(min(1000, len(df)), random_state=i).copy(deep=True)
            preprocess_pipeline(df_sample, config_sample)
            y = df_sample["target"]
            X = df_sample.drop("target", axis=1)

            if len(selected_columns) > 0:
                X = X.drop(selected_columns, axis=1)

            if len(X.columns) > 0:
                selected_columns += select_features(X, y, config["mode"])
            else:
                break

        log("Selected columns: {}".format(selected_columns))

        drop_number_columns = [c for c in df if (c.startswith("number_") or c.startswith("id_")) and c not in selected_columns]
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
        log("Drop number columns: {}".format(config["drop_number_columns"]))
        df.drop(config["drop_number_columns"], axis=1, inplace=True)

    if "drop_datetime_columns" in config:
        log("Drop datetime columns: {}".format(config["drop_datetime_columns"]))
        df.drop(config["drop_datetime_columns"], axis=1, inplace=True)


@timeit
# https://github.com/bagxi/sdsj2018_lightgbm_baseline
# https://forum-sdsj.datasouls.com/t/topic/304/3
def leak_detect(df: pd.DataFrame, config: Config) -> bool:
    if config.is_predict():
        return "leak" in config

    id_cols = [c for c in df if c.startswith('id_')]
    dt_cols = [c for c in df if c.startswith('datetime_')]

    if id_cols and dt_cols:
        num_cols = [c for c in df if c.startswith('number_')]
        for id_col in id_cols:
            group = df.groupby(by=id_col).get_group(df[id_col].iloc[0])

            for dt_col in dt_cols:
                sorted_group = group.sort_values(dt_col)

                for lag in range(-1, -10, -1):
                    for col in num_cols:
                        corr = sorted_group['target'].corr(sorted_group[col].shift(lag))
                        if corr >= 0.99:
                            config["leak"] = {
                                "num_col": col,
                                "lag": lag,
                                "id_col": id_col,
                                "dt_col": dt_col,
                            }
                            return True

    return False
