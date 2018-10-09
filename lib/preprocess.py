import pandas as pd
import datetime
from sklearn import preprocessing
from lib.util import timeit, log
from typing import Dict


@timeit
def preprocess(df: pd.DataFrame, config: Dict):
    drop_columns(df)
    fillna(df, config)
    transform_datetime(df, config)
    transform_categorical(df, config)
    drop_constant_columns(df, config)


@timeit
def drop_columns(df: pd.DataFrame):
    df.drop([c for c in ["is_test", "line_id", "target"] if c in df], axis=1, inplace=True)


@timeit
def fillna(df: pd.DataFrame, config: Dict):
    for c in [c for c in df if c.startswith("number_")]:
        df[c].fillna(config["float_type"](-1), inplace=True)

    for c in [c for c in df if c.startswith("string_")]:
        df[c].fillna("", inplace=True)

    for c in [c for c in df if c.startswith("datetime_")]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)


@timeit
def drop_constant_columns(df: pd.DataFrame, config: Dict):
    if "constant_columns" not in config:
        config["constant_columns"] = [c for c in df if c.startswith("number_") and not (df[c] != df[c].iloc[0]).any()]
        log("Constant columns: " + ", ".join(config["constant_columns"]))

    if len(config["constant_columns"]) > 0:
        df.drop(config["constant_columns"], axis=1, inplace=True)


@timeit
def transform_datetime(df: pd.DataFrame, config: Dict):
    date_parts = ["year", "weekday", "month", "day", "hour"]

    if "date_columns" not in config:
        config["date_columns"] = {}

        for c in [c for c in df if c.startswith("datetime_")]:
            config["date_columns"][c] = []
            for part in date_parts:
                part_col = c + "_" + part
                df[part_col] = getattr(df[c].dt, part)

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
def transform_categorical(df: pd.DataFrame, config: Dict):
    if "categorical_columns" not in config:
        config["categorical_columns"] = {}
        for c in [c for c in df if c.startswith("string_")]:
            config["categorical_columns"][c] = preprocessing.LabelEncoder().fit(df[c])

    for c, encoder in config["categorical_columns"].items():
        df[c] = encoder.transform(df[c])
