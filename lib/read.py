import pandas as pd
from lib.util import timeit, log, Config


@timeit
def read_df(csv_path: str, config: Config) -> pd.DataFrame:
    if "dtype" not in config:
        preview_df(csv_path, config)

    df = pandas_read_csv(csv_path, config)
    if config.is_train():
        config["nrows"] = len(df)

    return df


@timeit
def pandas_read_csv(csv_path: str, config: Config) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="utf-8", low_memory=False, dtype=config["dtype"], parse_dates=config["parse_dates"])


@timeit
def preview_df(train_csv: str, config: Config, nrows: int=3000):
    num_rows = sum(1 for line in open(train_csv)) - 1
    log("Rows in train: {}".format(num_rows))

    df = pd.read_csv(train_csv, encoding="utf-8", low_memory=False, nrows=nrows)
    mem_per_row = df.memory_usage(deep=True).sum() / nrows
    log("Memory per row: {:0.2f} Kb".format(mem_per_row / 1024))

    df_size = (num_rows * mem_per_row) / 1024 / 1024
    log("Approximate dataset size: {:0.2f} Mb".format(df_size))

    config["parse_dates"] = []
    config["dtype"] = {
        "line_id": int,
    }

    counters = {
        "id": 0,
        "number": 0,
        "string": 0,
        "datetime": 0,
    }

    for c in df:
        if c.startswith("number_"):
            counters["number"] += 1
        elif c.startswith("string_"):
            counters["string"] += 1
            config["dtype"][c] = str
        elif c.startswith("datetime_"):
            counters["datetime"] += 1
            config["dtype"][c] = str
            config["parse_dates"].append(c)
        elif c.startswith("id_"):
            counters["id"] += 1

    log("Number columns: {}".format(counters["number"]))
    log("String columns: {}".format(counters["string"]))
    log("Datetime columns: {}".format(counters["datetime"]))

    config["counters"] = counters
