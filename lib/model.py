import subprocess
import pandas as pd
import numpy as np
import lightgbm as lgb
from vowpalwabbit.sklearn_vw import tovw
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from lib.util import timeit, log, Config
from typing import List


@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config):
    train_lightgbm(X, y, config)


@timeit
def predict(X: pd.DataFrame, config: Config) -> List:
    return predict_lightgbm(X, config)


@timeit
def validate(preds: pd.DataFrame, target_csv: str, mode: str) -> np.float64:
    df = pd.merge(preds, pd.read_csv(target_csv), on="line_id", left_index=True)
    score = roc_auc_score(df.target.values, df.prediction.values) if mode == "classification" else \
        np.sqrt(mean_squared_error(df.target.values, df.prediction.values))
    log("Score: {:0.4f}".format(score))
    return score


@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config):
    params = {
        "objective": "regression" if config["mode"] == "regression" else "binary",
        "metric": "rmse" if config["mode"] == "regression" else "auc",
        "learning_rate": 0.01,
        "num_leaves": 200,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.70,
        "bagging_freq": 4,
        "max_depth": -1,
        "verbosity": -1,
        "reg_alpha": 0.3,
        "reg_lambda": 0.1,
        "min_child_weight": 10,
        "zero_as_missing": True,
        "seed": 1,
    }

    config["model"] = lgb.train(params, lgb.Dataset(X, label=y), 600)


@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    return config["model"].predict(X)


@timeit
def train_vw(X: pd.DataFrame, y: pd.Series, config: Config):
    cache_file = config.tmp_dir + "/.vw_cache"
    data_file = config.tmp_dir + "/vw_data_train.csv"

    cmd = " ".join([
        "rm -f {cache} && vw",
        "-f {f}",
        "--cache_file {cache}",
        "--passes {passes}",
        "-l {l}",
        "--early_terminate {early_terminate}",
        "{df}"
    ]).format(
        cache=cache_file,
        df=data_file,
        f=config.model_dir + "/vw.model",
        passes=max(20, int(1000000/len(X))),
        l=25,
        early_terminate=1,
    )

    if config["mode"] == "classification":
        cmd += " --loss_function logistic --link logistic"
        y = y.replace({0: -1})

    save_to_vw(data_file, X, y)
    subprocess.Popen(cmd, shell=True).communicate()


@timeit
def predict_vw(X: pd.DataFrame, config: Config) -> List:
    preds_file = config.tmp_dir + "/.vw_preds"
    data_file = config.tmp_dir + "/vw_data_test.csv"
    save_to_vw(data_file, X)

    subprocess.Popen("vw -i {i} -p {p} {df}".format(
        df=data_file,
        i=config.model_dir + "/vw.model",
        p=preds_file
    ), shell=True).communicate()

    return [np.float64(line) for line in open(preds_file, "r")]


@timeit
def save_to_vw(filepath: str, X: pd.DataFrame, y: pd.Series=None, chunk_size=1000):
    with open(filepath, "w+") as f:
        for pos in range(0, len(X), chunk_size):
            chunk_X = X.iloc[pos:pos + chunk_size, :]
            chunk_y = y.iloc[pos:pos + chunk_size] if y is not None else None
            for row in tovw(chunk_X, chunk_y):
                f.write(row + "\n")


@timeit
def get_model_predicts(X: pd.DataFrame, config: Config) -> pd.DataFrame:
    models = [
        ("vw", predict_vw(X, config)),
        ("lightgbm", predict_lightgbm(X, config)),
    ]

    return pd.DataFrame({m[0]: m[1] for m in models}, index=X.index, columns=[m[0] for m in models])


@timeit
def train_ensemble(X: pd.DataFrame, y: pd.Series, config: Config):
    predicts = get_model_predicts(X, config)

    model = LinearRegression()
    model.fit(predicts, y)

    for i, m in enumerate(predicts.columns):
        log("{}: {}".format(m, model.coef_[i]))
    log("intercept: {}".format(model.intercept_))

    config["ensemble"] = model


@timeit
def predict_ensemble(X: pd.DataFrame, config: Config) -> List:
    predicts = get_model_predicts(X, config)
    return config["ensemble"].predict(predicts)
