import pandas as pd
import numpy as np
import lightgbm as lgb
import hyperopt
from hyperopt import hp, tpe, STATUS_OK, space_eval
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from lib.util import timeit, log, Config
from typing import List, Dict


@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config):
    train_lightgbm(X, y, config)


@timeit
def predict(X: pd.DataFrame, config: Config) -> List:
    preds = predict_lightgbm(X, config)
    if config["non_negative_target"]:
        preds = [max(0, p) for p in preds]

    return preds


@timeit
def validate(preds: pd.DataFrame, target_csv: str, mode: str) -> np.float64:
    df = pd.merge(preds, pd.read_csv(target_csv), on="line_id", left_index=True)
    score = roc_auc_score(df.target.values, df.prediction.values) if mode == "classification" else \
        np.sqrt(mean_squared_error(df.target.values, df.prediction.values))
    log("Score: {:0.4f}".format(score))
    return score


@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config, use_hyperopt: bool=False):
    params = {
        "objective": "regression" if config["mode"] == "regression" else "binary",
        "metric": "rmse" if config["mode"] == "regression" else "auc",
        "learning_rate": 0.01,
        "verbosity": -1,
        "seed": 1,
    }

    if use_hyperopt:
        X_sample, y_sample = data_sample(X, y, config)
        hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, config)

        X_train, X_val, y_train, y_val = data_split(X, y, config)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        config["model"] = lgb.train({**params, **hyperparams, "early_stopping_round": 20}, train_data, 3000, valid_data)
    else:
        hyperparams = {
            "num_leaves": 200,
            "feature_fraction": 0.70,
            "bagging_fraction": 0.70,
            "bagging_freq": 4,
            "max_depth": -1,
            "reg_alpha": 0.3,
            "reg_lambda": 0.1,
            "min_child_weight": 10,
            "zero_as_missing": True,
        }

        train_data = lgb.Dataset(X, label=y)
        config["model"] = lgb.train({**params, **hyperparams}, train_data, 600)


@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    return config["model"].predict(X)


@timeit
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    X_train, X_val, y_train, y_val = data_split(X, y, config)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.05),
        "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
        "reg_alpha": hp.uniform("reg_alpha", 0, 30),
        "reg_lambda": hp.uniform("reg_lambda", 0, 30),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
    }

    def objective_score(model):
        predict = model.predict(X_val)
        return -roc_auc_score(y_val.values, predict) if config["mode"] == "classification" else \
            np.sqrt(mean_squared_error(y_val.values, predict))

    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams, "early_stopping_round": 3}, train_data, 3000, valid_data)
        score = objective_score(model)

        return {'loss': score, 'status': STATUS_OK}

    best = hyperopt.fmin(fn=objective,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=50,
                        verbose=1)

    return space_eval(space, best)


def ts_split(X: pd.DataFrame, y: pd.Series, test_size: float) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    test_len = int(len(X) * test_size)
    return X[:-test_len], X[-test_len:], y[:-test_len], y[-test_len:]


def data_split(X: pd.DataFrame, y: pd.Series, config: Config, test_size: float=0.2) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    if config["time_series"]:
        return ts_split(X, y, test_size=test_size)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=1)


def data_sample(X: pd.DataFrame, y: pd.Series, config: Config, nrows: int=5000) -> (pd.DataFrame, pd.Series):
    if len(X) > nrows:
        if config["time_series"]:
            X_sample = X.iloc[:nrows]
            y_sample = y.iloc[:nrows]
        else:
            X_sample = X.sample(nrows)
            y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample
