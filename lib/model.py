import pandas as pd
import numpy as np
import lightgbm as lgb
import hyperopt
from hyperopt import hp, tpe, STATUS_OK, space_eval, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from lib.util import Log, Config, time_limit, TimeoutException
from typing import List, Dict


@Log.timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config):
    train_lightgbm(X, y, config)


@Log.timeit
def predict(X: pd.DataFrame, config: Config) -> List:
    preds = predict_lightgbm(X, config)

    if config["non_negative_target"]:
        preds = [max(0, p) for p in preds]

    return preds

@Log.timeit
def validate(preds: pd.DataFrame, target_csv: str, mode: str) -> np.float64:
    df = pd.merge(preds, pd.read_csv(target_csv), on="line_id", left_index=True)
    score = roc_auc_score(df.target.values, df.prediction.values) if mode == "classification" else \
        np.sqrt(mean_squared_error(df.target.values, df.prediction.values))
    Log.print("Score: {:0.4f}".format(score))
    return score


@Log.timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config):
    params = {
        "objective": "regression" if config.is_regression() else "binary",
        "metric": "rmse" if config.is_regression() else "auc",
        "verbosity": -1,
        "seed": 1,
    }

    X_sample, y_sample = data_sample(X, y, config, nrows=20000)
    hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, config)

    X_train, X_val, y_train, y_val = data_split(X, y, config)

    config["model"] = lgb.train(
        {**params, **hyperparams},
        lgb.Dataset(X_train, label=y_train),
        5000,
        lgb.Dataset(X_val, label=y_val),
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    config.save()

    try:
        with time_limit(config.time_left() - 10):
            config["model"] = lgb.train(
                {**params, **hyperparams},
                lgb.Dataset(X, label=y),
                int(1.2 * config["model"].best_iteration),
            )
    except TimeoutException:
        Log.print("Timed out!")


@Log.timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    return config["model"].predict(X)


@Log.timeit
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    X_train, X_val, y_train, y_val = data_split(X, y, config, test_size=0.5)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.choice("learning_rate", np.arange(0.01, 0.05, 0.01)),
        "boost_from_average": hp.choice("boost_from_average", [True, False]),
        "is_unbalance": hp.choice("is_unbalance", [True, False]),
        "zero_as_missing": hp.choice("zero_as_missing", [True, False]),
        "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6, 7]),
        "num_leaves": hp.choice("num_leaves", [11, 31, 51, 101, 151, 201]),
        "feature_fraction": hp.choice("feature_fraction", np.arange(0.5, 1.0, 0.1)),
        "bagging_fraction": hp.choice("bagging_fraction", np.arange(0.5, 1.0, 0.1)),
        "bagging_freq": hp.choice("bagging_freq", [1, 3, 5, 10, 20, 50]),
        "reg_alpha": hp.uniform("reg_alpha", 0, 10),
        "reg_lambda": hp.uniform("reg_lambda", 0, 10),
        "min_child_weight": hp.uniform("min_child_weight", 0, 10),
    }

    config.limit_time_fraction(0.15)

    def objective(hyperparams):
        if config.is_time_fraction_limit():
            score = np.inf if config.is_regression() else 0
            return {'loss': score, 'status': STATUS_OK}

        model = lgb.train({**params, **hyperparams}, train_data, 300, valid_data,
                          early_stopping_rounds=100, verbose_eval=False)

        score = model.best_score["valid_0"][params["metric"]]
        Log.print(score)
        if config.is_classification():
            score = -score

        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials, algo=tpe.suggest, max_evals=100, verbose=1,
                         rstate= np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    Log.print("{:0.4f} {}".format(trials.best_trial['result']['loss'], hyperparams))
    return hyperparams


def ts_split(X: pd.DataFrame, y: pd.Series, test_size: float) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    test_len = int(len(X) * test_size)
    return X[:-test_len], X[-test_len:], y[:-test_len], y[-test_len:]


def data_split(X: pd.DataFrame, y: pd.Series, config: Config, test_size: float=0.2) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    if "sort_values" in config:
        return ts_split(X, y, test_size=test_size)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=1)


def data_sample(X: pd.DataFrame, y: pd.Series, config: Config, nrows: int=10000) -> (pd.DataFrame, pd.Series):
    if len(X) > nrows:
        if "sort_values" in config:
            X_sample = X.iloc[:nrows]
            y_sample = y.iloc[:nrows]
        else:
            X_sample = X.sample(nrows, random_state=1)
            y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample
