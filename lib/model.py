import subprocess
import pandas as pd
import numpy as np
import lightgbm as lgb
import h2o
from h2o.automl import H2OAutoML
import hyperopt
from hyperopt import hp, tpe, STATUS_OK, space_eval
from vowpalwabbit.sklearn_vw import tovw
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
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
def train_lm(X: pd.DataFrame, y: pd.Series, config: Config):
    if config["mode"] == "regression":
        model = Ridge()
    else:
        model = LogisticRegression(solver="liblinear")

    config["model_lm"] = model.fit(X, y)


@timeit
def predict_lm(X: pd.DataFrame, config: Config) -> List:
    if config["mode"] == "regression":
        return config["model_lm"].predict(X)
    else:
        return config["model_lm"].predict_proba(X)[:, 1]


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


@timeit
def train_h2o(X: pd.DataFrame, y: pd.Series, config: Config):
    h2o.init()

    X["target"] = y
    train = h2o.H2OFrame(X)
    train_x = train.columns
    train_y = "target"
    train_x.remove(train_y)

    if config["mode"] == "classification":
        train[train_y] = train[train_y].asfactor()

    aml = H2OAutoML(max_runtime_secs=60)
    aml.train(x=train_x, y=train_y, training_frame=train)

    config["model_h2o"] = h2o.save_model(model=aml.leader, path=config.model_dir + "/h2o.model", force=True)
    print(aml.leaderboard)

    X.drop("target", axis=1, inplace=True)


@timeit
def predict_h2o(X: pd.DataFrame, config: Config) -> List:
    h2o.init()
    model = h2o.load_model(config["model_h2o"])

    return model.predict(h2o.H2OFrame(X)).as_data_frame()["predict"].tolist()


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