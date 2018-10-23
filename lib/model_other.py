import subprocess
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from vowpalwabbit.sklearn_vw import tovw
from sklearn.linear_model import LogisticRegression, Ridge
from lib.util import timeit, Config
from typing import List


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
