import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, roc_auc_score
from lib.util import timeit, log
from typing import Dict, List


@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Dict):
    train_lightgbm(X, y, config)


@timeit
def predict(X: pd.DataFrame, config: Dict) -> List:
    return predict_lightgbm(X, config)


@timeit
def validate(preds: pd.DataFrame, target_csv: str, mode: str):
    df = pd.merge(preds, pd.read_csv(target_csv), on="line_id", left_index=True)
    score = roc_auc_score(df.target.values, df.prediction.values) if mode == "classification" else \
        np.sqrt(mean_squared_error(df.target.values, df.prediction.values))
    log("Score: {:0.4f}".format(score))


@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Dict):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression' if config["mode"] == 'regression' else 'binary',
        'metric': 'rmse',
        "learning_rate": 0.01,
        "num_leaves": 200,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.70,
        'bagging_freq': 4,
        "max_depth": -1,
        "verbosity" : -1,
        "reg_alpha": 0.3,
        "reg_lambda": 0.1,
        "min_child_weight":10,
        'zero_as_missing':True,
        'num_threads': 4,
        'seed': 1,
    }

    config['model'] = lgb.train(params, lgb.Dataset(X, label=y), 600)


@timeit
def predict_lightgbm(X: pd.DataFrame, config: Dict) -> List:
    return config['model'].predict(X)
