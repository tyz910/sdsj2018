import pandas as pd
import lightgbm as lgb
from boruta import BorutaPy
from typing import List, Optional


class LGBMFeatureEstimator():
    def __init__(self, params, n_estimators: int=50):
        self.params = params
        self.n_estimators = n_estimators

    def get_params(self):
        return self.params

    def set_params(self, n_estimators:Optional[int]=None, random_state:Optional[int]=None):
        if n_estimators is not None:
            self.n_estimators = n_estimators

    def fit(self, X: pd.DataFrame, y: pd.Series):
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(self.params, train_data, self.n_estimators)
        self.feature_importances_ = model.feature_importance(importance_type="gain")


def select_features(X: pd.DataFrame, y: pd.Series, mode: str, n_estimators: int=50, max_iter: int=100, perc: int=75) -> List[str]:
    feat_estimator = LGBMFeatureEstimator({
        "objective": "regression" if mode == "regression" else "binary",
        "metric": "rmse" if mode == "regression" else "auc",
        "learning_rate": 0.01,
        "verbosity": -1,
        "seed": 1,
        "max_depth": -1,
    }, n_estimators)

    feat_selector = BorutaPy(feat_estimator, n_estimators=n_estimators, max_iter=max_iter, verbose=2, random_state=1, perc=perc)

    try:
        feat_selector.fit(X.values, y.values.ravel())
    except TypeError:
        pass

    return X.columns[feat_selector.support_].tolist()
