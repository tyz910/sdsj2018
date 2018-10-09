import os
import time
import pickle
import pandas as pd
import numpy as np
from lib.util import timeit
from lib.read import read_df
from lib.preprocess import preprocess
from lib.model import train, predict, validate
from typing import Optional


class AutoML:
    def __init__(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.config = {
            "start_time": time.time(),
            "time_limit": int(os.environ.get("TIME_LIMIT", 5 * 60)),
        }

    def train(self, train_csv: str, mode: str):
        self.config["mode"] = mode

        X = read_df(train_csv, self.config)
        y = X["target"]

        preprocess(X, self.config)
        train(X, y, self.config)

    def predict(self, test_csv: str, prediction_csv: str) -> (pd.DataFrame, Optional[np.float64]):
        X = read_df(test_csv, self.config)
        result = X[["line_id"]].copy()

        preprocess(X, self.config)
        result["prediction"] = predict(X, self.config)
        result.to_csv(prediction_csv, index=False)

        target_csv = test_csv.replace("test", "test-target")
        if os.path.exists(target_csv):
            score = validate(result, target_csv, self.config["mode"])
        else:
            score = None

        return result, score

    @timeit
    def save(self):
        with open(os.path.join(self.model_dir, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f, protocol=pickle.HIGHEST_PROTOCOL)

    @timeit
    def load(self):
        with open(os.path.join(self.model_dir, "config.pkl"), "rb") as f:
            config = pickle.load(f)

        self.config = {**config, **self.config}
