import pandas as pd
import numpy as np
import time
from lib.automl import AutoML
from lib.util import timeit, log

DATASETS = [
    ("1", "regression", 300),
    ("2", "regression", 300),
    ("3", "regression", 300),
    ("4", "classification", 300),
    ("5", "classification", 300),
    ("6", "classification", 600),
    ("7", "classification", 1800),
    ("8", "classification", 1800),
]


@timeit
def validate_dataset(alias: str, mode: str, train_limit: int) -> np.float64:
    log(alias)

    automl = AutoML("models/check_{}".format(alias))

    automl.config["time_limit"] = train_limit
    automl.train("data/check_{}/train.csv".format(alias), mode)

    automl.config["time_limit"] = 300
    _, score = automl.predict("data/check_{}/test.csv".format(alias), "predictions/check_{}.csv".format(alias))

    return score


if __name__ == '__main__':
    scores = {
        "dataset": [],
        "score": [],
        "time": [],
    }

    for i, mode, train_limit in DATASETS:
        alias = "{}_{}".format(i, mode[0])

        start_time = time.time()
        score = validate_dataset(alias, mode, train_limit)
        end_time = time.time()

        scores["dataset"].append(alias)
        scores["score"].append(score)
        scores["time"].append(end_time - start_time)

    scores = pd.DataFrame(scores)
    scores.to_csv("scores/{}.csv".format(int(time.time())))
    print(scores)
