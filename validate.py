from lib.automl import AutoML
from lib.util import timeit, log

DATASETS = [
    ("1", "regression"),
    ("2", "regression"),
    #("3", "regression"),
    #("4", "classification"),
    #("5", "classification"),
    #("6", "classification"),
    #("7", "classification"),
    #("8", "classification"),
]


@timeit
def validate_dataset(alias: str, mode: str):
    log(alias)
    automl = AutoML()
    automl.train("data/check_{}/train.csv".format(alias), mode)
    automl.predict("data/check_{}/test.csv".format(alias), "predictions/check_{}.csv".format(alias))
    del automl


for i, mode in DATASETS:
    alias = "{}_{}".format(i, mode[0])
    validate_dataset(alias, mode)
