import os
import time
import pickle
import signal
from contextlib import contextmanager
from typing import Any


class Log:
    nesting_level = 0
    is_silent = False
    is_method_start = None

    @staticmethod
    def silent(silent: bool):
        Log.is_silent = silent

    @staticmethod
    def print(entry: Any="", nesting: bool=True):
        if Log.is_silent:
            return

        space = "." * (4 * Log.nesting_level) if nesting else ""
        print("{}{}".format(space, entry))

    @staticmethod
    def nest(n: int):
        Log.nesting_level += n

    @staticmethod
    def timeit(method):
        def timed(*args, **kw):
            if not Log.is_method_start:
                Log.print(nesting=False)

            Log.is_method_start = True
            Log.print("Start {}.".format(method.__name__))
            Log.nest(1)

            start_time = time.time()
            result = method(*args, **kw)
            end_time = time.time()

            Log.nest(-1)
            Log.print("End {}. Time: {:0.2f} sec.".format(method.__name__, end_time - start_time))
            Log.is_method_start = False

            return result

        return timed


class Config:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tmp_dir = model_dir
        self.current_time_limit = 0
        self.current_time_limit_start = 0
        self.data = {
            "start_time": time.time(),
            "time_limit": int(os.environ.get("TIME_LIMIT", 5 * 60)),
        }

    def is_train(self) -> bool:
        return self["task"] == "train"

    def is_predict(self) -> bool:
        return self["task"] == "predict"

    def is_regression(self) -> bool:
        return self["mode"] == "regression"

    def is_classification(self) -> bool:
        return self["mode"] == "classification"

    def limit_time_fraction(self, fraction: float=0.1):
        self.current_time_limit = int(self["time_limit"] * fraction)
        self.current_time_limit_start = self.time_left()

    def is_time_fraction_limit(self) -> bool:
        return self.current_time_limit_start - self.time_left() >= self.current_time_limit

    def time_left(self) -> float:
        return self["time_limit"] - (time.time() - self["start_time"])

    def save(self):
        with open(os.path.join(self.model_dir, "config.pkl"), "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(os.path.join(self.model_dir, "config.pkl"), "rb") as f:
            data = pickle.load(f)

        self.data = {**data, **self.data}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    seconds = int(seconds)
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
