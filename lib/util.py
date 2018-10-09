import time
from typing import Any

nesting_level = 0
is_start = None


def timeit(method):
    def timed(*args, **kw):
        global is_start
        global nesting_level

        if not is_start:
            print()

        is_start = True
        log("Start {}.".format(method.__name__))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log("End {}. Time: {:0.2f} sec.".format(method.__name__, end_time - start_time))
        is_start = False

        return result

    return timed


def log(entry: Any):
    global nesting_level
    space = " " * (4 * nesting_level)
    print("{}{}".format(space, entry))
