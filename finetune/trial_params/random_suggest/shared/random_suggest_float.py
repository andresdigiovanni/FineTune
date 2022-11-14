import math
import random

import numpy as np


def _min_max(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def _random_suggest_float(min_value, max_value, items=[]):
    STD_DEV_TO_USE_MEAN = 0.2
    MAX_SIGMA = 0.2
    MIN_SIGMA = 0.01

    range_values = max_value - min_value
    std_dev = np.std(items)

    mu = np.mean(items) if std_dev > range_values * STD_DEV_TO_USE_MEAN else items[0]
    sigma = _min_max(std_dev / 2, range_values * MIN_SIGMA, range_values * MAX_SIGMA)

    result = random.gauss(mu, sigma)
    return _min_max(result, min_value, max_value)


def _random_suggest_float_log(min_value, max_value, items):
    min_value = math.log10(min_value)
    max_value = math.log10(max_value)
    items = [math.log10(x) for x in items]

    result = _random_suggest_float(min_value, max_value, items)
    return 10**result


def random_suggest_float(min_value, max_value, items=[], log=False):
    if log:
        return _random_suggest_float_log(min_value, max_value, items)
    else:
        return _random_suggest_float(min_value, max_value, items)
