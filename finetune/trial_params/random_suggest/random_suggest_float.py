from .shared import random_suggest_float as rnd


def random_suggest_float(min_value, max_value, items=[], log=False):
    return rnd(min_value, max_value, items, log)
