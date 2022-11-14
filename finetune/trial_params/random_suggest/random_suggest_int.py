from .shared import random_suggest_float as rnd


def random_suggest_int(min_value, max_value, items=[], log=False):
    return round(rnd(min_value, max_value, items, log))
