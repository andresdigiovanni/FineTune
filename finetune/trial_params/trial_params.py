import math
import random

import numpy as np

from .freeze_params import freeze_params
from .random_suggest import (
    random_suggest_categorical,
    random_suggest_float,
    random_suggest_int,
)


def _suggest_categorical(choices, items, suggest):
    if suggest:
        return random_suggest_categorical(choices, items=items)
    else:
        return random.choice(choices)


def _suggest_float(min_value, max_value, items, suggest, log):
    if suggest:
        return random_suggest_float(min_value, max_value, items=items, log=log)
    elif log:
        min_value = math.log10(min_value)
        max_value = math.log10(max_value)
        result = random.uniform(min_value, max_value)
        return 10**result
    else:
        return random.uniform(min_value, max_value)


def _suggest_int(min_value, max_value, items, suggest, log):
    if suggest:
        return random_suggest_int(min_value, max_value, items=items, log=log)
    elif log:
        return round(np.random.lognormal(min_value, max_value))
    else:
        return round(random.uniform(min_value, max_value))


def get_trial_params(param_grid, trials, p_iter):
    P_EXPLORATION = 0.15

    suggest = p_iter > P_EXPLORATION

    params = {}
    fix_params = [x for x in param_grid if not type(param_grid[x]) is tuple]

    n_params = len(param_grid) - len(fix_params)
    freezed_params = freeze_params(n_params, p_iter)

    i = -1
    for key in param_grid.keys():
        if key in fix_params:
            params[key] = param_grid[key]
            continue

        i += 1
        items = [d["params"][key] for d in trials]
        variable_type = param_grid[key][0]

        if i in freezed_params:
            params[key] = trials[0]["params"][key]

        elif variable_type == "categorical":
            params[key] = _suggest_categorical(param_grid[key][1], items, suggest)

        elif "float" in variable_type:
            params[key] = _suggest_float(
                param_grid[key][1],
                param_grid[key][2],
                items,
                suggest,
                log=("log" in variable_type),
            )

        elif "int" in variable_type:
            params[key] = _suggest_int(
                param_grid[key][1],
                param_grid[key][2],
                items,
                suggest,
                log=("log" in variable_type),
            )

    return params
