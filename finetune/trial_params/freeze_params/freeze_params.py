import math
import random


def freeze_params(n_param_grid, p_iter):
    MAX_FREEZE_PARAMS = 0.8

    p_freeze_params = min(p_iter, MAX_FREEZE_PARAMS)
    n_freeze_params = math.floor(n_param_grid * p_freeze_params)

    freeze_params = (
        random.sample(range(n_param_grid), n_freeze_params) if n_freeze_params else []
    )
    return freeze_params
