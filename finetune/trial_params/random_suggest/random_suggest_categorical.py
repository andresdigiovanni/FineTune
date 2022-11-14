import random

import pandas as pd


def random_suggest_categorical(choices, items):
    MIN_PCT_POPULATION_NOT_REPRESENTED = 0.05
    MIN_PCT_POPULATION_NOT_REPRESENTED /= 1 - MIN_PCT_POPULATION_NOT_REPRESENTED

    counts = pd.Series(items).value_counts()
    population = list(counts.keys())
    population_not_represented = [x for x in choices if x not in population]
    weights = [x / len(items) for x in counts]

    for x in population_not_represented:
        population.append(x)
        weights.append(
            MIN_PCT_POPULATION_NOT_REPRESENTED / len(population_not_represented)
        )

    return random.choices(population, weights)[0]
