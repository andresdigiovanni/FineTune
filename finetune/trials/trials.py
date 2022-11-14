import numpy as np


class Trials:
    def __init__(self, objective):
        self._trials = []
        self._objective_maximize = objective == "maximize"

    def get_best_score(self):
        if len(self._trials) == 0:
            return float("-inf") if self._objective_maximize else float("inf")

        return self._trials[0]["score"]

    def get_best_params(self):
        if len(self._trials) == 0:
            return {}

        return self._trials[0]["params"]

    def get_best_trial_number(self):
        if len(self._trials) == 0:
            return -1

        return self._trials[0]["trial_number"]

    def get_best_trials(self):
        if len(self._trials) == 0:
            return []

        scores = [x["score"] for x in self._trials]
        std_dev = np.std(scores)
        threshold = self._trials[0]["score"] - std_dev

        return [x for x in self._trials if x["score"] >= threshold]

    def get_trials(self):
        return self._trials

    def set_trials(self, trials):
        self._trials = trials

    def add_trial(self, params, score):
        trial = {
            "score": score,
            "trial_number": len(self._trials),
            "params": params,
        }

        self._trials.append(trial)
        self._trials = sorted(
            self._trials, key=lambda x: x["score"], reverse=self._objective_maximize
        )
