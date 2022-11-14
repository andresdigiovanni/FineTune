import pickle

from .trial_params import get_trial_params
from .trials import Trials
from .visualizations import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_parameter_history,
)


class Finetune:
    def __init__(self, param_grid, objective, study_name=""):
        self.study_name = study_name
        self.param_grid = param_grid
        self.objective = objective

        self.trials = Trials(objective)

    def get_best_score(self):
        return self.trials.get_best_score()

    def get_best_params(self):
        return self.trials.get_best_params()

    def get_best_trial_number(self):
        return self.trials.get_best_trial_number()

    def get_trials(self):
        return self.trials.get_trials()

    def dump(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.trials.get_trials(), f)

    def load(self, path):
        with open(path, "rb") as f:
            trials = pickle.load(f)

        self.trials.set_trials(trials)

    def plot_optimization_history(self):
        fig = plot_optimization_history(
            self.study_name, self.trials.get_trials(), self.objective
        )
        fig.show()

    def plot_parallel_coordinate(self, params=[]):
        fig = plot_parallel_coordinate(
            self.study_name, self.param_grid, self.trials.get_trials(), params
        )
        fig.show()

    def plot_parameter_history(self, param):
        fig = plot_parameter_history(
            self.study_name, self.param_grid, self.trials.get_trials(), param
        )
        fig.show()

    def optimize(self, func, n_trials):
        print(f"Starting {self.study_name}")

        for i in range(0, n_trials):
            p_iter = i / n_trials

            params = get_trial_params(
                self.param_grid, self.trials.get_best_trials(), p_iter
            )
            score = func(params)
            self.trials.add_trial(params, score)

            print(
                f"\nTrial {i} ({round(p_iter*100)}%)"
                + f" finished with value: {score}"
                + f" and parameters: {params}."
                + f" Best is trial {self.trials.get_best_trial_number()}"
                + f" with value: {self.trials.get_best_score()}."
            )
