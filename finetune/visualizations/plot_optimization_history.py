import numpy as np
import plotly.graph_objs as go


def plot_optimization_history(study_name, trials, direction):
    trials = sorted(trials, key=lambda x: x["trial_number"])
    trial_numbers = [trial["trial_number"] for trial in trials]
    scores = [trial["score"] for trial in trials]
    traces = []

    # scores
    traces.append(
        go.Scatter(x=trial_numbers, y=scores, mode="markers", name="Objective value")
    )

    # accumulate scores
    if direction == "minimize":
        accumulate_scores = list(np.minimum.accumulate(scores))
    else:
        accumulate_scores = list(np.maximum.accumulate(scores))

    traces.append(
        go.Scatter(
            x=trial_numbers, y=accumulate_scores, mode="lines", name="Best value"
        )
    )

    title = study_name + (" - " if len(study_name) else "") + "Score history"
    layout = go.Layout(title=title, xaxis_title="Ticks", yaxis_title="Score")
    figure = go.Figure(data=traces, layout=layout)
    return figure
