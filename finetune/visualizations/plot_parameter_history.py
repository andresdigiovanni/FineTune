import plotly.graph_objs as go


def plot_parameter_history(study_name, param_grid, trials, param):
    trials = sorted(trials, key=lambda x: x["trial_number"])
    trial_numbers = [trial["trial_number"] for trial in trials]
    values = [trial["params"][param] for trial in trials]

    title = study_name + (" - " if len(study_name) else "") + param + " history"
    layout = go.Layout(title=title, xaxis_title="Ticks", yaxis_title="Value")
    traces = go.Scatter(
        x=trial_numbers, y=values, mode="markers", name="Objective value"
    )
    figure = go.Figure(data=traces, layout=layout)

    if "log" in param_grid[param][0]:
        figure.update_yaxes(type="log")

    return figure
