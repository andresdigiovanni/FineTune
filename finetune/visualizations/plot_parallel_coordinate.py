import plotly.graph_objs as go


def plot_parallel_coordinate(study_name, param_grid, trials, params=[]):
    params = params if params else param_grid.keys()
    trials = sorted(trials, key=lambda x: x["trial_number"])
    dims = []

    # params columns
    for param in params:
        if type(param_grid[param]) is not tuple:
            continue

        values = [trial["params"][param] for trial in trials]

        if "categorical" in param_grid[param][0]:
            dummies_list = list(set(values))
            dummies_map = dict(zip(dummies_list, list(range(0, len(dummies_list)))))
            dummies_values = [dummies_map[x] for x in values]

            tickvals = list(range(0, len(dummies_list)))

            dims.append(
                {
                    "label": param,
                    "values": dummies_values,
                    "tickvals": tickvals,
                    "ticktext": dummies_list,
                }
            )
        else:
            dims.append(
                {
                    "label": param,
                    "values": values,
                }
            )

    # score column
    dims.append(
        {
            "label": "Score",
            "values": [trial["score"] for trial in trials],
        }
    )

    traces = [
        go.Parcoords(
            dimensions=dims,
            labelangle=30,
            labelside="bottom",
            line={
                "color": dims[-1]["values"],
                "colorbar": {"title": "Objective value"},
                "showscale": True,
            },
        )
    ]

    layout = go.Layout(title=study_name)
    figure = go.Figure(data=traces, layout=layout)
    return figure
