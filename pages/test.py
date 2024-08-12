import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import html, dcc, callback, Input, Output, State

dash.register_page(__name__, path="/test")

resolution = 1000
t = np.linspace(0, np.pi * 2, resolution)
x, y = np.cos(t), np.sin(t)

# Example app.
figure = dict(
    data=[{"x": [], "y": []}],
    layout=dict(xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1])),
)  # remove "Updating..." from title

layout = html.Div(
    [
        dcc.Graph(id="graph", figure=figure),
        dcc.Interval(id="interval", interval=25),
        dcc.Store(id="offset", data=0),
        dcc.Store(id="store", data=dict(x=x, y=y, resolution=resolution)),
    ]
)


@callback(
    [
        Output(component_id="graph", component_property="extendData"),
        Output(component_id="offset", component_property="data"),
    ],
    Input(component_id="interval", component_property="n_intervals"),
    [
        State(component_id="store", component_property="data"),
        State(component_id="offset", component_property="data"),
    ],
)
def update_graph(n_intervals, data, offset):
    offset = offset % len(data["x"])
    end = min(offset + 10, len(data["x"]))
    return [
        [{"x": [data["x"][offset:end]], "y": [data["y"][offset:end]]}, [0], 500],
        end,
    ]
