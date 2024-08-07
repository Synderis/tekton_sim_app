import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objs as go
import pandas as pd


dash.register_page(__name__, path="/temp_data-dashboard")

# Sample data
df = pd.DataFrame(
    {
        "Month": [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        "Sales": [150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
        "Profit": [50, 70, 80, 90, 100, 130, 150, 170, 180, 200, 220, 250],
    }
)


layout = html.Div(
    children=[
        html.H1(children="Sample Plotly Dashboard"),
        html.Div(
            children="""
        A simple dashboard with a line chart and a bar chart.
    """
        ),
        dcc.Graph(
            id="line-chart",
            figure={
                "data": [
                    go.Scatter(
                        x=df["Month"], y=df["Sales"], mode="lines+markers", name="Sales"
                    )
                ],
                "layout": go.Layout(
                    title="Monthly Sales",
                    xaxis={"title": "Month"},
                    yaxis={"title": "Sales"},
                ),
            },
        ),
        dcc.Graph(
            id="bar-chart",
            figure={
                "data": [go.Bar(x=df["Month"], y=df["Profit"], name="Profit")],
                "layout": go.Layout(
                    title="Monthly Profit",
                    xaxis={"title": "Month"},
                    yaxis={"title": "Profit"},
                ),
            },
        ),
    ]
)
