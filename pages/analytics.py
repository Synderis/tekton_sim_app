import dash
from dash import html, dcc, callback, Input, Output


dash.register_page(__name__, path="/analytics-dashboard")

layout = html.Div(
    [
        html.H1("This is our Analytics page"),
        html.Div(
            [
                "Select a city: ",
                dcc.RadioItems(
                    options=["New York City", "Montreal", "San Francisco"],
                    value="Montreal",
                    id="analytics-input",
                ),
            ]
        ),
        html.Br(),
        html.Div(id="analytics-output"),
    ],
    style={"color": "white"},
)


@callback(Output("analytics-output", "children"), Input("analytics-input", "value"))
def update_city_selected(input_value):
    return f"You selected: {input_value}"
