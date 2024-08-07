import dash
from dash import html

dash.register_page(__name__, path="/archives-page")

layout = html.Div(
    [
        html.H1("This is our Archive page"),
        html.Div("This is our Archive page content."),
    ],
    style={"color": "white"},
)
