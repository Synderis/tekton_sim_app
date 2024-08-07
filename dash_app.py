import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

external_stylesheets = ["/assets/main.css"]
app = Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets)

app.css.append_css({"external_url": "/assets/main.css"})
app.server.static_folder = "assets"
app.css.config.serve_locally = False


app.layout = html.Div(
    children=[
        html.H1("Multi-page app with Dash Pages"),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.A(
                            className="nav_button",
                            children=f"{page['name']}",
                            href=page["relative_path"],
                            # size="sm",
                        ),
                        html.Br(),
                        html.Br(),
                    ]
                )
                for page in dash.page_registry.values()
            ],
            style={"height": "20hw", "margin": "5px"},
        ),
        dash.page_container,
    ],
    style={"color": "white", "outline": "black"},
)

if __name__ == "__main__":
    app.run(debug=True)
