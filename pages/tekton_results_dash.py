import dash
from dash import html, dcc, callback, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, MetaData, engine, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import URL


dash.register_page(__name__, path="/tekton-dashboard")

connection_url = engine.URL.create(
    r"mssql+pyodbc",
    host=r"DESKTOP-J8L86O2",
    database=r"tekton_sim_data",
    query={
        r"driver": "ODBC Driver 17 for SQL Server",
        r"LongAsMax": "Yes",
    },
)


connection_string = r"Driver={ODBC Driver 17 for SQL Server}; Server=DESKTOP-J8L86O2; Database=tekton_sim_data; Trusted_Connection=yes;"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
engine = create_engine(connection_url)
conn = engine.connect()
data = conn.execute(
    text(
        """select TOP 10000 tick_times, anvil_count, hammer_count, hp_after_pre_anvil from tekton_results"""
    )
)

# Sample data
# df = pd.DataFrame(
#     {
#         "Month": [
#             "January",
#             "February",
#             "March",
#             "April",
#             "May",
#             "June",
#             "July",
#             "August",
#             "September",
#             "October",
#             "November",
#             "December",
#         ],
#         "Sales": [150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
#         "Profit": [50, 70, 80, 90, 100, 130, 150, 170, 180, 200, 220, 250],
#     }
# )

df = pd.DataFrame(data)


layout = html.Div(
    children=[
        html.H1(children="Sample Tekton Results Dashboard"),
        html.Div(
            children="""
        A simple tekton results dashboard with a line chart and a bar chart.
    """
        ),
        html.Div(
            className="row",
            children=[
                dcc.RadioItems(
                    options=["hp_after_pre_anvil", "anvil_count", "hammer_count"],
                    value="hp_after_pre_anvil",
                    inline=True,
                    id="my-radio-buttons-final",
                )
            ],
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="two_columns",
                    children=[
                        dash_table.DataTable(
                            data=df.to_dict("records"),
                            page_size=15,
                            style_table={"overflowX": "auto"},
                            style_cell={"padding": "5px", "backgroundColor": "#333"},
                            style_header={
                                "color": "white",
                                "backgroundColor": "#333",
                                "fontWeight": "bold",
                            },
                            style_data={"color": "white"},
                        )
                    ],
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            id="line-chart",
                        ),
                    ],
                    style={
                        "width": "90vw",
                        "height": "90vw",
                    },
                ),
            ],
            style={
                "display": "flex",
                "flex-direction": "row",
                "width": "90vw",
                "height": "90vw",
            },
        ),
    ]
)


@callback(
    Output(component_id="line-chart", component_property="figure"),
    Input(component_id="my-radio-buttons-final", component_property="value"),
)
def update_graph(col_chosen):
    fig = px.scatter(df, x="tick_times", y=col_chosen)
    fig.update_layout(plot_bgcolor="lightblue", paper_bgcolor="#333")
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="black",
        tickfont={"color": "white"},
        titlefont={"color": "white"},
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="black",
        tickfont={"color": "white"},
        titlefont={"color": "white"},
    )
    return fig
