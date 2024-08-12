import dash
from dash import html, dcc, callback, Input, Output, dash_table

# import plotly.express as px
import plotly
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


def get_data():
    data = conn.execute(
        text(
            """select TOP 10000 tick_times, anvil_count, hammer_count, hp_after_pre_anvil from tekton_results order by ID desc"""
        )
    )
    df = pd.DataFrame(data)
    return df


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
                            id="tek_data",
                            page_size=15,
                            style_table={"overflowX": "auto"},
                            style_cell={"padding": "5px", "backgroundColor": "#333"},
                            style_header={
                                "color": "white",
                                "backgroundColor": "#333",
                                "fontWeight": "bold",
                            },
                            style_data={"color": "white"},
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            id="line-chart",
                            # figure={}
                        ),
                        dcc.Interval(
                            id="interval-component", interval=1 * 10000, n_intervals=0
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
    Output(component_id="tek_data", component_property="data"),
    Input(component_id="my-radio-buttons-final", component_property="value"),
    Input(component_id="interval-component", component_property="n_intervals"),
)
def update_graph(colchosen, n):
    data = get_data()

    fig = px.scatter(data, x="tick_times", y=colchosen)
    fig.update_layout(plot_bgcolor="lightblue", paper_bgcolor="#333")
    # {'data': [{'x': data['tick_times'], 'y': data[colchosen]}], 'layout': {'xaxis': {'gridcolor': 'black', 'showgrid': 'True'}, 'yaxis': {'gridcolor': 'black', 'showgrid': 'True'}}}
    #  'marker': {...},
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
    return {
        "data": [{"x": data["tick_times", "y" : data[colchosen]]}],
        "layout": {
            "xaxis": {"gridcolor": "black", "showgrid": "True"},
            "yaxis": {"gridcolor": "black", "showgrid": "True"},
        },
    }, data.to_dict("records")
