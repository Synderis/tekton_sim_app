import pandas as pd
import wtforms
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
from sqlalchemy import URL
from SQL_file_connection import app, QueryForm, TektonResults, session, db_engine
from flask import render_template, Flask, request, redirect, url_for
from sqlalchemy import create_engine, MetaData, engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from flask_wtf import FlaskForm
from generate_graphs import plot_img
from wtforms import IntegerField, TextAreaField, SubmitField, RadioField, SelectField, BooleanField
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sqlalchemy.ext.declarative import declarative_base
from urllib import parse


def output_formatter(numerator, divisor, long):
    if long:
        output = f'{str(numerator)}, {str(round(((numerator / divisor) * 100), 2))}%'
    else:
        output = f'{str(round(((numerator / divisor) * 100), 2))}%'
    return output


@app.route('/database', methods=['GET', 'POST',])
def database():
    if request.method == 'POST':
        form_q = QueryForm()
        print(form_q.data)
        print(form_q.ring.data)
        ring_val = str(form_q.ring.data)
        cm_val = bool(form_q.cm.data)
        inq_val = bool(form_q.inq.data)
        feros_val = bool(form_q.feros.data)
        tort_val = bool(form_q.tort.data)
        fang_val = bool(form_q.fang.data)
        five_tick_val = bool(form_q.five_tick.data)
        pre_veng_val = bool(form_q.pre_veng.data)
        veng_camp_val =bool(form_q.veng_camp.data)
        vuln_val = bool(form_q.vuln.data)
        vuln_book_val = bool(form_q.vuln_book.data)
        data_n = session.query(TektonResults).where(TektonResults.ring == ring_val, TektonResults.cm == cm_val,
                                                    TektonResults.inq == inq_val, TektonResults.feros == feros_val,
                                                    TektonResults.tort == tort_val, TektonResults.fang == fang_val,
                                                    TektonResults.five_tick == five_tick_val,
                                                    TektonResults.pre_veng == pre_veng_val,
                                                    TektonResults.veng_camp == veng_camp_val,
                                                    TektonResults.vuln == vuln_val,
                                                    TektonResults.vuln_book == vuln_book_val
                                                    ).limit(7)
        print(data_n.statement)
        fig = output_graph(df)
        png_image = io.BytesIO()
        FigureCanvas(fig).print_png(png_image)

        # Encode PNG image to base64 string
        png_image_b64_string = "data:image/png;base64,"
        png_image_b64_string += base64.b64encode(png_image.getvalue()).decode('utf8')
        model = LinearRegression()
        model = model.fit(df[ ["anvil_count"] ], df["hp_after_pre_anvil"] )

        print(df)
        # return render_template(r"table_refresh.html", tables=[table_dataframe.to_html(classes='table_data', header=True)])
        return render_template(r"table_refresh.html", tables=[df.to_html(classes='data_n')])
