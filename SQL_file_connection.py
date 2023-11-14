import pandas as pd
import wtforms
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
from sqlalchemy import URL
from flask import render_template, Flask, request, redirect, url_for
from sqlalchemy import create_engine, MetaData, engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from flask_wtf import FlaskForm
from wtforms import IntegerField, TextAreaField, SubmitField, RadioField, SelectField, BooleanField
from sqlalchemy.ext.declarative import declarative_base
from urllib import parse


# this variable, db, will be used for all SQLAlchemy commands
# db = SQLAlchemy()
# create the app
db = SQLAlchemy()
app = Flask(__name__, template_folder=r'C:\Users\Dylan\PycharmProjects\home_project\myflaskproject\.venv\templates')
# change string to the name of your database; add path if necessary

connection_url = engine.URL.create(
    r"mssql+pyodbc",
    host=r"DESKTOP-3TJHN4P\MSSQLSERVER01",
    database=r"tekton_sim_data",
    query={
        r"driver": "ODBC Driver 17 for SQL Server",
        r"LongAsMax": "Yes",
    },
)

db_engine = create_engine(connection_url, echo=False)

SessionObject = sessionmaker(bind=db_engine)
session = SessionObject()


app.config['SQLALCHEMY_DATABASE_URI'] = connection_url

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['SECRET_KEY'] = 'any secret string'

# initialize the app with Flask-SQLAlchemy
db.init_app(app)


# NOTHING BELOW THIS LINE NEEDS TO CHANGE
# this route will test the database connection - and nothing more
@app.route('/')
def testdb():
    try:
        db.session.query(text('1')).from_statement(text('SELECT 1')).all()
        return '<h1>It works.</h1>'
    except Exception as e:
        # e holds description of the error
        error_text = "<p>The error:<br>" + str(e) + "</p>"
        hed = '<h1>Something is broken.</h1>'
        return hed + error_text


headings = ('tick_times', 'anvil_count', 'hammer_count', 'cm', 'inq', 'five_tick_only', 'fang', 'b_ring', 'brim', 'feros', 'tort', 'lightbearer', 'preveng', 'veng_camp', 'vuln', 'book_of_water')


class TektonResults(db.Model):
    id = db.Column('ID', db.Integer, primary_key=True)
    tick_times = db.Column('tick_times', db.Integer)
    anvil_count = db.Column('anvil_count', db.Integer)
    hammer_count = db.Column('hammer_count', db.Integer)
    hp_after_pre_anvil = db.Column('hp_after_pre_anvil', db.Integer)
    ring = db.Column('ring', db.String)
    cm = db.Column('cm', db.Boolean)
    inq = db.Column('inq', db.Boolean)
    feros = db.Column('feros', db.Boolean)
    tort = db.Column('tort', db.Boolean)
    fang = db.Column('fang', db.Boolean)
    five_tick = db.Column('five_tick_only', db.Boolean)
    pre_veng = db.Column('preveng', db.Boolean)
    veng_camp = db.Column('veng_camp', db.Boolean)
    vuln = db.Column('vuln', db.Boolean)
    vuln_book = db.Column('book_of_water', db.Boolean)


class QueryForm(FlaskForm):
    ring = SelectField('Select ring', choices=[('b_ring', 'B ring'), ('brim', 'Brim'), ('ultor_ring', 'Ultor Ring'), ('lightbearer', 'Lightbearer')])
    cm = BooleanField('CM', default=False)
    inq = BooleanField('Inq', default=False)
    feros = BooleanField('Feros', default=False)
    tort = BooleanField('Tort', default=False)
    fang = BooleanField('Fang', default=False)
    five_tick = BooleanField('Five Tick Only', default=False)
    pre_veng = BooleanField('Pre Veng', default=False)
    veng_camp = BooleanField('Veng Camp', default=False)
    vuln = BooleanField('Vuln', default=False)
    vuln_book = BooleanField('Book of Water', default=False)
    submit = SubmitField("Submit")


@app.route("/table", methods=['GET', 'POST',])
def table():
    form_q = QueryForm()
    if request.method == 'POST':
        return redirect(url_for("database"))
    return render_template(r"table.html", form=form_q)



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
        df = pd.read_sql(data_n.statement, con=db_engine)
        # sub_115 = len(df[(df['tick_times'] <= 125)].copy())
        # sub_100 = len(df[(df['tick_times'] <= 100)].copy())
        # one_anvil_num = len(df[(df['anvil_count'] <= 1)].copy())
        # two_anvil_num = len(df[(df['anvil_count'] == 2)].copy())

        print(df)
        return render_template(r"table_refresh.html", tables=[df.to_html(classes='data_n')])


if __name__ == '__main__':
    app.run(debug=True)
