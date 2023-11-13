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
from wtforms import IntegerField, TextAreaField, SubmitField, RadioField, SelectField
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


class Gear:
    def __init__(self, b_ring_check, brim_check, ultor_check, feros_check, fang_check,
                 cm_check, inq_check, tort_check, lightbearer_check, preveng_check, veng_camp_check, vuln_check,
                 book_of_water_check, five_tick_only_check):
        self.b_ring_check = b_ring_check
        self.brim_check = brim_check
        self.ultor_check = ultor_check
        self.feros_check = feros_check
        self.fang_check = fang_check
        self.cm_check = cm_check
        self.inq_check = inq_check
        self.tort_check = tort_check
        self.lightbearer_check = lightbearer_check
        self.preveng_check = preveng_check
        self.veng_camp_check = veng_camp_check
        self.vuln_check = vuln_check
        self.book_of_water_check = book_of_water_check
        self.five_tick_only_check = five_tick_only_check


selection = Gear(b_ring_check=1, brim_check=0, ultor_check=0, feros_check=0,
                   fang_check=0, cm_check=1, inq_check=1, tort_check=0, lightbearer_check=0,
                   preveng_check=0, veng_camp_check=0, vuln_check=0, book_of_water_check=0, five_tick_only_check=0)


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
    ring = SelectField('Your ring', choices=[('b_ring', 'B ring'), ('brim', 'Brim'), ('ultor_ring', 'Ultor Ring'), ('lightbearer', 'Lightbearer')])
    submit = SubmitField("Submit")


@app.route("/table", methods=['GET', 'POST',])
def table():
    form_q = QueryForm()
    if request.method == 'POST':
        return redirect(url_for("database"))
    return render_template(r"table.html", form=form_q)


form_list = ['B ring', 'Brim', 'CM', 'Inq', 'Feros', 'Tort', 'Fang', 'Five Tick Only', 'Ultor', 'Lightbearer',
             'Pre Veng', 'Veng Camp', 'Vuln', 'Book of Water']
form_value = ['b ring', 'brim', 'cm', 'inq', 'feros', 'tort', 'fang', 'five_tick_only', 'ultor_ring', 'lightbearer',
              'preveng', 'veng_camp', 'vuln', 'vuln_book']
# query_params = [TektonResults.b_ring, TektonResults.brim, TektonResults.cm, TektonResults.inq, TektonResults.feros,
#                 TektonResults.tort, TektonResults.fang, TektonResults.five_tick, TektonResults.ultor_ring,
#                 TektonResults.lightbearer, TektonResults.pre_veng, TektonResults.veng_camp, TektonResults.vuln,
#                 TektonResults.vuln_book]


@app.route('/database', methods=['GET', 'POST',])
def database():
    if request.method == 'POST':
        form_q = QueryForm()
        print(form_q.data)
        print(form_q.ring.data)
        ring = str(form_q.ring.data)
        data_n = session.query(TektonResults).where(TektonResults.ring == ring).limit(7)
        df = pd.read_sql(data_n.statement, con=db_engine)
        # sub_115 = len(df[(df['tick_times'] <= 125)].copy())
        # sub_100 = len(df[(df['tick_times'] <= 100)].copy())
        # one_anvil_num = len(df[(df['anvil_count'] <= 1)].copy())
        # two_anvil_num = len(df[(df['anvil_count'] == 2)].copy())

        print(df)
        return render_template(r"table_refresh.html", tables=[df.to_html(classes='data_n')])


if __name__ == '__main__':
    app.run(debug=True)