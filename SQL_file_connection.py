import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
from sqlalchemy import URL
from flask import render_template, Flask, request, redirect, url_for
from sqlalchemy import create_engine, MetaData, engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
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
# Base = declarative_base()
# Base.metadata.reflect(engine)

app.config['SQLALCHEMY_DATABASE_URI'] = connection_url

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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

x = 0

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
    b_ring = db.Column('b_ring', db.Integer)
    brim = db.Column('brim', db.Integer)
    anvil_count = db.Column('anvil_count', db.Integer)
    hammer_count = db.Column('hammer_count', db.Integer)
    cm = db.Column('cm', db.Integer)
    inq = db.Column('inq', db.Integer)
    feros = db.Column('feros', db.Integer)
    tort = db.Column('tort', db.Integer)
    fang = db.Column('fang', db.Integer)
    five_tick = db.Column('five_tick_only', db.Integer)
    lightbearer = db.Column('lightbearer', db.Integer)
    pre_veng = db.Column('preveng', db.Integer)
    veng_camp = db.Column('veng_camp', db.Integer)
    vuln = db.Column('vuln', db.Integer)
    vuln_book = db.Column('book_of_water', db.Integer)


@app.route("/table/", methods=['GET', 'POST',])
def table():
    global x
    if request.method == 'POST':

        if request.form.get('B ring') == 1:
            selection.b_ring_check = request.form.get('B ring')
        elif request.form.get('Brim') == 1:
            x = request.form.get('Brim')
        return redirect(database())
    elif request.method == 'GET':
        data = db.session.query(TektonResults).where(TektonResults.b_ring == 1).limit(5).all()

        return render_template(r"table.html", headings=headings, data=data)


@app.route('/database', methods=['GET', 'POST',])
def database():
    print(request.form.get('B ring'))
    if request.form.get('B ring') == 'b ring':
        print('this executed')
        data_n = session.query(TektonResults).where(TektonResults.b_ring == 1)
    elif request.form.get('Brim') == 'brim':
        data_n = session.query(TektonResults).where(TektonResults.brim == 1).limit(7)
    else:
        data_n = session.query(TektonResults).limit(5)

    df = pd.read_sql(data_n.statement, con=db_engine)
    # sub_115 = len(df[(df['tick_times'] <= 125)].copy())
    # sub_100 = len(df[(df['tick_times'] <= 100)].copy())
    # one_anvil_num = len(df[(df['anvil_count'] <= 1)].copy())
    # two_anvil_num = len(df[(df['anvil_count'] == 2)].copy())

    print(df)
    return render_template(r"table_refresh.html", titles=df.columns.values, tables=[df.to_html(classes='data_n')])


if __name__ == '__main__':
    app.run(debug=True)