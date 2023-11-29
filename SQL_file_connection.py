import pandas as pd
import wtforms
from flask_sqlalchemy import SQLAlchemy
import matplotlib
from matplotlib import pyplot as plt
from sqlalchemy.sql import text
from sqlalchemy import URL
from flask import render_template, Flask, request, redirect, url_for, Response, jsonify, session
from sqlalchemy import create_engine, MetaData, engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from flask_wtf import FlaskForm
from flask_restful import Resource, Api
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from flask_bootstrap import Bootstrap4

from generate_graphs import output_graph
from wtforms import IntegerField, TextAreaField, SubmitField, RadioField, SelectField, BooleanField, widgets
from wtforms.widgets import Input, SubmitInput
from sqlalchemy.ext.declarative import declarative_base
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from urllib import parse
import io
import base64


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
session_obj = SessionObject()


app.config['SQLALCHEMY_DATABASE_URI'] = connection_url

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['SECRET_KEY'] = 'any secret string'

# initialize the app with Flask-SQLAlchemy
db.init_app(app)
Bootstrap4(app)

# NOTHING BELOW THIS LINE NEEDS TO CHANGE
# this route will test the database connection - and nothing more


@app.route('/')
def testdb():
    return redirect(url_for("gear"))


class style():
    RED = '\033[31m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    RESET = '\033[0m'


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
    five_tick_only = db.Column('five_tick_only', db.Boolean)
    preveng = db.Column('preveng', db.Boolean)
    veng_camp = db.Column('veng_camp', db.Boolean)
    vuln = db.Column('vuln', db.Boolean)
    book_of_water = db.Column('book_of_water', db.Boolean)


class QueryForm(FlaskForm):
    ring = SelectField('Select ring', choices=[('b_ring', 'B ring'), ('brim', 'Brim'), ('ultor_ring', 'Ultor Ring'), ('lightbearer', 'Lightbearer'), (None, 'None')])
    cm = BooleanField('CM', default=False)
    inq = BooleanField('Inq', default=False)
    feros = BooleanField('Feros', default=False)
    tort = BooleanField('Tort', default=False)
    fang = BooleanField('Fang', default=False)
    five_tick_only = BooleanField('Five Tick Only', default=False)
    preveng = BooleanField('Pre Veng', default=False)
    veng_camp = BooleanField('Veng Camp', default=False)
    vuln = BooleanField('Vuln', default=False)
    book_of_water = BooleanField('Book of Water', default=False)
    submit = SubmitField("Submit")


@app.route("/gear", methods=['GET', 'POST',])
def gear():
    form_q = QueryForm()
    return render_template(r"gear.html", form_q=form_q)


@app.route('/database', methods=['GET', 'POST',])
def database():
    form_q = QueryForm()
    if form_q.data.get('ring') is not None:
        query_dict = form_q.data
        if 'submit' and 'csrf_token' in query_dict:
            query_dict.pop('submit')
            query_dict.pop('csrf_token')
        session['query_params'] = query_dict
    try:
        print(style.BLUE + "{0}".format(request.json) + style.RESET)
        flag = True
    except:
        flag = False
        pass

    if not flag:
        print(style.GREEN + "{0}".format(session['query_params']) + style.RESET)
        print('------------------------------------------Generating Output Graphs------------------------------------------')
        data_n = session_obj.query(TektonResults).filter_by(**session['query_params'])
        # print(data_n.statement)
        matplotlib.use('agg')
        df = pd.read_sql(data_n.statement, con=db_engine)
        if len(df) == 0:
            redirect(url_for('database/update'))
        fig = output_graph(df)
        fig.dpi = 100
        fig.set_figwidth(16)
        fig.set_figheight(9)
        fig.subplots_adjust(wspace=.2, hspace=.05, right=.95, left=0.04, top=.91, bottom=.07)
        png_image = io.BytesIO()
        FigureCanvas(fig).print_png(png_image)
        df_new = df.iloc[0:2].copy()
        # Encode PNG image to base64 string
        png_image_b64_string = "data:image/png;base64,"
        png_image_b64_string += base64.b64encode(png_image.getvalue()).decode('utf8')
        plt.close()
        return render_template(r"table_refresh.html", data_table=[df_new.to_html(classes='data_n', index=False)],
                               image=png_image_b64_string, json_data=session['query_params'])
    else:
        print('------------------------------------------Generating Hp Table------------------------------------------')
        prev_q = eval(request.json.get("query_params"))
        data_n = session_obj.query(TektonResults).filter_by(**prev_q)
        table_df = pd.read_sql(data_n.statement, con=db_engine)
        # now we can run the query
        hp_val = int(request.json.get("hpInput"))
        subset_table = table_df[(table_df['anvil_count'] == 1)].copy()
        total_above = round((len(table_df[table_df['hp_after_pre_anvil'] > hp_val].copy()) / len(table_df)) * 100, 2)
        total_below = round((len(table_df[table_df['hp_after_pre_anvil'] < hp_val].copy()) / len(table_df)) * 100, 2)
        total_above_subset = round((len(table_df[(table_df['hp_after_pre_anvil'] > hp_val) & (table_df['anvil_count'] == 1)].copy()) / len(
            subset_table)) * 100, 2)
        total_below_subset = round((len(
           table_df[(table_df['hp_after_pre_anvil'] < hp_val) & (table_df['anvil_count'] == 1)].copy()) / len(
            subset_table)) * 100, 2)
        new_tbl = {'Data Set': ['Above', 'Below', 'One Anvils Above', 'One Anvils Below', 'HP Selected'], 'Data Value': [f'{total_above}%', f'{total_below}%', f'{total_above_subset}%', f'{total_below_subset}%', hp_val]}
        new_tbl = pd.DataFrame(data=new_tbl)
        return new_tbl.to_html(index=False)
        #
    #
    #         x = str('working')

        # print(df.describe())
        # df = df[['tick_times', 'anvil_count', 'hammer_count', 'hp_after_pre_anvil']]
        # array = df.values
        # X = array[:, 0:3]
        # y = array[:, 3]
        # X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
        # ...
        # # Spot Check Algorithms
        # models = []
        # models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        # models.append(('LDA', LinearDiscriminantAnalysis()))
        # models.append(('KNN', KNeighborsClassifier()))
        # models.append(('CART', DecisionTreeClassifier()))
        # models.append(('NB', GaussianNB()))
        # models.append(('SVM', SVC(gamma='auto')))
        # # evaluate each model in turn
        # results = []
        # names = []
        # for name, model in models:
        #     kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
        #     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=None)
        #     results.append(cv_results)
        #     names.append(name)
        #     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
        # model = SVC(gamma='auto')
        # model.fit(X_train, Y_train)
        # predictions = model.predict(X_validation)
        # print(accuracy_score(Y_validation, predictions))
        # print(confusion_matrix(Y_validation, predictions))
        # print(classification_report(Y_validation, predictions))

@app.route('/database/update', methods=['GET', 'POST',])
def database_import():
    return

if __name__ == '__main__':
    app.run(debug=True)
