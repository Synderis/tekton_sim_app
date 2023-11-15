import pandas as pd
import wtforms
from flask_sqlalchemy import SQLAlchemy
import matplotlib
from matplotlib import pyplot as plt
from sqlalchemy.sql import text
from sqlalchemy import URL
from flask import render_template, Flask, request, redirect, url_for
from sqlalchemy import create_engine, MetaData, engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from flask_wtf import FlaskForm
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
from generate_graphs import output_graph
from wtforms import IntegerField, TextAreaField, SubmitField, RadioField, SelectField, BooleanField
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
session = SessionObject()


app.config['SQLALCHEMY_DATABASE_URI'] = connection_url

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['SECRET_KEY'] = 'any secret string'

# initialize the app with Flask-SQLAlchemy
db.init_app(app)

# app.register_blueprint(database_bp)

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
    ring = SelectField('Select ring', choices=[('b_ring', 'B ring'), ('brim', 'Brim'), ('ultor_ring', 'Ultor Ring'), ('lightbearer', 'Lightbearer'), (None, 'None')])
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
        veng_camp_val = bool(form_q.veng_camp.data)
        vuln_val = bool(form_q.vuln.data)
        vuln_book_val = bool(form_q.vuln_book.data)
        data_n = session.query(TektonResults).where(TektonResults.ring == ring_val, TektonResults.cm == cm_val,
                                                    TektonResults.inq == inq_val, TektonResults.feros == feros_val,
                                                    TektonResults.tort == tort_val, TektonResults.fang == fang_val,
                                                    TektonResults.five_tick == five_tick_val,
                                                    TektonResults.pre_veng == pre_veng_val,
                                                    TektonResults.veng_camp == veng_camp_val,
                                                    TektonResults.vuln == vuln_val,
                                                    TektonResults.vuln_book == vuln_book_val)
        print(data_n.statement)
        matplotlib.use('agg')
        df = pd.read_sql(data_n.statement, con=db_engine)
        fig = output_graph(df)
        fig.dpi = 100
        fig.set_figwidth(16)
        fig.set_figheight(9)
        fig.subplots_adjust(wspace=.2, hspace=.05, right=.95, left=0.04, top=.92, bottom=.07)
        png_image = io.BytesIO()
        FigureCanvas(fig).print_png(png_image)
        df_new = df.iloc[0:7].copy()
        # Encode PNG image to base64 string
        png_image_b64_string = "data:image/png;base64,"
        png_image_b64_string += base64.b64encode(png_image.getvalue()).decode('utf8')
        plt.close()
        print(df)
        one_anvils = df[df['anvil_count'] == 1].copy()
        avg_hp = one_anvils['hp_after_pre_anvil'].mean()
        max_hp = one_anvils['hp_after_pre_anvil'].max()
        min_hp = one_anvils['hp_after_pre_anvil'].min()
        if not request.form.get('data'):
            estimate = request.form.get('data')
            total_above = (len(df[df['hp_after_pre_anvil'] > estimate].copy()) / len(df)) ** 100
            total_below = (len(df[df['hp_after_pre_anvil'] < estimate].copy()) / len(df)) ** 100
            total_above_subset = (len(df[(df['hp_after_pre_anvil'] > estimate) & (df['anvil_count'] == 1)].copy()) / len(df)) ** 100
            total_below_subset = (len(df[(df['hp_after_pre_anvil'] < estimate) & (df['anvil_count'] == 1)].copy()) / len(df)) ** 100
        else:
            total_above = 1 / len(df) ** 100
            total_below = 1 / len(df) ** 100
            total_above_subset = 1 / len(df) ** 100
            total_below_subset = 1 / len(df) ** 100
        misc_table = {'Average pre anvil hp': avg_hp, 'Max pre anvil hp': max_hp, 'Min pre anvil hp': min_hp,
                      '% above input': f'{total_above}%', '% below input': f'{total_below}%',
                      '% 1 anvils above input': f'{total_above_subset}%',
                      '% 1 anvils below input': f'{total_below_subset}%'}
        misc_table = pd.DataFrame([misc_table])
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
        return render_template(r"table_refresh.html", data_table=[df_new.to_html(classes='data_n')], misc_table=[misc_table.to_html(classes='data_n')], image=png_image_b64_string)


@app.route('/process', methods=['POST'])
def process():
    data = request.form.get('data')
    # process the data using Python code

    result = int(data) * 2
    return str(result)


if __name__ == '__main__':
    app.run(debug=True)
