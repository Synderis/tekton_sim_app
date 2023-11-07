import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
from sqlalchemy import URL
from flask import render_template, Flask, request, redirect, url_for
from sqlalchemy import engine
from sqlalchemy import text
from urllib import parse


# this variable, db, will be used for all SQLAlchemy commands
# db = SQLAlchemy()
# create the app
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

app.config['SQLALCHEMY_DATABASE_URI'] = connection_url

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# initialize the app with Flask-SQLAlchemy
db = SQLAlchemy(app)


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


main_sql = text('')

sql_cmd = text("""SELECT TOP (10) [tick_times]
          ,[anvil_count]
          ,[hammer_count]
          ,[cm]
          ,[inq]
          ,[five_tick_only]
          ,[fang]
          ,[b_ring]
          ,[brim]
          ,[feros]
          ,[tort]
          ,[lightbearer]
          ,[preveng]
          ,[veng_camp]
          ,[vuln]
          ,[book_of_water]
      FROM [tekton_sim_data].[dbo].[tekton_results]""")


sql_cmd1 = text("""SELECT TOP (10) [tick_times]
          ,[anvil_count]
          ,[hammer_count]
          ,[cm]
          ,[inq]
          ,[five_tick_only]
          ,[fang]
          ,[b_ring]
          ,[brim]
          ,[feros]
          ,[tort]
          ,[lightbearer]
          ,[preveng]
          ,[veng_camp]
          ,[vuln]
          ,[book_of_water]
      FROM [tekton_sim_data].[dbo].[tekton_results]
      WHERE ([b_ring] = 1 AND [brim] = 0)""")

sql_cmd2 = text("""SELECT TOP (10) [tick_times]
          ,[anvil_count]
          ,[hammer_count]
          ,[cm]
          ,[inq]
          ,[five_tick_only]
          ,[fang]
          ,[b_ring]
          ,[brim]
          ,[feros]
          ,[tort]
          ,[lightbearer]
          ,[preveng]
          ,[veng_camp]
          ,[vuln]
          ,[book_of_water]
      FROM [tekton_sim_data].[dbo].[tekton_results]
      WHERE ([b_ring] = 0 AND [brim] = 1)""")

headings = ('tick_times', 'anvil_count', 'hammer_count', 'cm', 'inq',
       'five_tick_only', 'fang', 'b_ring', 'brim', 'feros', 'tort',
       'lightbearer', 'preveng', 'veng_camp', 'vuln', 'book_of_water')


# @app.route('/table/')
# def table():
#     data = db.session.execute(sql_cmd)
#     # data = pd.DataFrame(data)
#     return render_template(r"table.html", headings=headings, data=data)


@app.route("/table/", methods=['GET', 'POST',])
def table():
    global main_sql
    if request.method == 'POST':
        if request.form.get('action1') == 'b_ring':
            main_sql = sql_cmd1
        elif request.form.get('action2') == 'brim':
            main_sql = sql_cmd2
        else:
            main_sql = sql_cmd
        return redirect(database()), main_sql
    elif request.method == 'GET':
        data = db.session.execute(sql_cmd)
        return render_template(r"table_refresh.html", headings=headings, data=data)


@app.route('/database', methods=['POST'])
def database():
    print('main:', main_sql)
    data = db.session.execute(main_sql)

    return render_template(r"table_refresh.html", headings=headings, data=data)


if __name__ == '__main__':
    app.run(debug=True)