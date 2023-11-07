from SQL_file_connection import db, app
from flask import render_template, Flask
import flask_sqlalchemy as fsa

rings = db.Table('tekton_results', db.Column('b_ring', db.Integer), db.Column('brim', db.Integer))








if (__name__ == '__main__'):
    app.run()


