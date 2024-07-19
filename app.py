import time
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
import matplotlib
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from flask import render_template, Flask, request, redirect, url_for, Response, jsonify, session
from sqlalchemy import create_engine, MetaData, engine, select
from sqlalchemy.orm import sessionmaker
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, r2_score, mean_squared_error
import update_sim_data
from generate_graphs import output_graph
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import FuncFormatter, MaxNLocator
import io
import base64


# create the app
db = SQLAlchemy()
app = Flask(__name__)
# change string to the name of your database; add path if necessary

connection_url = engine.URL.create(
    r"mssql+pyodbc",
    host=r"DESKTOP-J8L86O2",
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
    short_lure = db.Column('short_lure', db.Boolean)


@app.route("/gear", methods=['GET', 'POST',])
def gear():
    return render_template(r"gear.html")


@app.route('/database', methods=['GET', 'POST',])
def database():
    query_dict = {'ring': 'ultor_ring', 'cm': True, 'inq': True, 'feros': True, 'tort': True, 'fang': False,
                  'five_tick_only': False, 'preveng': True, 'veng_camp': False, 'vuln': False, 'book_of_water': False,
                  'short_lure': False}
    # query_dict = {'ring': 'ultor_ring', 'cm': True, 'inq': True, 'feros': True, 'tort': True, 'fang': False,
    #               'five_tick_only': False, 'preveng': True, 'veng_camp': False, 'vuln': False, 'book_of_water': False}

    if not request.form.to_dict():
        temp_dict = session['query_params'].copy()
    else:
        temp_dict = request.form.to_dict()
    for key in query_dict.keys():
        if key in temp_dict.keys():
            query_dict[key] = True
        else:
            query_dict[key] = False
        if key == 'ring':
            query_dict[key] = temp_dict[key]
    query_dict1 = query_dict
    session['query_params'] = query_dict1
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
        print(df.columns)
        length = len(df)
        if length != 40000:
            # print(**session['query_params'])
            update_sim_data.map_parameters(**session['query_params'])
            time.sleep(1)
            df = pd.read_sql(data_n.statement, con=db_engine)
            length = 40000
        df = df[:10000]
        hp_avg = int(df['hp_after_pre_anvil'].mean())
        df_new = df.iloc[0:2].copy()
        group_plot_fig, cumul_graph_fig, one_anvil_hist_kde_fig, total_sample_hist_kde_fig = output_graph(df)

        def create_img_str(figure_, subplots_):
            figure_.dpi = 100
            figure_.set_figwidth(17)
            figure_.set_figheight(9)
            if subplots_:
                figure_.subplots_adjust(wspace=.2, hspace=.05, right=.95, left=0.04, top=.91, bottom=.07)
            png_image = io.BytesIO()
            FigureCanvas(figure_).print_png(png_image)
            plt.close()
            png_image_b64_string = "data:image/png;base64,"
            png_image_b64_string += base64.b64encode(png_image.getvalue()).decode('utf8')
            return png_image_b64_string

        group_plot_img, cumul_graph_img, one_anvil_hist_kde_img, total_sample_hist_kde_img = create_img_str(group_plot_fig, True), create_img_str(cumul_graph_fig, False), create_img_str(one_anvil_hist_kde_fig, False), create_img_str(total_sample_hist_kde_fig, False)
        return render_template(r"table_refresh.html", data_table=[df_new.to_html(classes='data_n', index=False)],
                               group_image=group_plot_img, cdf_img=cumul_graph_img,
                               one_anvil_img=one_anvil_hist_kde_img, total_anvil_img=total_sample_hist_kde_img,
                               json_data=session['query_params'], hp_default=hp_avg)
    else:
        print('------------------------------------------Generating Hp Table------------------------------------------')
        prev_q = eval(request.json.get("query_params"))
        data_n = session_obj.query(TektonResults).filter_by(**prev_q)
        table_df = pd.read_sql(data_n.statement, con=db_engine)
        # now we can run the query
        hp_val = int(request.json.get("hpInput"))
        subset_table = table_df[(table_df['anvil_count'] <= 1)].copy()
        total_above = round((len(table_df[table_df['hp_after_pre_anvil'] > hp_val].copy()) / len(table_df)) * 100, 2)
        total_below = round((len(table_df[table_df['hp_after_pre_anvil'] < hp_val].copy()) / len(table_df)) * 100, 2)
        total_above_subset = round((len(table_df[(table_df['hp_after_pre_anvil'] > hp_val) & (table_df['anvil_count'] <= 1)].copy()) / len(
            subset_table)) * 100, 2)
        total_below_subset = round((len(
           table_df[(table_df['hp_after_pre_anvil'] < hp_val) & (table_df['anvil_count'] == 1)].copy()) / len(
            subset_table)) * 100, 2)
        new_tbl = {'Data Set': ['Above', 'Below', 'One Anvils Above', 'One Anvils Below', 'HP Selected'], 'Data Value': [f'{total_above}%', f'{total_below}%', f'{total_above_subset}%', f'{total_below_subset}%', hp_val]}
        new_tbl = pd.DataFrame(data=new_tbl)
        return new_tbl.to_html(index=False)


@app.route('/model', methods=['GET', 'POST',])
def ml_model():
    query_dict = {'hp_after_anvil': 0, 'hammer_count': 0, 'ring': 'ultor_ring', 'cm': True, 'inq': True, 'feros': True, 'tort': True, 'fang': False,
                  'five_tick_only': False, 'preveng': True, 'veng_camp': False, 'vuln': False, 'book_of_water': False,
                  'short_lure': False}
    # query_dict = {'ring': 'ultor_ring', 'cm': True, 'inq': True, 'feros': True, 'tort': True, 'fang': False,
    #               'five_tick_only': False, 'preveng': True, 'veng_camp': False, 'vuln': False, 'book_of_water': False}
    print(request.form.to_dict())
    if not request.form.to_dict():
        temp_dict = session['query_params'].copy()
    else:
        temp_dict = request.form.to_dict()
    for key in query_dict.keys():
        if key in temp_dict.keys():
            query_dict[key] = True
        else:
            query_dict[key] = False
        if key in ['ring', 'hammer_count', 'hp_after_anvil']:
            query_dict[key] = temp_dict[key]
    query_dict1 = query_dict
    session['query_params'] = query_dict1
    print(query_dict1)
    s_dict = session['query_params'].copy()
    s_dict.pop('hammer_count')
    s_dict.pop('hp_after_anvil')

    data_n = session_obj.query(TektonResults).filter_by(**s_dict).limit(40000)
    df = pd.read_sql(data_n.statement, con=db_engine)

    # Drop unused columns
    df = df.drop(columns=['ring', 'short_lure', 'anvil_count', 'ID'])
    # df = df.drop(columns=['short_lure', 'anvil_count', 'ID'])

    # Prepare features and target
    X = df.drop(columns=['tick_times']).values
    y = df['tick_times'].values

    # Split the dataset into training and validation sets
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=1)

    # Normalize features (standardization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)

    # Define the best parameters obtained from GridSearchCV
    best_params = {
        'bootstrap': True,
        'max_depth': 10,
        'max_features': 'sqrt',
        'min_samples_leaf': 1,
        'min_samples_split': 5,
        'n_estimators': 100
    }

    # Train the final model using the best parameters
    model = RandomForestRegressor(**best_params, random_state=1)
    model.fit(X_train, y_train)

    # Make predictions with the best model
    predictions = model.predict(X_validation)

    # Evaluate the model
    mse = mean_squared_error(y_validation, predictions)
    r2 = r2_score(y_validation, predictions)
    print("Mean squared error: %.2f" % mse)
    print("Coefficient of determination: %.2f" % r2)
    matplotlib.use('agg')
    # Plot actual vs. predicted values

    def format_func(value, tick_number):
        minutes = int(value // 60)
        seconds = int(value % 60)
        return f'{minutes}:{seconds:02}'

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary plot
    ax1.scatter(y_validation, predictions, color='orange')
    ax1.set_facecolor('gray')
    ax1.plot([min(y_validation), max(y_validation)], [min(y_validation), max(y_validation)], color='crimson', lw=2)
    ax1.set_title('Actual vs Predicted Tick Times', color='orange')
    ax1.set_xlabel('Actual Tick Times', color='orange')
    ax1.set_ylabel('Predicted Tick Times', color='orange')

    # Secondary x-axis
    ax2 = ax1.twiny()
    ax2.set_xlabel('Actual Tick Times in Minutes and Seconds', color='orange')
    primary_xmin, primary_xmax = ax1.get_xlim()
    secondary_xmin, secondary_xmax = primary_xmin * .6, primary_xmax * .6
    ax2.set_xlim(secondary_xmin, secondary_xmax)
    ax2.xaxis.set_major_formatter(FuncFormatter(format_func))

    # Secondary y-axis
    ax3 = ax1.twinx()
    ax3.set_ylabel('Predicted Tick Times in Minutes and Seconds', color='orange')
    primary_ymin, primary_ymax = ax1.get_ylim()
    secondary_ymin, secondary_ymax = primary_ymin * .6, primary_ymax * .6
    ax3.set_ylim(secondary_ymin, secondary_ymax)
    ax3.yaxis.set_major_formatter(FuncFormatter(format_func))

    ax1.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=12))
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=12))

    ax1.grid(True, color='black')

    # Change tick labels to white
    ax1.tick_params(axis='both', colors='white')
    ax2.tick_params(axis='both', colors='white')
    ax3.tick_params(axis='both', colors='white')

    # Save the plot to a BytesIO object
    png_image = io.BytesIO()
    plt.savefig(png_image, format='png', facecolor='none')
    plt.close()

    # Encode the image in base64
    png_image_b64_string = "data:image/png;base64,"
    png_image_b64_string += base64.b64encode(png_image.getvalue()).decode('utf8')


    # plt.show()

    # Plot residuals
    # residuals = y_validation - predictions
    # plt.figure(figsize=(10, 6))
    # sns.histplot(residuals, bins=30, kde=True)
    # plt.title('Residuals of Predictions')
    # plt.xlabel('Residual')
    # plt.ylabel('Frequency')
    # plt.show()

    # Feature Importance
    # feature_importances = model.feature_importances_
    # features = df.drop(columns=['tick_times']).columns
    # importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    # importance_df = importance_df.sort_values(by='Importance', ascending=False)
    #
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='Importance', y='Feature', data=importance_df)
    # plt.title('Feature Importance')
    # plt.show()

    query_dict1.pop('ring')
    test_sample = []
    for each in query_dict1.values():
        test_sample.append(each)
    l1, l2 = test_sample[0], test_sample[1]
    test_sample[0], test_sample[1] = int(l2), int(l1)
    test_sample = [test_sample[:12]]
    test_sample = np.array(
        test_sample)
    print(test_sample)

    if test_sample.shape[1] != X.shape[1]:
        print("Error: Hard-coded sample has an incorrect number of features.")
        return

    # Normalize the test_sample
    test_sample = scaler.transform(test_sample)

    # Predict using the test_sample
    predicted_tick_times = model.predict(test_sample)

    print(f"Test sample features: {test_sample}")
    print(f"Predicted tick_times: {predicted_tick_times[0]}")
    return render_template(r"ml_model.html",  predicted_time=predicted_tick_times[0], mse=mse, r2=r2, ml_graph=png_image_b64_string)


@app.route('/database/update', methods=['GET', 'POST',])
def database_import():
    return


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
