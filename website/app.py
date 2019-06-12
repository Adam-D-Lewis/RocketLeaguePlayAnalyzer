from flask import Flask, render_template, request, redirect
from bokeh.embed import components
from bokeh.plotting import figure
import dill
from pathlib import Path
from website.python.server_data_processing import do_exactly_what_I_want_function, get_recommendations

app = Flask(__name__)
# construct column_index_dict
# column_index_dict = {[(i, name) for i, name in enumerate(columns['name'])]}

@app.route('/', methods=['GET'])
def index():
    return render_template('about.html')

@app.route('/player_analyzer', methods=['GET'])
def player_analyzer():
    return render_template('index.html')

@app.route('/player_analysis', methods=['GET'])
def player_analysis():
    base_path = Path('../Saved_Data')
    pr_df = dill.load(open(base_path / 'pr_df_cleaned_and_typed.pkd', 'rb'))
    scaled_neighbor_ball_tree_dict = dill.load(open(base_path / 'scaled_neighbor_ball_tree_dict.pkd', 'rb'))

    steam_id = request.args['steam_id'].strip()

    if steam_id not in pr_df['id'].values:
        return render_template('index.html')

    p, training_recommendation_categories, recommendation_resources = do_exactly_what_I_want_function(int(steam_id), base_path, pr_df, scaled_neighbor_ball_tree_dict)

    script, div = components(p)
    var_names = ['cat1', 'cat2', 'cat3', 'cat4', 'cat1r1', 'cat1r2', 'cat1r3', 'cat2r1', 'cat2r2', 'cat2r3', 'cat3r1',
                 'cat3r2', 'cat3r3', 'cat4r1', 'cat4r2', 'cat4r3']
    var_vals = training_recommendation_categories + recommendation_resources
    webpage_vars_dict = dict(zip(var_names, var_vals))

    return render_template('player_analysis.html', script=script, div=div, steam_id=steam_id, **webpage_vars_dict)


if __name__ == '__main__':
  app.run(port=33507)
