from flask import Flask, render_template, request, redirect
from bokeh.embed import components
from bokeh.plotting import figure
from python.server_data_processing import create_plot, get_recommendations

app = Flask(__name__)
# construct column_index_dict
# column_index_dict = {[(i, name) for i, name in enumerate(columns['name'])]}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/player_analysis', methods=['GET'])
def player_analysis():
    steam_id = request.args['steam_id']
    # print(next(request.form.items()))
    # for key, val in request.form.items():
    #     print(key, val)


    p, training_recommendation_categories, recommendation_resources = create_plot(int(steam_id))

    p2 = make_plot()
    script, div = components(p)
    script2, div2 = components(p2)

    var_names = ['cat1', 'cat2', 'cat3', 'cat4', 'cat1r1', 'cat1r2', 'cat1r3', 'cat2r1', 'cat2r2', 'cat2r3', 'cat3r1',
                 'cat3r2', 'cat3r3', 'cat4r1', 'cat4r2', 'cat4r3']
    var_vals = training_recommendation_categories + recommendation_resources

    webpage_vars_dict = dict(zip(var_names, var_vals))

    return render_template('player_analysis.html', script=script, div=div, steam_id=steam_id, script2=script2,
                           div2=div2, **webpage_vars_dict)


def make_plot():
    # prepare some data
    x = [1, 2, 3, 4, 5]
    y = [6, 7, 2, 4, 5]

    # create a new plot with a title and axis labels
    p = figure(title="simple line example", x_axis_label='x', y_axis_label='y', sizing_mode='scale_width', height=250)
    # p.sizing_mode = 'scale_width'
    # add a line renderer with legend and line thickness
    p.line(x, y, legend="Temp.", line_width=2)

    return p


if __name__ == '__main__':
  app.run(port=33507)
