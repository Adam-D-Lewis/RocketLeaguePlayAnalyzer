# import seaborn as sns
# sns.set()
import dill
from pathlib import Path
from os import path
import os
import numpy as np
from bokeh.core.properties import value
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import dodge
import pandas as pd


# TODO:  convert csv of recommendations to 1. a label to display on web page and 2. the url to go to a website


def create_plot(steam_id):

    base_path = Path('../Saved_Data')
    # if not base_path.exists():
    #     base_path = Path(r'../../Saved_Data')

    independent_cols = ['id', 'match_date', 'assists', 'numStolenBoosts', 'saves', 'shots', \
                        'timeFullBoost', 'timeHighInAir', 'totalDribbleConts', 'totalGoals', 'totalShots',
                        'turnoversOnMyHalf', 'turnoversOnTheirHalf', 'demoed_opponent', 'demoed_by_opponent', 'rank']

    stat_dict= {'id': 'nonstat', 'match_date': 'nonstat', 'assists': 'Passing', 'numStolenBoosts': 'Boost Management',
                'saves': 'Goalie', 'shots': 'Striker', 'timeFullBoost': 'Boost Management', 'timeHighInAir': 'Aerials',
                'totalDribbleConts': 'Dribbling', 'totalGoals': 'Striker', 'totalShots': 'Striker',
                'turnoversOnMyHalf': 'Dribbling', 'turnoversOnTheirHalf': 'Dribbling', 'demoed_opponent':
                    'Demolition', 'demoed_by_opponent': 'Demolition Avoidance', 'rank': 'nonstat'}

    numeric_cols = set(independent_cols)-set(['match_date', 'id', 'rank'])
    lower_is_better_cols = ['turnoversOnMyHalf', 'turnoversOnTheirHalf', 'demoed_by_opponent']

    pr_df = dill.load(open(base_path / 'pr_df_cleaned_and_typed.pkd', 'rb'))
    # scaled_pr_df = dill.load(open(path.join(base_path, r'pr_df_cleaned_and_typed_scaled.pkd'), 'rb'))
    fitted_ss_dict = dill.load(open(base_path / 'fitted_ss_dict.pkd', 'rb'))
    scaled_neighbor_ball_tree_dict = dill.load(open(base_path / 'scaled_neighbor_ball_tree_dict.pkd', 'rb'))
    rel_feature_importance_df = dill.load(open(base_path / 'rel_feature_importance_df.pkl', 'rb'))

    account_for_lower_better_stats_df = pd.DataFrame(data=np.ones(shape=(1, len(numeric_cols))), columns=numeric_cols)
    account_for_lower_better_stats_df.loc[:, lower_is_better_cols] = account_for_lower_better_stats_df.loc[:, lower_is_better_cols]-2

    def get_replays_at_rank(df, rank):
        return df.loc[np.isclose(pr_df['rank'], player_desired_rank), :]


    def get_ave_player_game_and_desired_rank(steamid):
        players_games = pr_df.loc[pr_df['id'] == str(steamid), :]
        ave_player_game = players_games.sort_values(by='match_date', ascending=False).head(
            20).mean().to_frame().transpose().loc[:, list(numeric_cols) + ['rank']]
        player_desired_rank = (np.round(ave_player_game['rank'].values) + 1)[0]
        return ave_player_game[numeric_cols], player_desired_rank

    def convert_scaled_to_raw(scaled_df, fitted_standard_scalar):
        return (scaled_df * fitted_standard_scalar.scale_) + fitted_standard_scalar.mean_

    def calculate_relative_deficiencies(ave_player_game, player_desired_rank, fitted_ss_dict,
                                        scaled_neighbor_ball_tree_dict):

        # scale the data
        scaled_replays_at_desired_rank = (pr_df.loc[np.isclose(pr_df['rank'], player_desired_rank), numeric_cols] -
                                          fitted_ss_dict[player_desired_rank].mean_) / fitted_ss_dict[
                                             player_desired_rank].scale_
        scaled_ave_player_game = fitted_ss_dict[player_desired_rank].transform(ave_player_game)

        distances, indices = scaled_neighbor_ball_tree_dict[player_desired_rank].kneighbors(scaled_ave_player_game)

        scaled_neighbors = scaled_replays_at_desired_rank.iloc[indices[0], :]
        scores = scaled_neighbors.subtract(scaled_ave_player_game).multiply(account_for_lower_better_stats_df.values)
        return scores.mean().to_frame().transpose(), scaled_neighbors

    ave_player_game, player_desired_rank = get_ave_player_game_and_desired_rank(steam_id)

    deficiencies, scaled_neighbors = calculate_relative_deficiencies(ave_player_game, player_desired_rank, fitted_ss_dict, scaled_neighbor_ball_tree_dict)
    deficiencies[deficiencies<0] = 0
    deficiencies = deficiencies.div(deficiencies.sum(axis=1), axis=0)
    deficiencies.index += player_desired_rank
    training_recommendation_weights = rel_feature_importance_df.loc[player_desired_rank:player_desired_rank, :].mul(deficiencies, axis=1) # the product of feature importance and rel. deficiency gives the recommendation weight
    training_recommendation_weights = training_recommendation_weights.div(training_recommendation_weights.max(axis=1), axis=0)

    training_recommendation_cats = training_recommendation_weights.transpose().nlargest(4, columns=player_desired_rank).index

    training_recommendation_weights.transpose().nlargest(4, columns=player_desired_rank)



    fruits = training_recommendation_cats.values #['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
    # years = ['You', 'Players At Desired Rank']

    # data = {cat: data for cat, data in zip(training_recommendation_cats, )}

    data = {'fruits': training_recommendation_cats.values, #fruits,
            'You': ave_player_game.loc[:, training_recommendation_cats].values[0] * [1, 1/10, 1, 1],
            'Players': convert_scaled_to_raw(scaled_neighbors, fitted_ss_dict[player_desired_rank]).loc[:,
                        training_recommendation_cats].mean().values * [1, 1/10, 1, 1],
            'alpha': training_recommendation_weights.transpose().nlargest(4, columns=player_desired_rank).values,
             'width': training_recommendation_weights.transpose().nlargest(4, columns=player_desired_rank).values*0.4}
    source = ColumnDataSource(data=data)

    p = figure(title="Improvement Statistics", x_range=fruits, height=250, sizing_mode='scale_width',
               toolbar_location='right', border_fill_alpha=0.0, min_border=50)

    p.xaxis.major_label_text_color = "white"
    p.yaxis.major_label_text_color = "white"

    p.xaxis.major_tick_line_color = "white"
    p.yaxis.major_tick_line_color = "white"

    p.xaxis.minor_tick_line_color = "white"
    p.yaxis.minor_tick_line_color = "white"

    p.title.text_font_size = '22pt'
    p.title.text_color = 'gray'
    p.title.align = 'center'

    p.xaxis.major_label_text_font_size = '14pt'
    p.yaxis.major_label_text_font_size = '18pt'

    p.vbar(x=dodge('fruits', -0.2, range=p.x_range), top='You', width=0.3, source=source, color='#e84d60', legend=value("You"))

    p.vbar(x=dodge('fruits',  0.2,  range=p.x_range), top='Players', width=0.3, source=source, color="#718dbf", legend=value("Players"))

    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"

    training_recommendation_cats = [stat_dict[x] for x in training_recommendation_cats]
    recommendation_resources = get_recommendations(training_recommendation_cats,
                                                  int(player_desired_rank-1))

    return p, training_recommendation_cats, recommendation_resources


def get_recommendations(categories, level):

    base_path = Path('../Saved_Data')
    rank_names = ['Bronze', 'Bronze', 'Bronze', 'Silver', 'Silver', 'Silver', 'Gold', 'Gold', 'Gold', 'Platinum',
                  'Platinum', 'Platinum', 'Diamond', 'Diamond', 'Diamond', 'Champion', 'Champion', 'Champion',
                  'Grand Champion']

    rank_numbers = list(range(1, 20))
    rank_dict = dict(zip(rank_numbers, rank_names))

    # load the grid, convert null to empty strin
    # try:
    # print(base_path)
    # files_path = [os.path.abspath(x) for x in os.listdir(base_path)]
    # for path in files_path:
    #     print('file:  ', path)
    recommendation_grid = pd.read_csv(base_path / 'skill_recommendation_table.csv', index_col="Category")

    # except FileNotFoundError:
    #     print(os.getcwd())
    #     recommendation_grid = pd.read_csv(path.join('../../', 'skill_recommendation_table.csv'), index_col="Category")
    recommendation_grid[recommendation_grid.isnull()] = ""

    return recommendation_grid.loc[categories, rank_dict[level]].values.tolist()



if __name__ == "__main__":
    output_file("dodged_bars.html")
    sample_steamid = 76561197990515587
    show(create_plot(sample_steamid))
