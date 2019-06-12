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

# def get_training_recommendation_stats():
#     training_recommendation_cats = training_recommendation_weights.transpose().nlargest(4,
#                                                                                         columns=player_desired_rank).index


# def get_training_recommendation_cats():


def _create_plot(xlabels, grp1_data, grp2_data):
    """This function creates a 2 group bar plot with specific formatting

    :param xlabels: labels for the x axis of the bar plot
    :param grp1_data: the data for the first group
    :param grp2_data: the data for the second group
    :return: returns the formatted 2 group bar plot
    """
    # initialize data
    data = {'xlabels': xlabels,
            "grp1_data": grp1_data,
            "grp2_data": grp2_data}
    source = ColumnDataSource(data=data)

    # create figure
    p = figure(title="Improvement Statistics", x_range=xlabels, height=250, sizing_mode='scale_width',
               toolbar_location='right', border_fill_alpha=0.0, min_border=50)

    # format figure
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

    # plot each group's data
    p.vbar(x=dodge('xlabels', -0.2, range=p.x_range), top="grp1_data", width=0.3, source=source, color='#e84d60',
           legend=value("grp1_data"))

    p.vbar(x=dodge('xlabels', 0.2, range=p.x_range), top="grp2_data", width=0.3, source=source, color="#718dbf",
           legend=value("grp2_data"))

    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None

    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"

    return p

def do_exactly_what_I_want_function(steam_id, base_path, pr_df, scaled_neighbor_ball_tree_dict):
    # TODO: Call all normalized data "normalized" instead of scaled
    # TODO: Call all denormalized data raw

    # initialize variables
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

    fitted_ss_dict = dill.load(open(base_path / 'fitted_ss_dict.pkd', 'rb'))
    # relative feature importance for each rank
    rel_feature_importance_df = dill.load(open(base_path / 'rel_feature_importance_df.pkl', 'rb'))

    account_for_lower_better_stats_df = pd.DataFrame(data=np.ones(shape=(1, len(numeric_cols))), columns=numeric_cols)
    account_for_lower_better_stats_df.loc[:, lower_is_better_cols] = account_for_lower_better_stats_df.loc[:, lower_is_better_cols]-2

    # initialize sub methods
    def get_ave_player_game_and_desired_rank(steamid):
        players_games = pr_df.loc[pr_df['id'] == str(steamid), :]
        ave_player_game = players_games.sort_values(by='match_date', ascending=False).head(
            20).mean().to_frame().transpose().loc[:, list(numeric_cols) + ['rank']]
        player_desired_rank = (np.round(ave_player_game['rank'].values) + 1)[0]
        return ave_player_game[numeric_cols], player_desired_rank

    def convert_scaled_to_raw(scaled_df, fitted_standard_scalar):
        return (scaled_df * fitted_standard_scalar.scale_) + fitted_standard_scalar.mean_

    def calculate_raw_deficiencies(ave_player_game, player_desired_rank, fitted_ss_dict,
                                        scaled_neighbor_ball_tree_dict):

        # filter the df by rank
        raw_filtered_pr_df = pr_df.loc[np.isclose(pr_df['rank'], player_desired_rank), numeric_cols]

        # get normalization params
        norm_param_mean = fitted_ss_dict[player_desired_rank].mean_
        norm_param_std = fitted_ss_dict[player_desired_rank].scale_

        def normalize_df(df, mean, stdev):
            return (df - mean)/stdev

        # rescale (denormalize) the data
        normalized_filtered_pr_df = normalize_df(raw_filtered_pr_df, norm_param_mean, norm_param_std)

        normalized_ave_player_game = fitted_ss_dict[player_desired_rank].transform(ave_player_game)

        distances, neighbor_indices = scaled_neighbor_ball_tree_dict[player_desired_rank].kneighbors(normalized_ave_player_game)

        normalized_neighbors = normalized_filtered_pr_df.iloc[neighbor_indices[0], :]
        scores = normalized_neighbors.subtract(normalized_ave_player_game).multiply(account_for_lower_better_stats_df.values)
        return scores.mean().to_frame().transpose(), normalized_neighbors

    ave_player_game, player_desired_rank = get_ave_player_game_and_desired_rank(steam_id)

    raw_deficiencies, normalized_neighbors = calculate_raw_deficiencies(ave_player_game, player_desired_rank, fitted_ss_dict, scaled_neighbor_ball_tree_dict)

    # Only keep stats where the player is worse than the average of the k nearest neighbors, then normalize
    raw_deficiencies[raw_deficiencies<0] = 0
    normalized_deficiencies = raw_deficiencies.div(raw_deficiencies.sum(axis=1), axis=0)
    normalized_deficiencies.index += player_desired_rank

    # training
    training_recommendation_weights = rel_feature_importance_df.loc[player_desired_rank:player_desired_rank, :].mul(normalized_deficiencies, axis=1) # the product of feature importance and rel. deficiency gives the recommendation weight
    training_recommendation_weights = training_recommendation_weights.div(training_recommendation_weights.max(axis=1), axis=0)

    training_recommendation_cats = training_recommendation_weights.transpose().nlargest(4, columns=player_desired_rank).index

    # take the 4 largest normalized_deficiencies
    training_recommendation_weights.transpose().nlargest(4, columns=player_desired_rank)

    p = _create_plot(xlabels=training_recommendation_cats.values,
                     grp1_data=ave_player_game.loc[:, training_recommendation_cats].values[0],
                     grp2_data=convert_scaled_to_raw(normalized_neighbors, fitted_ss_dict[player_desired_rank]).loc[:,
                           training_recommendation_cats].mean().values)

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
    show(do_exactly_what_I_want_function(sample_steamid))
