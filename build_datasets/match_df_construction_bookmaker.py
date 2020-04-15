# modify this for your own path
SCRIPT_PATH = '/Users/weijianli/tennis_prediction/jingya/tennis_match_prediction/build_datasets/sackmann'
TOUR = 'atp'
COUNT = False
START_YEAR = 2000
ONLY_PBP = 0
# DATE = '11_26'

import sys

sys.path.insert(0, SCRIPT_PATH)
import tennisGameProbability, tennisMatchProbability, tennisSetProbability, tennisTiebreakProbability
from tennisMatchProbability import matchProb
from helper_functions import *
from data_functions import *
import pandas as pd

pd.options.mode.chained_assignment = None
import numpy as np
import re
import math
import copy
from multiprocessing import Pool
import signal


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def f(atp_all_matches1, k1):
    kk1, kk2, a = 0, 0, 0
    for k2 in xrange(1, 101):
        atp_all_matches = stephanie_generate_elo(atp_all_matches1, k1, k2)
        df = atp_all_matches
        df['elo_diff'] = [df['w_elo'][i] - df['l_elo'][i] for i in xrange(len(df))]
        df2 = df[df['match_year'] == 2017].reset_index(drop=True)
        acc = sum((df2['elo_diff'] > 0)) / float(len(df2))
        print 'baseline: ', acc, "k1, k2: ", k1, " ", k2
        if acc > a:
            a = acc
            kk1 = k1
            kk2 = k2
    return (a, kk1, kk2)


def grid_search_k1k2(atp_all_matches1):
    try:
        K1, K2, ACC = 0, 0, 0
        pool = Pool(processes=10, initializer=initializer)
        # multiple_results = pool.map(f, xrange(1, 101))
        multiple_results = [pool.apply_async(f, args=(atp_all_matches1, i)) for i in xrange(1, 101)]
        pool.close()
        pool.join()
        results = [i.get() for i in multiple_results]
        for result in results:
            if ACC < result[0]:
                K1 = result[1]
                K2 = result[2]
        # for k1 in xrange(1, 101):
        #     for k2 in xrange(1, 101):
        #         atp_all_matches = stephanie_generate_elo(atp_all_matches1, k1, k2)
        #         df = atp_all_matches
        #         df['elo_diff'] = [df['w_elo'][i] - df['l_elo'][i] for i in xrange(len(df))]
        #         df2 = df[df['match_year']==2017].reset_index(drop=True)
        #         acc = sum((df2['elo_diff']>0))/float(len(df2))
        #         print 'baseline: ',  acc, "k1, k2: ", k1, " ", k2
        #         if acc > ACC:
        #             ACC = acc
        #             K1 = k1
        #             K2 = k2
        print "best accuracy: ", ACC, "k1, k2: ", K1, K2
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()

def prepare_data(df):
    all_fields = {}
    column1, column2, column3, column4 = [], [], [], []
    for i, row in df.iterrows():
        # tournament level
        tmp_df = df.iloc[:i]
        p1_w_num_in_tournament = len(tmp_df[(tmp_df['w_name'] == row['w_name']) & (tmp_df['tny_name'] == row['tny_name'])])
        p1_l_num_in_tournament = len(tmp_df[(tmp_df['l_name'] == row['w_name']) & (tmp_df['tny_name'] == row['tny_name'])])
        s_tournament_w = 0
        if (p1_w_num_in_tournament + p1_l_num_in_tournament) != 0:
            s_tournament_w = float(p1_w_num_in_tournament) / (p1_w_num_in_tournament + p1_l_num_in_tournament)
        column1.append(s_tournament_w)

        p2_w_num_in_tournament = len(tmp_df[(tmp_df['w_name'] == row['l_name']) & (tmp_df['tny_name'] == row['tny_name'])])
        p2_l_num_in_tournament = len(tmp_df[(tmp_df['l_name'] == row['l_name']) & (tmp_df['tny_name'] == row['tny_name'])])
        s_tournament_l = 0
        if (p2_w_num_in_tournament + p2_l_num_in_tournament) != 0:
            s_tournament_l = float(p2_w_num_in_tournament) / (p2_w_num_in_tournament + p2_l_num_in_tournament)
        column2.append(s_tournament_l)

        len_win_w = sum(tmp_df[(tmp_df['w_name'] == row['w_name'])]['match_duration'])
        len_lose_w = sum(tmp_df[(tmp_df['l_name'] == row['w_name'])]['match_duration'])
        num_win_w = len(tmp_df[(tmp_df['w_name'] == row['w_name'])])
        num_lose_w = len(tmp_df[(tmp_df['l_name'] == row['w_name'])])
        s_match_duration_w = 0
        if len_win_w != 0 and num_lose_w != 0:
            s_match_duration_w = float(len_lose_w) * num_win_w / len_win_w / num_lose_w
        column3.append(s_match_duration_w)

        len_win_l = sum(tmp_df[(tmp_df['w_name'] == row['l_name'])]['match_duration'])
        len_lose_l = sum(tmp_df[(tmp_df['l_name'] == row['l_name'])]['match_duration'])
        num_win_l = len(tmp_df[(tmp_df['w_name'] == row['l_name'])])
        num_lose_l = len(tmp_df[(tmp_df['l_name'] == row['l_name'])])
        s_match_duration_l = 0
        if len_win_l != 0 and num_lose_l != 0:
            s_match_duration_l = float(len_lose_l) * num_win_l / len_win_l / num_lose_l
        column4.append(s_match_duration_l)


    df['s_tournament_w'] = column1
    df['s_tournament_l'] = column2
    df['s_match_duration_w'] = column3
    df['s_match_duration_l'] = column4

    df.to_csv("../my_data/bookmaker_delta4.csv")


if __name__ == '__main__':
    print 'main'
    atp_year_list = []
    # for i in xrange(2001,2019):
    #     atp_year_list.append(pd.read_csv("../my_data/.csv".format(i)))
    df = pd.read_csv("../my_data/bookmaker_delta4.csv")
    

    # these may be changes specific to atp dataframe; normalize_name() is specific to atp/wta...
    df = df.rename(columns={'winner': 'w_name', 'loser': 'l_name', 'tourney_year_id': 'tny_id', \
                            'tourney_name_x': 'tny_name', 'start_date': 'tny_date', \
                            'winner_first_serve_points_won': 'w_1stWon', \
                            'winner_second_serve_points_won': 'w_2ndWon', \
                            'loser_first_serve_points_won': 'l_1stWon', \
                            'loser_second_serve_points_won': 'l_2ndWon', \
                            'match_index': 'match_num', 'winner_player_id': 'winner_id', 'loser_player_id': 'loser_id', \
                            'Binary ELO Delta Winner': 'w_delta2', \
                            'Binary ELO Delta Loser': 'l_delta2', \
                            'M4 Delta Winner': 'w_delta1', \
                            'M4 Delta Loser': 'l_delta1', \
                            })
    # prepare_data(df)
    df['w_name'] = [normalize_name(x, tour=TOUR) for x in df['w_name']]
    df['l_name'] = [normalize_name(x, tour=TOUR) for x in df['l_name']]
    df['tny_name'] = ['Davis Cup' if 'Davis Cup' in s else s for s in df['tny_name']]
    df['tny_name'] = [s.replace('Australian Chps.', 'Australian Open').replace('Australian Open-2', \
                                                                               'Australian Open').replace(
        'U.S. National Chps.', 'US Open') for s in df['tny_name']]

    ret_strings = ('ABN', 'DEF', 'In Progress', 'RET', 'W/O', ' RET', ' W/O', 'nan', 'walkover')
    abd_strings = ('abandoned', 'ABN', 'ABD', 'DEF', 'def', 'unfinished', 'Walkover')
    atp_all_matches = format_match_df(df, ret_strings=ret_strings, abd_strings=abd_strings)

    # do the grid search for optimal k1 k2 in Stephanie's rating update equation
    # grid_search_k1k2(atp_all_matches)
    # atp_all_matches = generate_elo(atp_all_matches, 0)
    # atp_all_matches = generate_elo(atp_all_matches, 1)
    atp_all_matches = stephanie_generate_elo(atp_all_matches, 2.5, 24, 0)
    atp_all_matches = stephanie_generate_elo(atp_all_matches, 2.5, 24, 1)
    df = atp_all_matches
    # df['elo_diff'] = [df['w_elo'][i] - df['l_elo'][i] for i in xrange(len(df))]
    df['elo_diff'] = df['w_elo'] - df['l_elo']
    df['sf_elo_diff'] = [df['w_sf_elo'][i] - df['l_sf_elo'][i] for i in xrange(len(df))]
    df['elo_diff_538'] = [df['w_elo_538'][i] - df['l_elo_538'][i] for i in xrange(len(df))]
    df['sf_elo_diff_538'] = [df['w_sf_elo_538'][i] - df['l_sf_elo_538'][i] for i in xrange(len(df))]
    df2 = df[df['match_year'] == 2017].reset_index(drop=True)
    print 'elo baseline: ', sum((df2['elo_diff']>0)) / float(len(df2)), "k1, k2: ", 2.5, " ", 24
    print 'surface elo baseline: ', sum(df2['sf_elo_diff']>0)/float(len(df2))
    print 'elo 538 baseline: ',  sum((df2['elo_diff_538']>0))/float(len(df2))
    print 'surface elo 538 baseline: ', sum(df2['sf_elo_diff_538']>0)/float(len(df2))


# useless

    # generate tourney stats from one year behind START_DATE for stats_52
    # get elo with constant and dynamic K
    # start_ind = atp_all_matches[atp_all_matches['match_year'] >= START_YEAR - 1].index[0]
    # atp_all_matches = generate_elo(atp_all_matches, 0)
    # atp_all_matches = generate_elo(atp_all_matches, 1)
    # atp_all_matches = generate_52_stats(atp_all_matches,start_ind)
    # atp_all_matches = generate_52_adj_stats(atp_all_matches,start_ind)
    # atp_all_matches = generate_tny_stats(atp_all_matches,start_ind)

    # print 'adj stats: ', atp_all_matches[atp_all_matches['match_year']==2014][['w_52_s_adj','w_52_r_adj']]
    # print 'now: ', atp_all_matches[['match_year','match_month','w_name','l_name','w_52_s_adj','l_52_s_adj','l_52_svpt']].loc[137969]

    # Combine all the matches that have pbp (point by point) information into one dataframe
    # and clean up columns in preparation for merging with all_atp_matches

    # pbp_matches_archive = pd.read_csv("../my_data/pbp/pbp_matches_atp_main_archive.csv")
    # pbp_matches_archive_old = pd.read_csv("../my_data/pbp/pbp_matches_atp_main_archive_old.csv")
    # pbp_matches_current = pd.read_csv("../my_data/pbp/pbp_matches_atp_main_current.csv")
    # pbp_matches = pd.concat([pbp_matches_archive_old.loc[:932],pbp_matches_archive,pbp_matches_current])
    # pbp_matches.winner = pbp_matches.winner - 1
    # pbp_matches = pbp_matches.reset_index(); del pbp_matches['index']
    # pbp_matches = format_pbp_df(pbp_matches,tour=TOUR)


    # dictionary with each key as 'w_name'+'l_name'+'match_year'+'score' to connect pbp
    # strings to atp_all_matches (I removed parentheses terms from tb scores)
    # duplicates = ['Janko Tipsarevic Kei Nishikori 2011 6-4 6-4', 'Robin Soderling Michael Berrer 2011 6-3 7-6',
    #             'Juan Martin Kevin Anderson 2011 6-4 6-4', 'Philipp Kohlschreiber Mikhail Youzhny 2011 6-4 6-2',
    #             'Philipp Kohlschreiber Olivier Rochus 2012 6-1 6-4', 'Viktor Troicki Radek Stepanek 2012 2-6 6-4 6-3',
    #             'Gilles Simon Grigor Dimitrov 2012 6-3 6-3', 'Alexandr Dolgopolov Gilles Simon 2012 6-3 6-4',
    #             'Fabio Fognini Tommy Haas 2013 6-2 6-4', 'Richard Gasquet Florian Mayer 2013 6-3 7-6',
    #             'Novak Djokovic Rafael Nadal 2013 6-3 6-4', 'Tomas Berdych Gael Monfils 2015 6-1 6-4',
    #             'Novak Djokovic Rafael Nadal 2015 6-3 6-3']
    # collision_d = dict(zip(duplicates, [0] * len(duplicates)))

    # # connects the two dataframes on match keys and reformats columns fro w/l to p0/p1
    # cols = ['_name','_elo','_sf_elo','_elo_538','_sf_elo_538','_52_swon','_52_svpt','_52_rwon',\
    #     '_52_rpt','_sf_52_swon','_sf_52_svpt','_sf_52_rwon','_sf_52_rpt','_52_s_adj','_52_r_adj']
    # df = connect_df(match_df=atp_all_matches,pbp_df=pbp_matches,col_d=collision_d,player_cols=cols,\
    #                 start_year=START_YEAR)

    # df = atp_all_matches

    # df['elo_diff'] = [df['w_elo'][i] - df['l_elo'][i] for i in xrange(len(df))]
    # df['sf_elo_diff'] = [df['w_sf_elo'][i] - df['l_sf_elo'][i] for i in xrange(len(df))]
    # df['elo_diff_538'] = [df['w_elo_538'][i] - df['l_elo_538'][i] for i in xrange(len(df))]
    # df['sf_elo_diff_538'] = [df['w_sf_elo_538'][i] - df['l_sf_elo_538'][i] for i in xrange(len(df))]

    # generate win probabilities from logit of elo/s_elo 538 differences, trained on 2011-2013 data
    # df = generate_logit_probs(df,cols=['elo_diff_538','sf_elo_diff_538'],col_name='logit_elo_538_prob')

    # df = generate_logit_probs(df,cols=['elo_diff','sf_elo_diff'],col_name='logit_elo_prob')
    # df = generate_logit_probs(df,cols=['elo_diff'],col_name='logit_elo_diff_prob')
    # df = generate_logit_probs(df,cols=['elo_diff_538'],col_name='logit_elo_diff_538_prob')


    # dataframe with only matches that have pbp
    # if ONLY_PBP:
    #     df = df[df['pbp']!='None']
    # else:
    #     df = df[df['winner']!='None']

    # df = df.reset_index(drop=True)

    # cols = ['52_s_adj','52_r_adj']
    # test_df = generate_JS_stats(df,cols)

    # keep relevant columns
    # df = df[['tny_id','tny_name','surface','tny_date','match_year','match_month',
    #          u'p0_name', u'p1_name', u'p0_elo',
    #          u'p1_elo', u'p0_sf_elo', u'p1_sf_elo', u'p0_elo_538', u'p1_elo_538',
    #          u'p0_sf_elo_538', u'p1_sf_elo_538', u'p0_52_swon',u'p0_52_svpt',
    #          u'p1_52_swon', u'p1_52_svpt', u'p0_52_rwon', u'p0_52_rpt',
    #          u'p1_52_rwon', u'p1_52_rpt',
    #          u'elo_diff', u'sf_elo_diff',
    #          u'elo_diff_538', u'sf_elo_diff_538',
    #          u'p0_s_pct', u'p0_r_pct', u'p1_s_pct', u'p1_r_pct',
    #          u'p0_s_pct_JS', u'p1_s_pct_JS', u'p0_r_pct_JS', u'p1_r_pct_JS',
    #          u'p0_sf_52_swon', u'p0_sf_52_svpt',u'p1_sf_52_swon', u'p1_sf_52_svpt',
    #          u'p0_sf_52_rwon', u'p0_sf_52_rpt', u'p1_sf_52_rwon', u'p1_sf_52_rpt',
    #          u'p0_sf_s_pct', u'p0_sf_r_pct', u'p1_sf_s_pct', u'p1_sf_r_pct',
    #          u'p0_sf_s_pct_JS', u'p1_sf_s_pct_JS', u'p0_sf_r_pct_JS', u'p1_sf_r_pct_JS',
    #          u'p0_52_s_adj',u'p0_52_r_adj',u'p1_52_s_adj',u'p1_52_r_adj',
    #          u'p0_52_s_adj_JS',u'p0_52_r_adj_JS',u'p1_52_s_adj_JS',u'p1_52_r_adj_JS',
    #          u'avg_52_s', u'avg_52_r', u'sf_avg_52_s', u'sf_avg_52_r',
    #          'tny_stats','best_of','score','pbp',
    #          'logit_elo_538_prob', #'logit_elo_prob','logit_elo_diff_prob','logit_elo_diff_538_prob',
    #          'winner']]

    # binary indicator for whether player 0 won
    # df['winner'] = [1-winner for winner in df['winner']]

    # generate serving probabilities for Klaassen-Magnus model
    # df['match_id'] = range(len(df))
    # df['tny_stats'] = [df['avg_52_s'][i] if df['tny_stats'][i]==0 else df['tny_stats'][i] for i in xrange(len(df))]
    # df['p0_s_kls'] = df['tny_stats']+(df['p0_s_pct']-df['avg_52_s']) - (df['p1_r_pct']-df['avg_52_r'])
    # df['p1_s_kls'] = df['tny_stats']+(df['p1_s_pct']-df['avg_52_s']) - (df['p0_r_pct']-df['avg_52_r'])
    # df['p0_s_kls_JS'] = df['tny_stats']+(df['p0_s_pct_JS']-df['avg_52_s']) - (df['p1_r_pct_JS']-df['avg_52_r'])
    # df['p1_s_kls_JS'] = df['tny_stats']+(df['p1_s_pct_JS']-df['avg_52_s']) - (df['p0_r_pct_JS']-df['avg_52_r'])
    # df['p0_sf_s_kls'] = df['tny_stats']+(df['p0_sf_s_pct']-df['sf_avg_52_s']) - (df['p1_sf_r_pct']-df['sf_avg_52_r'])
    # df['p1_sf_s_kls'] = df['tny_stats']+(df['p1_sf_s_pct']-df['sf_avg_52_s']) - (df['p0_sf_r_pct']-df['sf_avg_52_r'])
    # df['p0_sf_s_kls_JS'] = df['tny_stats']+(df['p0_sf_s_pct_JS']-df['sf_avg_52_s']) - (df['p1_sf_r_pct_JS']-df['sf_avg_52_r'])
    # df['p1_sf_s_kls_JS'] = df['tny_stats']+(df['p1_sf_s_pct_JS']-df['sf_avg_52_s']) - (df['p0_sf_r_pct_JS']-df['sf_avg_52_r'])
    # df['p0_s_kls_adj'] = df['tny_stats']+(df['p0_52_s_adj']) - (df['p1_52_r_adj'])
    # df['p1_s_kls_adj'] = df['tny_stats']+(df['p1_52_s_adj']) - (df['p0_52_r_adj'])
    # df['p0_s_kls_adj_JS'] = df['tny_stats']+(df['p0_52_s_adj_JS']) - (df['p1_52_r_adj_JS'])
    # df['p1_s_kls_adj_JS'] = df['tny_stats']+(df['p1_52_s_adj_JS']) - (df['p0_52_r_adj_JS'])


    # generate match probabilities and z-scores for Klaassen method, with and w/o JS estimators
    # df['match_prob_kls'] = [matchProb(row['p0_s_kls'],1-row['p1_s_kls']) for i,row in df.iterrows()]
    # df['match_prob_kls_JS'] = [matchProb(row['p0_s_kls_JS'],1-row['p1_s_kls_JS']) for i,row in df.iterrows()]
    # df['match_prob_sf_kls'] = [matchProb(row['p0_sf_s_kls'],1-row['p1_sf_s_kls']) for i,row in df.iterrows()]
    # df['match_prob_sf_kls_JS'] = [matchProb(row['p0_sf_s_kls_JS'],1-row['p1_sf_s_kls_JS']) for i,row in df.iterrows()]
    # df['match_prob_adj_kls'] = [matchProb(row['p0_s_kls_adj'],1-row['p1_s_kls_adj']) for i,row in df.iterrows()]
    # df['match_prob_adj_kls_JS'] = [matchProb(row['p0_s_kls_adj_JS'],1-row['p1_s_kls_adj_JS']) for i,row in df.iterrows()]

    # print "useless stuff"
    # # generate win probabilities from elo differences
    # df['elo_prob'] = [(1 + 10 ** (diff / -400.)) ** -1 for diff in df['elo_diff']]
    # df['elo_prob_538'] = [(1 + 10 ** (diff / -400.)) ** -1 for diff in df['elo_diff_538']]
    # df['sf_elo_prob'] = [(1 + 10 ** (diff / -400.)) ** -1 for diff in df['sf_elo_diff']]
    # df['sf_elo_prob_538'] = [(1 + 10 ** (diff / -400.)) ** -1 for diff in df['sf_elo_diff_538']]

    # # elo-induced serve percentages
    # # df = generate_elo_induced_s(df, 'elo',start_ind=0)
    # # df = generate_elo_induced_s(df, 'logit_elo_538',start_ind=0)

    # # depending on ONLY_PBP, this will have point-by-point matches, or all
    # # tour-level matches from START_DATE to present
    # # name = 'elo_with_surface_pbp_' if ONLY_PBP else 'elo_with_surface'
    # # substr = '_baseline'
    # # print name + substr + '.csv saved to my_data'
    # # df.to_csv('../my_data/'+name+substr+'.csv')

    # # ************Evaluation*******************
    # # from pre-match jupyter notebook

    # # df = pd.read_csv('../my_data/elo_pbp_with_surface_11_26_dynamic_rating_tny_level_wrong.csv')
    # # del df['Unnamed: 0']

    # # currently looking at 2014 tour-level matches, excluding Davis Cup
    # df = df[df['match_year'] == 2017].reset_index(drop=True)

    # print 'elo baseline: ', sum((df['elo_diff'] > 0)) / float(len(df))
    # # print log_loss(df['winner'],[(1+10**(diff/-400.))**-1 for diff in df['elo_diff']])
    # # print log_loss(df['winner'],[(1+10**(diff/-400.))**-1 for diff in df['sf_elo_diff']])
    # print 'surface elo baseline: ', sum(df['sf_elo_diff'] > 0) / float(len(df))
    # print 'elo 538 baseline: ', sum((df['elo_diff_538'] > 0)) / float(len(df))
    # # print log_loss(df['winner'],[(1+10**(diff/-400.))**-1 for diff in df['elo_diff_538']])
    # # print log_loss(df['winner'],[(1+10**(diff/-400.))**-1 for diff in df['sf_elo_diff_538']])
    # print 'surface elo 538 baseline: ', sum(df['sf_elo_diff_538'] > 0) / float(len(df))
