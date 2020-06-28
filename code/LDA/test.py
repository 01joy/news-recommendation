import sys
sys.path.append("..")
import my_utils

import pickle as pkl
import numpy as np
import pandas as pd
import configparser
from gensim import corpora,models
from gensim.matutils import hellinger
from tqdm import tqdm
import random
import datetime

config = configparser.ConfigParser()
config.read('../config.ini')

mp_news_word_lst = my_utils.read_pkl(config['DEFAULT']['path_all_news_word_list'])
mp_train_hist = my_utils.read_pkl(config['DEFAULT']['path_train_hist'])
mp_test_hist = my_utils.read_pkl(config['DEFAULT']['path_test_hist'])
num_recommendations = int(config['DEFAULT']['num_recommendations'])


def save_prediction(mp_pred):
    df_pred = pd.DataFrame(mp_pred.items(), columns = ['user_id', 'news_id'])
    df_pred['train_hist'] = df_pred.apply(lambda x: x['news_id'][0], axis=1)
    df_pred['test_hist'] = df_pred.apply(lambda x: x['news_id'][1], axis=1)
    df_pred['pred_hist'] = df_pred.apply(lambda x: x['news_id'][2], axis=1)
    df_pred = df_pred.drop(columns=['news_id'])

    my_utils.write_pkl(mp_pred, config['LDA']['path_pred_pkl'])
    df_pred.to_csv(config['LDA']['path_pred_txt'], index = False, sep = '\t')


def get_test_news_ids():
    test_news_ids = set()
    for user_id, clicked_news_ids in mp_test_hist.items():
        test_news_ids |= set(clicked_news_ids)
    return test_news_ids

def test():
    train_dict = corpora.Dictionary.load(config['LDA']['path_train_dict'])
    tfidf_model = models.TfidfModel.load(config['LDA']['path_tfidf_model'])
    lda_model = models.LdaModel.load(config['LDA']['path_lda_model'])
    mp_user_lda_vec = my_utils.read_pkl(config['LDA']['path_user_lda_vec'])

    test_news_ids = get_test_news_ids()

    print('calculating news lda vector...')
    mp_test_news_lda_vec = {}
    for news_id in tqdm(test_news_ids):
        word_list = mp_news_word_lst[news_id]
        news_bow = train_dict.doc2bow(word_list)
        news_tfidf_vec = tfidf_model[news_bow]
        news_lda_vec = lda_model[news_tfidf_vec]
        mp_test_news_lda_vec[news_id] = news_lda_vec

    print('calculating hellinger distance between user and news...')
    mp_pred = {} # key=user_id, value=[[true_hist], [pred_topn]]
    for user_id, news_ids in tqdm(mp_test_hist.items()):
        train_news_ids = mp_train_hist[user_id]
        user_lda_vec = mp_user_lda_vec[user_id]

        clicked_news_ids = set(news_ids)
        un_clicked_news_ids = list(test_news_ids - clicked_news_ids)

        candidates = []

        for news_id in random.sample(un_clicked_news_ids, int(config['DEFAULT']['num_test_negatives'])):
            news_lda_vec = mp_test_news_lda_vec[news_id]
            dist = hellinger(user_lda_vec, news_lda_vec)
            candidates.append([news_id, dist])

        for news_id in clicked_news_ids:
            news_lda_vec = mp_test_news_lda_vec[news_id]
            dist = hellinger(user_lda_vec, news_lda_vec)
            candidates.append([news_id, dist])

        candidates = sorted(candidates, key = lambda x : x[1])
        topn_candidates = [x[0] for x in candidates[:num_recommendations]]
        mp_pred[user_id] = [train_news_ids, list(clicked_news_ids), topn_candidates]
    
    save_prediction(mp_pred)

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    test()
    endtime = datetime.datetime.now()
    print('Used time: %f sec.'%((endtime - starttime).seconds))