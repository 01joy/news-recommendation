import sys
sys.path.append("..")
import my_utils

import pickle as pkl
import numpy as np
import pandas as pd
import configparser
from gensim import corpora,models
from tqdm import tqdm

config = configparser.ConfigParser()
config.read('../config.ini')

mp_news_word_lst = my_utils.read_pkl(config['DEFAULT']['path_all_news_word_list'])
mp_train_hist = my_utils.read_pkl(config['DEFAULT']['path_train_hist'])
mp_test_hist = my_utils.read_pkl(config['DEFAULT']['path_test_hist'])

num_used_hists = int(config['DEFAULT']['num_used_hists'])
num_topics = int(config['LDA']['num_topics'])

# 提取训练集中的新闻词表
def get_train_word_lst():
    train_news_ids = set()
    for user_id, news_ids in mp_train_hist.items():
        train_news_ids |= set(news_ids)

    train_word_lst = []
    for news_id in train_news_ids:
        train_word_lst.append(mp_news_word_lst[news_id])
    
    return train_word_lst

# 根据训练集生成LDA模型
def generate_lda_model():
    # 对训练集中的新闻构造词典
    train_word_lst = get_train_word_lst()
    train_dict = corpora.Dictionary(train_word_lst)
    train_dict.filter_extremes(no_below = 1, no_above = 0.9, keep_n = None)
    train_dict.save(config['LDA']['path_train_dict'])

    train_corpus = [train_dict.doc2bow(word_list) for word_list in train_word_lst]

    # 将每篇新闻转换为tf-idf向量
    tfidf_model = models.TfidfModel(train_corpus)
    tfidf_model.save(config['LDA']['path_tfidf_model'])
    tfidf_vectors = tfidf_model[train_corpus]

    #通过tf-idf向量生成LDA模型
    lda_model = models.LdaModel(tfidf_vectors, id2word = train_dict, num_topics = num_topics)
    lda_model.save(config['LDA']['path_lda_model'])


# 用户画像
# 根据每个用户的阅读历史，生成用户感兴趣的主题向量
def generate_user_vector():
    train_dict = corpora.Dictionary.load(config['LDA']['path_train_dict'])
    tfidf_model = models.TfidfModel.load(config['LDA']['path_tfidf_model'])
    lda_model = models.LdaModel.load(config['LDA']['path_lda_model'])

    mp_user_lda_vec = {}
    for user_id, news_ids in tqdm(mp_train_hist.items()):
        user_news_hist_word_list = []
        num_news = len(news_ids)
        for news_id in news_ids[max(0, num_news - num_used_hists) : num_news]: # 仅使用最近阅读过的num_used_hist篇新闻
            user_news_hist_word_list.extend(mp_news_word_lst[news_id])
        
        user_bow = train_dict.doc2bow(user_news_hist_word_list)
        user_tfidf_vec = tfidf_model[user_bow]
        user_lda_vec = lda_model[user_tfidf_vec]
        mp_user_lda_vec[user_id] = user_lda_vec

    my_utils.write_pkl(mp_user_lda_vec, config['LDA']['path_user_lda_vec'])


def train():
    print('generating lda model...')
    generate_lda_model()
    
    print('generating user vector...')
    generate_user_vector()


if __name__ == "__main__":
    train()
