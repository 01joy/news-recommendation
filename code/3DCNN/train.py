from model import CNN
import os
import numpy as np
import random
import pandas as pd

import tensorflow as tf
import configparser
from tqdm import tqdm

import sys
sys.path.append("..")
import my_utils

config = configparser.ConfigParser()
config.read('../config.ini')

mp_train_hist = my_utils.read_pkl(config['DEFAULT']['path_train_hist'])
num_used_hists = int(config['DEFAULT']['num_used_hists'])
num_word_embedding_dims = int(config['DEFAULT']['num_word_embedding_dims'])
num_words = int(config['3DCNN']['num_words'])
sample_batch_size = int(config['3DCNN']['sample_batch_size'])
num_epochs = int(config['3DCNN']['num_epochs'])
num_train_negatives = int(config['3DCNN']['num_train_negatives'])


# 指定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True   #不全部占满显存, 按需分配
sess = tf.Session(config = tf_config)


def get_train_news_ids():
    train_news_ids = set()
    for user_id, clicked_news_ids in mp_train_hist.items():
        train_news_ids |= set(clicked_news_ids)
    return train_news_ids


def get_negative_news_id(train_news_ids, user_id):
    clicked_news_ids = set(mp_train_hist[user_id])
    un_clicked_news_ids = train_news_ids - clicked_news_ids
    return random.choice(list(un_clicked_news_ids))


def train():

    model = CNN(num_used_hists, num_words, num_word_embedding_dims)
    model.create_model()
    model.get_model_summary()
    
    print('loading doc embedding...')
    mp_doc_embedding = my_utils.read_pkl(config['DEFAULT']['path_all_news_doc_embedding'])
    
    print('constructing input data...')
    train_news_ids = get_train_news_ids()
    user_in = []
    article_in = []
    truth = []
    for user_id, clicked_news_ids in tqdm(mp_train_hist.items()):
        user_embedding = []
        if len(clicked_news_ids) > num_used_hists:
            for news_id in clicked_news_ids[:num_used_hists]:
                user_embedding.append(mp_doc_embedding[news_id])
            for news_id in clicked_news_ids[num_used_hists:]:
                article_in.append(mp_doc_embedding[news_id])
                user_in.append(user_embedding)
                truth.append(1)
                for i in range(num_train_negatives):
                    article_in.append(mp_doc_embedding[get_negative_news_id(train_news_ids, user_id)])
                    user_in.append(user_embedding)
                    truth.append(0)
        else:
            for news_id in clicked_news_ids[:-1]:
                user_embedding.append(mp_doc_embedding[news_id])
            num_paddings = num_used_hists - len(user_embedding)
            for i in range(num_paddings):
                user_embedding.append(np.zeros((num_words, num_word_embedding_dims)))
            article_in.append(mp_doc_embedding[clicked_news_ids[-1]])
            user_in.append(user_embedding)
            truth.append(1)
            for i in range(num_train_negatives):
                article_in.append(mp_doc_embedding[get_negative_news_id(train_news_ids, user_id)])
                user_in.append(user_embedding)
                truth.append(0)

    print('reshaping input data...')
    user_in = np.array(user_in)
    article_in = np.array(article_in)
    user_in = np.resize(user_in, (user_in.shape[0], 1) + user_in.shape[1:])
    article_in = np.resize(article_in, (article_in.shape[0], 1) + article_in.shape[1:])

    print('start training...')
    model.fit_model([user_in, article_in], np.array(truth), sample_batch_size, num_epochs)


if __name__ == "__main__":
    train()
