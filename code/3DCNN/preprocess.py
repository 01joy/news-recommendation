import sys
sys.path.append("..")
import my_utils

import pkuseg
import configparser
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors

import datetime


config = configparser.ConfigParser()
config.read('../config.ini')

mp_news_txt = my_utils.read_pkl(config['DEFAULT']['path_all_news_txt'])
mp_news_word_lst = my_utils.read_pkl(config['DEFAULT']['path_all_news_word_list'])
num_words = int(config['3DCNN']['num_words'])
num_word_embedding_dims = int(config['DEFAULT']['num_word_embedding_dims'])


# 将新闻的多个词向量拼起来，相当于一张2d图片
def get_doc_embedding(news_id, word_embeddings):
    doc_vecs = []
    for word in mp_news_word_lst[news_id][:num_words]:
        if word in word_embeddings:
            doc_vecs.append(word_embeddings[word])
        else:
            doc_vecs.append([0] * num_word_embedding_dims)
    for i in range(num_words - len(mp_news_word_lst[news_id])):
        doc_vecs.append([0] * num_word_embedding_dims)
    return np.asarray(doc_vecs)

# 得到新闻的文档向量
def get_all_docs_embedding():
    print('loading word embedding...')
    word_embeddings = KeyedVectors.load_word2vec_format(config['DEFAULT']['path_embedding'], binary=False, encoding="utf8", unicode_errors='ignore')
    mp_doc_embedding = {}
    for news_id in tqdm(mp_news_word_lst.keys()):
        mp_doc_embedding[news_id] = get_doc_embedding(news_id, word_embeddings)
    
    my_utils.write_pkl(mp_doc_embedding, config['DEFAULT']['path_all_news_doc_embedding'])
    return mp_doc_embedding


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    get_all_docs_embedding()
    endtime = datetime.datetime.now()
    print('Used time: %f sec.'%((endtime - starttime).seconds))