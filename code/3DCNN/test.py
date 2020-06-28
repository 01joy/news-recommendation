from model import CNN
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
import configparser
from tqdm import tqdm

import sys
sys.path.append("..")
import my_utils
import datetime

config = configparser.ConfigParser()
config.read('../config.ini')

mp_train_hist = my_utils.read_pkl(config['DEFAULT']['path_train_hist'])
mp_test_hist = my_utils.read_pkl(config['DEFAULT']['path_test_hist'])

num_used_hists = int(config['DEFAULT']['num_used_hists'])
num_recommendations = int(config['DEFAULT']['num_recommendations'])
num_word_embedding_dims = int(config['DEFAULT']['num_word_embedding_dims'])

num_words = int(config['3DCNN']['num_words'])
sample_batch_size = int(config['3DCNN']['sample_batch_size'])
test_user_batch_size = int(config['3DCNN']['test_user_batch_size'])

# 指定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True   #不全部占满显存, 按需分配
sess = tf.Session(config = tf_config)


def get_test_news_ids():
    test_news_ids = set()
    for user_id, clicked_news_ids in mp_test_hist.items():
        test_news_ids |= set(clicked_news_ids)
    return test_news_ids

def save_prediction(user_ids, news_ids, out):
    preds = list(zip(user_ids, news_ids, out))
    preds = sorted(preds, key = lambda x : x[2], reverse = True)
    mp_pred_news_ids = {}
    for user_id, news_id, score in preds:
        if user_id not in mp_pred_news_ids:
            mp_pred_news_ids[user_id] = []
        mp_pred_news_ids[user_id].append(news_id)

    mp_pred = {}
    for user_id, news_ids in mp_pred_news_ids.items():
        mp_pred[user_id]=[mp_train_hist[user_id], list(set(mp_test_hist[user_id])), mp_pred_news_ids[user_id][:num_recommendations]]

    df_pred = pd.DataFrame(mp_pred.items(), columns = ['user_id', 'news_id'])
    df_pred['train_hist'] = df_pred.apply(lambda x: x['news_id'][0], axis=1)
    df_pred['test_hist'] = df_pred.apply(lambda x: x['news_id'][1], axis=1)
    df_pred['pred_hist'] = df_pred.apply(lambda x: x['news_id'][2], axis=1)
    df_pred = df_pred.drop(columns=['news_id'])

    my_utils.write_pkl(mp_pred, config['3DCNN']['path_pred_pkl'])
    df_pred.to_csv(config['3DCNN']['path_pred_txt'], index = False, sep = '\t')

def batch_test():

    model = CNN(num_used_hists, num_words, num_word_embedding_dims)
    model.create_model()
    model.get_model_summary()

    print('loading pre-trained model...')
    model.model.load_weights(r'%s/3dcnn_word_80_neg_1_epoch_07_val_loss_0.27.model'%config['3DCNN']['path_model_folder'])
    
    print('loading doc embedding...')
    mp_doc_embedding = my_utils.read_pkl(config['DEFAULT']['path_all_news_doc_embedding'])

    test_user_ids = list(mp_test_hist.keys())
    test_news_ids = get_test_news_ids()

    user_ids, news_ids, outs = [], [], []

    print('start predicting...')
    for user_batch_id in tqdm(range(0, len(test_user_ids), test_user_batch_size)):
        user_in = []
        article_in = []
        for user_id in test_user_ids[user_batch_id : user_batch_id + test_user_batch_size]:
            clicked_news_ids = set(mp_test_hist[user_id])
            un_clicked_news_ids = list(test_news_ids - clicked_news_ids)
            user_train_hist = mp_train_hist[user_id]

            user_embedding = []
            if len(user_train_hist) > num_used_hists:
                for news_id in user_train_hist[:num_used_hists]:
                    user_embedding.append(mp_doc_embedding[news_id])
            else:
                for news_id in user_train_hist[:-1]:
                    user_embedding.append(mp_doc_embedding[news_id])
                num_paddings = num_used_hists - len(user_embedding)
                for i in range(num_paddings):
                    user_embedding.append(np.zeros((num_words, num_word_embedding_dims)))
                
            for news_id in random.sample(un_clicked_news_ids, int(config['DEFAULT']['num_test_negatives'])):
                user_in.append(user_embedding)
                article_in.append(mp_doc_embedding[news_id])

                user_ids.append(user_id)
                news_ids.append(news_id)

            for news_id in clicked_news_ids:
                user_in.append(user_embedding)
                article_in.append(mp_doc_embedding[news_id])

                user_ids.append(user_id)
                news_ids.append(news_id)

        # #records必须是batchsize的整数倍，要不然会崩溃：https://stackoverflow.com/a/59971264/2468587
        while len(user_in) % sample_batch_size != 0:
            for user_id in test_user_ids[user_batch_id : user_batch_id + test_user_batch_size]:
                clicked_news_ids = set(mp_test_hist[user_id])
                un_clicked_news_ids = list(test_news_ids - clicked_news_ids)
                user_train_hist = mp_train_hist[user_id]

                user_embedding = []
                if len(user_train_hist) > num_used_hists:
                    for news_id in user_train_hist[:num_used_hists]:
                        user_embedding.append(mp_doc_embedding[news_id])
                else:
                    for news_id in user_train_hist[:-1]:
                        user_embedding.append(mp_doc_embedding[news_id])
                    num_paddings = num_used_hists - len(user_embedding)
                    for i in range(num_paddings):
                        user_embedding.append(np.zeros((num_words, num_word_embedding_dims)))
                
                news_id = random.sample(list(test_news_ids), 1)[0]
                user_in.append(user_embedding)
                article_in.append(mp_doc_embedding[news_id])

                user_ids.append(user_id)
                news_ids.append(news_id)
                if len(user_in) % sample_batch_size == 0:
                    break

        print('#records=%d,batchsize=%d'%(len(user_in), sample_batch_size))
        user_in = np.array(user_in)
        article_in = np.array(article_in)
        user_in = np.resize(user_in, (user_in.shape[0], 1) + user_in.shape[1:])
        article_in = np.resize(article_in, (article_in.shape[0], 1) + article_in.shape[1:])
        
        out = model.model.predict([user_in, article_in], batch_size = sample_batch_size, verbose = 1)
        outs.extend(list(out))
    
    save_prediction(user_ids, news_ids, outs)



# 所有用户一起测试，数据量太大了，memory error
def test():

    model = CNN(num_used_hists, num_words, num_word_embedding_dims)
    model.create_model()
    model.get_model_summary()

    print('loading pre-trained model...')
    model.model.load_weights(r'%s/3dcnn_word_80_neg_1_epoch_14_val_loss_0.26.model'%config['3DCNN']['path_model_folder'])
    
    print('loading doc embedding...')
    mp_doc_embedding = my_utils.read_pkl(config['DEFAULT']['path_all_news_doc_embedding'])

    test_user_ids = list(mp_test_hist.keys())
    test_news_ids = get_test_news_ids()

    user_ids, news_ids = [], []

    print('start predicting...')
    user_in = []
    article_in = []
    for user_id in tqdm(test_user_ids):
        clicked_news_ids = set(mp_test_hist[user_id])
        un_clicked_news_ids = list(test_news_ids - clicked_news_ids)
        user_train_hist = mp_train_hist[user_id]

        user_embedding = []
        if len(user_train_hist) > num_used_hists:
            for news_id in user_train_hist[:num_used_hists]:
                user_embedding.append(mp_doc_embedding[news_id])
        else:
            for news_id in user_train_hist[:-1]:
                user_embedding.append(mp_doc_embedding[news_id])
            num_paddings = num_used_hists - len(user_embedding)
            for i in range(num_paddings):
                user_embedding.append(np.zeros((num_words, num_word_embedding_dims)))
            
        for news_id in random.sample(un_clicked_news_ids, int(config['DEFAULT']['num_test_negatives'])):
            user_in.append(user_embedding)
            article_in.append(mp_doc_embedding[news_id])

            user_ids.append(user_id)
            news_ids.append(news_id)

        for news_id in clicked_news_ids:
            user_in.append(user_embedding)
            article_in.append(mp_doc_embedding[news_id])

            user_ids.append(user_id)
            news_ids.append(news_id)

    user_in = np.array(user_in)
    article_in = np.array(article_in)
    user_in = np.resize(user_in, (user_in.shape[0], 1) + user_in.shape[1:])
    article_in = np.resize(article_in, (article_in.shape[0], 1) + article_in.shape[1:])

    out = model.model.predict([user_in, article_in], batch_size = sample_batch_size, verbose = 1)
    
    save_prediction(user_ids, news_ids, out)

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    batch_test()
    endtime = datetime.datetime.now()
    print('Used time: %f sec.'%((endtime - starttime).seconds))