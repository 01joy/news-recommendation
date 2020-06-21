import sys
sys.path.append("..")
import my_utils

import pkuseg
import configparser
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors

config = configparser.ConfigParser()
config.read('../config.ini')

mp_news_txt = my_utils.read_pkl(config['DEFAULT']['path_all_news_txt'])
stop_words = my_utils.read_stop_words(config['DEFAULT']['path_stop_words'])

# 对所有新闻进行分词并去停用词
def cut_news():
    mp_news_word_lst = {}
    seg = pkuseg.pkuseg(model_name='news')  # 程序会自动下载所对应的细领域模型
    for news_id, (news_title, news_content, news_time) in tqdm(mp_news_txt.items()):
        text = news_title + " " + news_content # 新闻标题和正文都用
        # text = news_title # 只用新闻标题
        word_list = list(filter(lambda x: len(x) > 0 and x not in stop_words, map(my_utils.clean_word, seg.cut(text))))
        mp_news_word_lst[news_id] = word_list
    
    my_utils.write_pkl(mp_news_word_lst, config['DEFAULT']['path_all_news_word_list'])


if __name__ == "__main__":
    cut_news()
