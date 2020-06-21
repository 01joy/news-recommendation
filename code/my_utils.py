
import pickle
import re


def read_pkl(path):
    fin = open(path, 'rb')
    obj = pickle.load(fin)
    fin.close()
    return obj


def write_pkl(obj, path):
    fout = open(path,'wb')
    pickle.dump(obj, fout)
    fout.close()


regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
def clean_word(word):
    return regex.sub('', word) # 删除所有不是中文、字母、数字的词


def read_stop_words(path):
    fin = open(path)
    lines = fin.readlines()
    fin.close()
    return set([line.strip() for line in lines])