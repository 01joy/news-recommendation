import sys
sys.path.append("..")
import my_utils
import os
import matplotlib
matplotlib.use('AGG')#或者PDF, SVG或PS
import matplotlib.pyplot as plt
import seaborn as sns
import configparser
import ml_metrics as metrics
import numpy as np
import pandas as pd

config = configparser.ConfigParser()
config.read('../config.ini')

num_recommendations = int(config['DEFAULT']['num_recommendations'])

def evaluate_single_model(mp_pred):

    hit_k = [[] for i in range(num_recommendations)] # hit@k
    ap_k = [[] for i in range(num_recommendations)] # ap@k

    prec_k = [[] for i in range(num_recommendations)]
    recall_k = [[] for i in range(num_recommendations)]
    ap_k = [[] for i in range(num_recommendations)]

    for user_id, (train_news_ids, test_news_ids, topn_candidates) in mp_pred.items():
        true_ids = set(test_news_ids)
        for k in range(num_recommendations):
            pred_ids = set(topn_candidates[:k+1])
            hit = len(true_ids & pred_ids)
            prec_k[k].append(hit / (k + 1))
            recall_k[k].append(hit / len(true_ids))

            ap = 0
            for n in range(0, k + 1):
                if topn_candidates[n] in true_ids: # 每个召回率水平的准确度均值
                    ap += prec_k[n][-1]
            ap_k[k].append(ap / (k + 1))

    mean_prec_k = [np.mean(pks) for pks in prec_k]
    mean_recall_k = [np.mean(rks) for rks in recall_k]
    mean_ap_k = [np.mean(aks) for aks in ap_k]

    return mean_prec_k, mean_recall_k, mean_ap_k


def plot_evaluation(path_result_folder, df_prec, df_recall, df_ap):
    cols = list(range(1, num_recommendations + 1))
    df_prec.columns = df_recall.columns = df_ap.columns = cols

    plt.figure()
    df_prec.T.plot(legend = True)
    plt.title('Precision@N')
    plt.xlabel('Top N')
    plt.legend()
    plt.savefig(path_result_folder + '/Prec@N')

    plt.figure()
    df_recall.T.plot(legend = True)
    plt.title('Recall@N')
    plt.xlabel('Top N')
    plt.legend()
    plt.savefig(path_result_folder + '/Recall@N')


    plt.figure()
    df_ap.T.plot(legend = True)
    plt.title('MAP@N')
    plt.xlabel('Top N')
    plt.legend()
    plt.savefig(path_result_folder + '/MAP@N')


def save_evaluation(path_result_folder, model_names, mean_prec_ks, mean_recall_ks, mean_ap_ks):
    cols = ['Top%d'%(i + 1) for i in range(num_recommendations)]

    df_prec = pd.DataFrame(index = model_names, data = mean_prec_ks, columns = cols)
    df_recall = pd.DataFrame(index = model_names, data = mean_recall_ks, columns = cols)
    df_ap = pd.DataFrame(index = model_names, data = mean_ap_ks, columns = cols)

    df_prec.to_csv(path_result_folder + '/prec@N.csv')
    df_recall.to_csv(path_result_folder + '/recall@N.csv')
    df_ap.to_csv(path_result_folder + '/map@N.csv')

    return df_prec, df_recall, df_ap

def evaluate(path_result_folder):
    files = os.listdir(path_result_folder)

    model_names, mean_prec_ks, mean_recall_ks, mean_ap_ks = [], [], [], []

    for f in files:
        file_name, extension = os.path.splitext(f)
        if extension == '.pkl':
            mp_pred = my_utils.read_pkl(r'%s/%s'%(path_result_folder, f))
            mean_prec_k, mean_recall_k, mean_ap_k = evaluate_single_model(mp_pred)
            model_names.append(file_name)
            mean_prec_ks.append(mean_prec_k)
            mean_recall_ks.append(mean_recall_k)
            mean_ap_ks.append(mean_ap_k)

    df_prec, df_recall, df_ap = save_evaluation(path_result_folder, model_names, mean_prec_ks, mean_recall_ks, mean_ap_ks)
    plot_evaluation(path_result_folder, df_prec, df_recall, df_ap)

if __name__ == "__main__":
    path_result_folder = r'../../result'
    evaluate(path_result_folder)
    
