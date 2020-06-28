import sys
sys.path.append("..")
import my_utils

import matplotlib
matplotlib.use('AGG')#或者PDF, SVG或PS
import matplotlib.pyplot as plt
import seaborn as sns
import configparser
import ml_metrics as metrics

config = configparser.ConfigParser()
config.read('../config.ini')

num_recommendations = int(config['DEFAULT']['num_recommendations'])

def evaluate(mp_pred, fig_name = ''):

    hit_k = [[] for i in range(num_recommendations)] # hit@k
    ap_k = [[] for i in range(num_recommendations)] # ap@k

    gt = 0
    for user_id, (train_news_ids, test_news_ids, topn_candidates) in mp_pred.items():
        true_ids = set(test_news_ids)
        gt += len(true_ids)
        for k in range(num_recommendations):
            pred_ids = set(topn_candidates[:k+1])
            hit = len(true_ids & pred_ids)
            hit_k[k].append(hit)
            ap = metrics.apk(test_news_ids, topn_candidates[:k+1], k+1)
            ap_k[k].append(ap)

    hr_k=[sum(h)/gt for h in hit_k]
    map_k=[sum(a)/len(mp_pred) for a in ap_k]
    ks=[i + 1 for i in range(num_recommendations)]

    plt.figure()
    plt.plot(ks,hr_k,'b',label='HR_%s@K'%fig_name)
    plt.title('HR_%s@K'%fig_name)
    plt.legend()
    plt.savefig('HR_%s@K'%fig_name)

    plt.figure()
    plt.plot(ks,map_k,'b',label='MAP_%s@K'%fig_name)
    plt.title('MAP_%s@K'%fig_name)
    plt.legend()
    plt.savefig('MAP_%s@K'%fig_name)

    return hr_k, map_k


if __name__ == "__main__":
    mp_pred = my_utils.read_pkl(config['3DCNN']['path_pred_pkl'])
    fig_name = '3DCNN_news_title_body'
    hr_k, map_k = evaluate(mp_pred, fig_name)
    
