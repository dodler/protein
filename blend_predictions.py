import pickle
import sys
import numpy as np
from fastai_learn import learner, name_label_dict, LABELS, SAMPLE
import pandas as pd
import scipy.optimize as opt


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


preds = []
for p in sys.argv[1:]:
    preds_t = pickle.load(open(p, 'rb'))

    preds_t = np.stack(preds_t, axis=-1)
    preds_t = sigmoid_np(preds_t)
    preds.append(preds_t)

preds = np.array(preds).mean(axis=0)
pred_t = preds.max(axis=-1)  # max works better for F1 macro score


def save_pred(pred, th=0.5, fname='protein_classification.csv'):
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line > th)[0]]))
        pred_list.append(s)

    sample_df = pd.read_csv(SAMPLE)
    sample_list = list(sample_df.Id)
    pred_dic = dict((key, value) for (key, value)
                    in zip(learner.data.test_ds.fnames, pred_list))
    pred_list_cor = [pred_dic[id] for id in sample_list]
    df = pd.DataFrame({'Id': sample_list, 'Predicted': pred_list_cor})
    df.to_csv(fname, header=True, index=False)


th_t = np.array([0.565, 0.39, 0.55, 0.345, 0.33, 0.39, 0.33, 0.45, 0.38, 0.39,
                 0.34, 0.42, 0.31, 0.38, 0.49, 0.50, 0.38, 0.43, 0.46, 0.40,
                 0.39, 0.505, 0.37, 0.47, 0.41, 0.545, 0.32, 0.1])
print('Fractions: ', (pred_t > th_t).mean(axis=0))
save_pred(pred_t, th_t)

lb_prob = [
    0.362397820, 0.043841336, 0.075268817, 0.059322034, 0.075268817,
    0.075268817, 0.043841336, 0.075268817, 0.010000000, 0.010000000,
    0.010000000, 0.043841336, 0.043841336, 0.014198783, 0.043841336,
    0.010000000, 0.028806584, 0.014198783, 0.028806584, 0.059322034,
    0.010000000, 0.126126126, 0.028806584, 0.075268817, 0.010000000,
    0.222493880, 0.028806584, 0.010000000]


def Count_soft(preds, th=0.5, d=50.0):
    preds = sigmoid_np(d * (preds - th))
    return preds.mean(axis=0)


def fit_test(x, y):
    params = 0.5 * np.ones(len(name_label_dict))
    wd = 1e-5
    error = lambda p: np.concatenate((Count_soft(x, p) - y,
                                      wd * (p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p


def F1_soft(preds, targs, th=0.5, d=50.0):
    preds = sigmoid_np(d * (preds - th))
    targs = targs.astype(np.float)
    score = 2.0 * (preds * targs).sum(axis=0) / ((preds + targs).sum(axis=0) + 1e-6)
    return score


def fit_val(x, y):
    params = 0.5 * np.ones(len(name_label_dict))
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(x, y, p) - 1.0,
                                      wd * (p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p


th_t = fit_test(pred_t, lb_prob)
th_t[th_t < 0.1] = 0.1
print('Thresholds: ', th_t)
print('Fractions: ', (pred_t > th_t).mean(axis=0))
print('Fractions (th = 0.5): ', (pred_t > 0.5).mean(axis=0))

save_pred(pred_t, th_t, 'protein_classification_f.csv')
# save_pred(pred_t, th, 'protein_classification_v.csv')
save_pred(pred_t, 0.5, 'protein_classification_05.csv')
save_pred(pred_t, 0.3, 'protein_classification_03.csv')

class_list = [8, 9, 10, 15, 20, 24, 27]
# for i in class_list:
#     th_t[i] = th[i]
save_pred(pred_t, th_t, 'protein_classification_c.csv')

labels = pd.read_csv(LABELS).set_index('Id')
label_count = np.zeros(len(name_label_dict))
for label in labels['Target']:
    l = [int(i) for i in label.split()]
    label_count += np.eye(len(name_label_dict))[l].sum(axis=0)
label_fraction = label_count.astype(np.float) / len(labels)
label_count, label_fraction

th_t = fit_test(pred_t, label_fraction)
th_t[th_t < 0.05] = 0.05
print('Thresholds: ', th_t)
print('Fractions: ', (pred_t > th_t).mean(axis=0))
save_pred(pred_t, th_t, 'protein_classification_t.csv')