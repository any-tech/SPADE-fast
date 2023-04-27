import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def calc_imagewise_metrics(D, y):
    D_list = []
    y_list = []
    for type_test in D.keys():
        D_list.append(D[type_test])
        y_list.append(y[type_test])

    D_list = np.concatenate(D_list)
    y_list = np.concatenate(y_list)

    fpr, tpr, _ = roc_curve(y_list, D_list)
    rocauc = roc_auc_score(y_list, D_list)

    return fpr, tpr, rocauc


def calc_pixelwise_metrics(D, y):
    D_list = []
    y_list = []
    for type_test in D.keys():
        for i in range(len(D[type_test])):
            D_tmp = D[type_test][i]
            y_tmp = y[type_test][i]

            D_list.append(D_tmp)
            y_list.append(y_tmp)

    D_flatten_list = np.array(D_list).reshape(-1)
    y_flatten_list = np.array(y_list).reshape(-1)

    fpr, tpr, _ = roc_curve(y_flatten_list, D_flatten_list)
    rocauc = roc_auc_score(y_flatten_list, D_flatten_list)

    return fpr, tpr, rocauc
