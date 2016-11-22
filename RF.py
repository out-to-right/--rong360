# coding=utf-8

'''
author: ShiLei Miao
'''

import numpy as np
from numpy import *
import pandas as pd
from pandas import *
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn import metrics


os.chdir(r'E:\PycharmProjects\Rong360\dta')

def loadDataSetT(path):
    data = pd.read_csv(path)
    dataSet = data.values[0:,2:]
    dataLabel = data.values[0:,1:2]        
    return dataSet,dataLabel

def transLabel(Mat_Labels):
    labels = []
    for item in Mat_Labels:
        labels.append(item[0])
    labels = array(labels)
    return labels



def P_YYYY(N_train, target_train, N_test, target_test):
    clf = RandomForestClassifier(n_estimators=300, random_state=520, max_depth=9,\
                                 min_samples_split=3, class_weight='balanced_subsample')
    clf = clf.fit(N_train, target_train)

    pred = clf.predict_proba(N_test)
    pred = DataFrame(pred)[0].values
    N_auc = metrics.roc_auc_score(target_test, 1 - pred)
    print N_auc
    print '\n'
    return N_auc, clf

def preds_calculate(Mat_Train,Mat_Labels):
    kf = KFold(len(Mat_Train), n_folds=10)
    NN_auc = []
    for train_index, test_index in kf:
        X_train, X_test = Mat_Train[train_index], Mat_Train[test_index]
        y_train, y_test = Mat_Labels[train_index], Mat_Labels[test_index]
        N_auc, clf = P_YYYY(X_train, y_train,  X_test, y_test)
        NN_auc.append(N_auc)
    mean_auc = mean(NN_auc)
    print 'AUC均值：',mean_auc
    return mean_auc, clf


os.chdir(r'E:\PycharmProjects\Rong360\dta')
#filename = r'Generate_dta\S_train_user_info.csv'
#train_xy = pd.read_csv(filename)
train_xy = pd.read_csv(r'Generate_dta\ttmp5_train.csv')
train_xy = train_xy.fillna(-1)

Mat_Train = train_xy.drop(['user_id','lable'],axis=1)
Mat_Train = array(Mat_Train)

Mat_Labels = DataFrame(train_xy.lable).values[0:,0:]
Mat_Label = transLabel(Mat_Labels)

mean_auc, clf = preds_calculate(Mat_Train,Mat_Label)

