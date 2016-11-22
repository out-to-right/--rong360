#!/usr/bin/python
# coding=utf-8

import os
from numpy import *
import pandas as pd
from pandas import *
from pandas import DataFrame
import xgboost as xgb
from sklearn.cross_validation import KFold
import time


def parameter_optimize(dtrain ,dval ,dtest ,target_test):

    param={
        'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        
        'gamma':12,
        'eta': 0.104,
        'lambda':134,
        'subsample':0.83,
        'colsample_bytree':0.45,
        'max_depth':7,
        'scale_pos_weight': 1.1,
        
        'missing':-1,
        'seed':520,
        'nthread':4
        }
    watchlist  = [(dtrain,'train'),(dval,'val')]
    bst = xgb.train(param, dtrain, num_boost_round=81 ,evals=watchlist,\
                    verbose_eval=1)
    
    pred = bst.predict(dtest)
    auc_value = metrics.roc_auc_score(target_test, pred)
    return auc_value,bst

def auc_calculate(Mat_Train,Mat_Labels):
    kf = KFold(len(Mat_Train), n_folds=10)
    gather_auc = []
    i = 1
    for train_index, test_index in kf:
        print (i)
        i = i+1
        X_train, X_test = Mat_Train[train_index], Mat_Train[test_index]
        y_train, y_test = Mat_Labels[train_index], Mat_Labels[test_index]
        dtrain = xgb.DMatrix(X_train, y_train)
        dval = xgb.DMatrix(X_test,y_test)
        dtest = xgb.DMatrix(X_test)
        auc_value,bst = parameter_optimize(dtrain, dval ,dtest ,y_test)
        gather_auc.append(auc_value)
        print ('\n')
    mean_auc = mean(gather_auc)
    return mean_auc,bst


os.chdir(r'E:\PycharmProjects\Rong360\dta')

# 训练集
S_train_user_info = pd.read_csv(r'Generate_dta\S_train_user_info.csv')
N_train_user_info = pd.read_csv(r'Generate_dta\N_train_user_info.csv').drop(['lable'],axis=1)
relation1_train = pd.read_csv(r'Generate_dta\0909relation1_train.csv')
relation2_train = pd.read_csv(r'Generate_dta\0909relation2_train.csv')
N_train_consumption1 = pd.read_csv(r'Generate_dta\N_train_consumption1.csv').drop(['lable'],axis=1)
t_consumption = pd.read_csv(r'Generate_dta\t_consumption.csv')
N_rong_tag_train = pd.read_csv(r'Generate_dta\N_rong_tag_train.csv').drop(['lable'],axis=1)

train = merge(S_train_user_info,N_train_user_info,how="left", left_on='user_id', right_on='user_id')
train = merge(train,relation1_train,how="left", left_on='user_id', right_on='user_id')
train = merge(train,relation2_train,how="left", left_on='user_id', right_on='user_id')
train = merge(train,N_train_consumption1,how="left", left_on='user_id', right_on='user_id')
train = merge(train,t_consumption,how="left", left_on='user_id', right_on='user_id')
#train = merge(train,N_rong_tag_train,how="left", left_on='user_id', right_on='user_id')

train = train.fillna(-1)
Mat_Train = train.drop(['user_id','lable'],axis=1)
Mat_Train = array(Mat_Train)
Mat_Labels = train['lable'].astype(int)

mean_auc,bst = auc_calculate(Mat_Train,Mat_Labels)


