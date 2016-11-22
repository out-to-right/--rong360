# coding=utf-8

'''
author: ShiLei Miao
'''

from numpy import *
import pandas as pd
from pandas import *
import os
import time

os.chdir(r'E:\PycharmProjects\Rong360')
start =time.clock()


###
# 数据读取
user_info = pd.read_csv('***\\user_info.txt')
u_train = pd.read_csv('***\\train.txt')
u_test = pd.read_csv('***\\test.txt')


##### Step1:
### user_info表处理
# 大量数据重复，目前考虑将变量‘tm_encode’踢出来，将user_info表拆分

# user_info1 96153记录
user_info1 = user_info.drop(['tm_encode'],axis=1)
user_info1 = user_info1.drop_duplicates()

# user_info2 38261记录【train:26000, test:12261】
user_info['tm_encode'] = user_info['tm_encode'].astype(float)
N_user_info2 = DataFrame()
N_user_info2['min_tm_encode'] = user_info.groupby('user_id')['tm_encode'].min()
N_user_info2['max_tm_encode'] = user_info.groupby('user_id')['tm_encode'].max()
N_user_info2['sum_tm_encode'] = user_info.groupby('user_id')['tm_encode'].sum()
N_user_info2['mean_tm_encode'] = user_info.groupby('user_id')['tm_encode'].mean()
N_user_info2['std_tm_encode'] = user_info.groupby('user_id')['tm_encode'].std()
N_user_info2['count_tm_encode'] = user_info.groupby('user_id')['tm_encode'].count()
N_user_info2['count_user_id'] = user_info.groupby('user_id')['user_id'].count()

##### Step2:
#...................................................user_info2已处理完毕
### user_info1表处理
# 表中记录按照是否为38261行将user_info1表拆分【结果：Uni_dta为10】
Uni_dta = []
U_Uni_dta = []
U_Uni_dta.append('user_id')
L_dta = DataFrame()
for i in user_info1.columns:
    L_dta['user_id'] = user_info1['user_id']
    L_dta[i] = user_info1[i]
    L_dta = L_dta.drop_duplicates()
    if L_dta.shape[0] == 38261:
        Uni_dta.append(i)
    if L_dta.shape[0] != 38261:
        U_Uni_dta.append(i)
        #print i
        #print L_dta.shape[0]
    #print '\n'
    L_dta = DataFrame()

# 记录为38261行
user_info1_1 = DataFrame(user_info1,columns=Uni_dta)
N_user_info1_1 = user_info1_1.drop_duplicates()
N_user_info1_1 = DataFrame(N_user_info1_1.values,columns=N_user_info1_1.columns)
# N_user_info1_1的缺失处理方式【mode】
def MissData_deal_N_user_info1_1(data):
    for i in range(data.shape[1]):
        if data.ix[:, i].isnull().sum() >= 1:
            # 保留数据的缺失信息
            data[data.columns[i] + '_missingInfo'] = data.ix[:,i].isnull().astype(int)
            # 类别变量用众数填充
            data[data.columns[i]] = data[data.columns[i]].fillna(data.ix[:, i].mode()[0])
    return data
N_user_info1_1 = MissData_deal_N_user_info1_1(N_user_info1_1)

#              记录非38261行
user_info1_2 = DataFrame(user_info1,columns=U_Uni_dta)
user_info1_2 = user_info1_2.drop_duplicates()


##### Step3:
#...................................................user_info1_1已处理完毕
### user_info1_2表处理
# 按表中变量的性质拆分为：user_info1表拆分user_info1_2_1【类别型】与user_info1_2_2【数值型】
N_var1 = ['user_id','local_hk','money_function','sex','occupation','education','marital_status','live_info']
N_var2 = ['user_id','age','expect_quota','max_month_repay','salary']

def MissData_deal_user_info1_2_1(data):
    for i in range(data.shape[1]):
        if data.ix[:, i].isnull().sum() >= 1:
            # 保留数据的缺失信息
            data[data.columns[i] + '_missingInfo'] = data.ix[:,i].isnull().astype(int)
            # 类别变量用-1填充
            data[data.columns[i]] = data[data.columns[i]].fillna(-1)
    return data

user_info1_2_1 = DataFrame(user_info1_2,columns=N_var1)
user_info1_2_1 = MissData_deal_user_info1_2_1(user_info1_2_1)

def Dummy(data, variable):
    One_hot_dta = get_dummies(data, prefix=variable)
    return One_hot_dta

def Dummy_Master(data):
    names = data.columns
    for i in names:
        if i != "user_id" and len(data[i].value_counts()) > 2:
            One_hot_dtas = Dummy(data[i], i)
            data = data.join(One_hot_dtas)
            data = data.drop(i, axis=1)
    return data

# 将类别变量进行one_hot编码
user_info1_2_1 = Dummy_Master(user_info1_2_1)
# 将编码后的变量按照频数统计
N_user_info1_2_1 = DataFrame()
for i in range(1,user_info1_2_1.shape[1]):
    Var_N = user_info1_2_1.columns[i]
    N_user_info1_2_1[Var_N] = user_info1_2_1.groupby('user_id')[Var_N].count()



##### Step4:
#...................................................user_info1_2_1已处理完毕
### user_info1_2_2表处理
# 按表中变量统计相关的指标
user_info1_2_2 = DataFrame(user_info1_2,columns=N_var2)


N_user_info1_2_2_1 = DataFrame()
N_111 = DataFrame(user_info1_2_2[user_info1_2_2['age']!='NONE'],\
          columns=['user_id','age']).drop_duplicates()
N_111['age'] = N_111['age'].astype(float)
N_user_info1_2_2_1['age'] = N_111.groupby('user_id')['age'].mean()

user_info1_2_2_2 = user_info1_2_2.drop(['age'],axis=1)

def MissData_deal_Num(data):
    for i in range(data.shape[1]):
        if data.ix[:, i].isnull().sum() >= 1:
            # 保留数据的缺失信息
            data[data.columns[i] + '_missingInfo'] = data.ix[:,i].isnull().astype(int)
            # 类别变量用均值填充
            #data[data.columns[i]] = data[data.columns[i]].fillna(data.ix[:, i].mean())
    return data

user_info1_2_2_2 = MissData_deal_Num(user_info1_2_2_2)
N_user_info1_2_2_2 = DataFrame()
for i in range(1,user_info1_2_2_2.shape[1]):
    if i < 4:
        Var_N = user_info1_2_2_2.columns[i]
        N_user_info1_2_2_2[Var_N+'_max'] = user_info1_2_2_2.groupby('user_id')[Var_N].max()
        N_user_info1_2_2_2[Var_N+'_min'] = user_info1_2_2_2.groupby('user_id')[Var_N].min()
        N_user_info1_2_2_2[Var_N+'_mean'] = user_info1_2_2_2.groupby('user_id')[Var_N].mean()
        N_user_info1_2_2_2[Var_N+'_std'] = user_info1_2_2_2.groupby('user_id')[Var_N].std()
        N_user_info1_2_2_2[Var_N+'_count'] = user_info1_2_2_2.groupby('user_id')[Var_N].count()
    if i > 3:
        Var_N = user_info1_2_2_2.columns[i]
        N_user_info1_2_2_2[Var_N] = user_info1_2_2_2.groupby('user_id')[Var_N].count()
N_user_info1_2_2_2 = N_user_info1_2_2_2.fillna(-1)


#####  数据集的合并
##..........................训练集...........
N_user_info2['user_id'] = N_user_info2.index
N_user_info1_2_1['user_id'] = N_user_info1_2_1.index
N_user_info1_2_2_1['user_id'] = N_user_info1_2_2_1.index
N_user_info1_2_2_2['user_id'] = N_user_info1_2_2_2.index

Master_one = merge(u_train, N_user_info2, how="left", left_on='user_id', right_on='user_id')
Master_two = merge(Master_one, N_user_info1_1, how="left", left_on='user_id', right_on='user_id')
Master_three = merge(Master_two, N_user_info1_2_1, how="left", left_on='user_id', right_on='user_id')
Master_four = merge(Master_three, N_user_info1_2_2_1, how="left", left_on='user_id', right_on='user_id')
N_train_user_info = merge(Master_four, N_user_info1_2_2_2, how="left", left_on='user_id', right_on='user_id')

Master_one = merge(u_test, N_user_info2, how="left", left_on='user_id', right_on='user_id')
Master_two = merge(Master_one, N_user_info1_1, how="left", left_on='user_id', right_on='user_id')
Master_three = merge(Master_two, N_user_info1_2_1, how="left", left_on='user_id', right_on='user_id')
Master_four = merge(Master_three, N_user_info1_2_2_1, how="left", left_on='user_id', right_on='user_id')
N_test_user_info = merge(Master_four, N_user_info1_2_2_2, how="left", left_on='user_id', right_on='user_id')


N_train_user_info.to_csv('***\\N_train_user_info.csv',index=False)
N_test_user_info.to_csv('***\\N_test_user_info.csv',index=False)

end = time.clock()
print ('Running time: %s Seconds'%(end-start))




