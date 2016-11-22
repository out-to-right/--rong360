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
def Read_dta(f):
    N_dta = []
    N_dta_columns = f.readline().strip()
    N_dta_columns = N_dta_columns.split('\t')
    while True:
        line = f.readline().strip()
        if line:
            line = line.split('\t')
            N_dta.append(line)
        else:
            break
    N_dta = DataFrame(N_dta,columns=N_dta_columns)
    return N_dta

filename = open('***\\consumption_recode.txt','r')
consumption_recode = Read_dta(filename)

u_train = pd.read_csv('***\\train.txt')
u_test = pd.read_csv('***\\test.txt')


def Summarizing_basic_information(N_data,variable):
    
    N_data[variable] = N_data[variable].astype(float)
    data = DataFrame()
    data['min_'+variable] = N_data.groupby('user_id')[variable].min()
    data['max_'+variable] = N_data.groupby('user_id')[variable].max()
    data['sum_'+variable] = N_data.groupby('user_id')[variable].sum()
    data['mean_'+variable] = N_data.groupby('user_id')[variable].mean()
    data['std_'+variable] = N_data.groupby('user_id')[variable].std()
    data['count_'+variable] = N_data.groupby('user_id')[variable].count()
    data['user_id'] = data.index
    data = DataFrame(data.values,columns=data.columns)
    data = data.reindex(columns=['user_id','min_'+variable,'max_'+variable,'sum_'+variable,\
    'mean_'+variable,'std_'+variable,'count_'+variable])

    return data

## 将数据集进行区间估计
def C_confidence_interval(dta,variable):
    
    dta[variable+u'_upper'] = 0
    dta[variable+u'_lower'] = 0
    t_table = pd.read_excel(u'dta\\Original_dta\\user_dta\\t检验临界值表.xls','t_table')
    z_a = 1.96
    for i in range(len(dta)):
        if dta.ix[i][u'count_'+variable] < 30:
            t_a = t_table.ix[dta.ix[i][u'count_'+variable]-1]['a3']
            dta.loc[i,variable+u'_upper'] = dta.ix[i]['mean_'+variable] + \
                                            t_a*dta.ix[i]['std_'+variable]/math.sqrt(dta.ix[i][u'count_'+variable])
            dta.loc[i,variable+u'_lower'] = dta.ix[i]['mean_'+variable] - \
                                            t_a*dta.ix[i]['std_'+variable]/math.sqrt(dta.ix[i][u'count_'+variable])
        if dta.ix[i][u'count_'+variable] >= 30:
            dta.loc[i,variable+u'_upper'] = dta.ix[i]['mean_'+variable] + \
                                            z_a*dta.ix[i]['std_'+variable]/math.sqrt(dta.ix[i][u'count_'+variable])
            dta.loc[i,variable+u'_lower'] = dta.ix[i]['mean_'+variable] - \
                                            z_a*dta.ix[i]['std_'+variable]/math.sqrt(dta.ix[i][u'count_'+variable])
    return dta


## 数据探索
Freq_var = []
for i in range(2,len(consumption_recode.columns)):
    a = []
    a.append(consumption_recode.columns[i]);
    a.append(len(consumption_recode[consumption_recode.columns[i]].value_counts()))
    Freq_var.append(a)


list_1 = [];list_2 = []
for i in range(len(Freq_var)):
    if Freq_var[i][1] <= 3:
        list_1.append(Freq_var[i][0])
    if Freq_var[i][1] > 3:
        list_2.append(Freq_var[i][0])

# 单独处理 is_cheat_bill


for i in range(len(list_2)):
    N_cc = Summarizing_basic_information(consumption_recode,list_2[i])
    u_train = merge(u_train, N_cc, how="left", left_on='user_id', right_on='user_id')
    u_test = merge(u_test, N_cc, how="left", left_on='user_id', right_on='user_id')



## 2、处理 【bill_id:记录编号】
bill_id = DataFrame()
bill_id['count_bill_id'] = consumption_recode.groupby('user_id')['bill_id'].count()
bill_id['user_id'] = bill_id.index
N_bill_id = DataFrame(bill_id.values,columns = bill_id.columns)


## 3、处理 【is_cheat_bill:是否恶意账单】
N1_is_cheat_bill = DataFrame(consumption_recode['user_id']).join(consumption_recode['is_cheat_bill'])
N1_is_cheat_bill = N1_is_cheat_bill.drop_duplicates()
N_is_cheat_bill = DataFrame()
N_is_cheat_bill['is_cheat_bill'] = N1_is_cheat_bill.groupby('user_id')['is_cheat_bill'].max()
N_is_cheat_bill['user_id'] = N_is_cheat_bill.index
N_is_cheat_bill = DataFrame(N_is_cheat_bill.values,columns=N_is_cheat_bill.columns)
N_is_cheat_bill = N_is_cheat_bill.reindex(columns=['user_id','is_cheat_bill'])


## 4、处理 【card_type:  卡类型】
N_card_type = DataFrame()
N_card_type['max_card_type'] = consumption_recode.groupby('user_id')['card_type'].max()
N_card_type['min_card_type'] = consumption_recode.groupby('user_id')['card_type'].min()
N_card_type['count_card_type'] = consumption_recode.groupby('user_id')['user_id'].count()
N_card_type['user_id'] = N_card_type.index
N_card_type = DataFrame(N_card_type.values,columns=N_card_type.columns)



## 5、处理 【curr: 币种】
consumption_recode['curr'] = consumption_recode['curr'].astype(float)
N_curr = DataFrame()
N_curr['max_curr'] = consumption_recode.groupby('user_id')['curr'].max()
N_curr['min_curr'] = consumption_recode.groupby('user_id')['curr'].min()
N_curr['median_curr'] = consumption_recode.groupby('user_id')['curr'].median()
N_curr['count_curr'] = consumption_recode.groupby('user_id')['user_id'].count()
N_curr['user_id'] = N_curr.index
N_curr = DataFrame(N_curr.values,columns=N_curr.columns)


## 6、处理 【repay_stat: 还款状态】
N_repay_stat = DataFrame()
N_repay_stat['max_repay_stat'] = consumption_recode.groupby('user_id')['repay_stat'].max()
N_repay_stat['min_repay_stat'] = consumption_recode.groupby('user_id')['repay_stat'].min()
N_repay_stat['count_repay_stat'] = consumption_recode.groupby('user_id')['user_id'].count()
N_repay_stat['user_id'] = N_repay_stat.index
N_repay_stat = DataFrame(N_repay_stat.values,columns=N_repay_stat.columns)


u_train = merge(u_train, N_bill_id, how="left", left_on='user_id', right_on='user_id')
u_train = merge(u_train, N_is_cheat_bill, how="left", left_on='user_id', right_on='user_id')
u_train = merge(u_train, N_card_type, how="left", left_on='user_id', right_on='user_id')
u_train = merge(u_train, N_curr ,how="left", left_on='user_id', right_on='user_id')
u_train = merge(u_train, N_repay_stat, how="left", left_on='user_id', right_on='user_id')


u_test = merge(u_test, N_bill_id, how="left", left_on='user_id', right_on='user_id')
u_test = merge(u_test, N_is_cheat_bill, how="left", left_on='user_id', right_on='user_id')
u_test = merge(u_test, N_card_type, how="left", left_on='user_id', right_on='user_id')
u_test = merge(u_test, N_curr ,how="left", left_on='user_id', right_on='user_id')
u_test = merge(u_test, N_repay_stat, how="left", left_on='user_id', right_on='user_id')



u_train.to_csv('***\\N_train_consumption1.csv',index=False)
u_test.to_csv('***\\N_test_consumption1.csv',index=False)


end = time.clock()
print ('Running time: %s Seconds'%(end-start))

