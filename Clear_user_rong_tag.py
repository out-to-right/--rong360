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
rong_tag = pd.read_csv('***\\rong_tag.txt')
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
    data['count_'+variable] = N_data.groupby('user_id')['user_id'].count()
    data['user_id'] = data.index
    data = DataFrame(data.values,columns=data.columns)
    data = data.reindex(columns=['user_id','min_'+variable,'max_'+variable,'sum_'+variable,\
    'mean_'+variable,'min_'+variable,'std_'+variable,'count_'+variable])
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


N_rong_tag = Summarizing_basic_information(rong_tag,'rong_tag')
#N_rong_tag = C_confidence_interval(N_rong_tag,'rong_tag')



#####  数据集的合并
##..........................训练集...........
N_rong_tag_train = merge(u_train, N_rong_tag, how="left", left_on='user_id', right_on='user_id')
N_rong_tag_test = merge(u_test, N_rong_tag, how="left", left_on='user_id', right_on='user_id')


#N_rong_tag_train.to_csv('***\\N_rong_tag_train.csv',index=False)
#N_rong_tag_test.to_csv('***\\N_rong_tag_test.csv',index=False)

end = time.clock()
print ('Running time: %s Seconds'%(end-start))


