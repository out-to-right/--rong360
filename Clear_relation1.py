import os
import csv
import math
import time
import igraph
import numpy as np
import pandas as pd

import os
import math
import numpy as np
import pandas as pd
import time

os.chdir('E:\PycharmProjects\Rong360\dta\Original_dta')


relation1 = pd.read_csv('relation1.txt')
relation2 = pd.read_csv('relation2.txt')

train = pd.read_csv('train.txt')
test = pd.read_csv('test\\test.txt')


def genRealtionAvl(relation, train, test):
    relation_user = set(relation['user1_id'].unique()) | set(relation['user2_id'].unique())
    
    re1_train = set()
    for index, row in train.iterrows():
        if row['user_id'] not in relation_user:
            re1_train.add(index)
    
    re1_test = set()
    for index, row in test.iterrows():
        if row['user_id'] not in relation_user:
            re1_test.add(index)
    
    return train.drop(re1_train), test.drop(re1_test) 

re1_train, re1_test = genRealtionAvl(relation1, train, test)
re2_train, re2_test = genRealtionAvl(relation2, train, test)


re1_map = igraph.Graph.TupleList(csv.reader(open("relation1.txt")))

# set lable
for index, row in re1_train.iterrows():
    re1_map.vs[re1_map.vs.find(row['user_id']).index]['lable'] = row['lable']

train_vertex = [re1_map.vs.find(x).index for x in re1_train['user_id'].tolist()]


def searchRelation1(user_id, depth):
    aa = user_id
    if depth == 1:
        return [x for x in re1_map.neighbors(user_id)]
    else:
        neighbors = []
        for neighbor in searchRelation1(user_id, 1):
            a = searchRelation1(neighbor, depth - 1)
            b = list(a)
            neighbors.extend(list(searchRelation1(neighbor, depth - 1)))
        return set(neighbors)

def calRelation1(cal_set):
    res = []
    tlen = len(cal_set)
    for index, row in cal_set.iterrows():
        friends = searchRelation1(re1_map.vs.find(row['user_id']).index, 1)
        count, scount = 0.0 ,0.0
        for friend in friends:
            if friend in train_vertex:
                scount += 1
                if re1_map.vs[friend]['lable'] != None:
                    count += re1_map.vs[friend]['lable']
        
        friends2 = searchRelation1(re1_map.vs.find(row['user_id']).index, 2)
        count2, scount2 = 0.0 ,0.0
        for friend in friends2:
            if friend in train_vertex:
                scount2 += 1
                if re1_map.vs[friend]['lable'] != None:
                    count2 += re1_map.vs[friend]['lable']
        
        t = [count, scount, len(friends), count2, scount2, len(friends2)]
        t = [str(x) for x in t]
        res.append(row['user_id'] + ',' + ','.join(t))
        # if len(res) % 100 == 0:
        print len(res), tlen
        print res[-1]
    return res

relation1_train = calRelation1(re1_train)
#relation1_test = calRelation1(re1_test)
