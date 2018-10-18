# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:11:25 2018

@author: ych

E-mail:yao544303963@gmail.com
"""

from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

np.set_printoptions(suppress=True)
mem = Memory("./mycache")



# 输入数据格式为
# User movie1:ratting1 movie2:ratting2
@mem.cache
def get_data(filename):
    data = load_svmlight_file(filename)
    return data[0], data[1]


# 计算jaccard 相似度
def get_jaccard_similarity(X):
    n = X.T.shape[1]
    similarity = np.zeros([n, n])
    for i in range(n):
        v1 = X.T[i].toarray()
        for j in range(i + 1, n):
            v2 = X.T[j].toarray()
            sim = jaccard_similarity_score(v1, v2)
            similarity[i][j] = sim
            similarity[j][i] = sim
    return similarity


# 计算余弦相似度
def get_consine_similarity(X):
    similarity = cosine_similarity(X)
    return similarity


filename = "../../data/ratingslibsvm"
X, y = get_data(filename)
consine_sim = get_consine_similarity(X)
print(consine_sim)
jaccard_sim = get_jaccard_similarity(X)
print(jaccard_sim)







