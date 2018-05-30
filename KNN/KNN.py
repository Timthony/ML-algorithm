#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 22:24:48 2018

@author: arcstone_mems_108  tianhangz
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import sys
from KNNClassifier import KNNClassifier


iris = datasets.load_iris()


X = iris.data
y = iris.target

# 将测试数据和训练数据分离、
def train_test_split(X, y, test_ratio = 0.2, seed = None):
    # 随机数设置
    if seed:
        np.random.seed(seed)
    shuffle_indexes = np.random.permutation(len(X))
    
    test_size = int(len(X) * test_ratio)
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[test_size:]
    
    X_train = X[train_indexes]
    y_train = y[train_indexes]
    
    X_test = X[test_indexes]
    y_test = y[test_indexes]
    return X_train, y_train, X_test, y_test

# 进行KNN分类预测
    
X_train, y_train, X_test, y_test = train_test_split(X, y)
 

my_knn_clf = KNNClassifier(k=3)
my_knn_clf.fit(X_train, y_train)
y_predict = my_knn_clf.predict(X_test)

sum(y_predict == y_test)/len(y_test)













