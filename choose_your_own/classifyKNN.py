# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 13:30:51 2015

@author: Aaron Foss
"""

def classifyKNN(features_train, labels_train, features_test, labels_test, neigh):
    from sklearn.metrics import accuracy_score
    from time import time    
    from sklearn.neighbors import KNeighborsClassifier
    
    clf = KNeighborsClassifier(n_neighbors = neigh)
    
    train_time = time()
    clf.fit(features_train, labels_train)
    train_time = time() - train_time    
    
    pred_time = time()
    pred = clf.predict(features_test)
    pred_time = time() - pred_time    
    
    accuracy = accuracy_score(labels_test, pred)
    
    return clf, accuracy, train_time, pred_time

