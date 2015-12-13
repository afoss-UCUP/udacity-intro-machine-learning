# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 13:43:56 2015

@author: Aaron Foss
"""

def classifyAdaBoost(features_train, labels_train, features_test, labels_test, n_est):
    from sklearn.metrics import accuracy_score
    from time import time    
    from sklearn.ensemble import AdaBoostClassifier
    
    clf = AdaBoostClassifier(n_estimators = n_est)
    
    train_time = time()
    clf.fit(features_train, labels_train)
    train_time = time() - train_time    
    
    pred_time = time()
    pred = clf.predict(features_test)
    pred_time = time() - pred_time    
    
    accuracy = accuracy_score(labels_test, pred)
    
    return clf, accuracy, train_time, pred_time
