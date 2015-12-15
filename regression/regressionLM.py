# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 23:38:35 2015

@author: Aaron Foss
"""

def regressionLM(feature_train, target_train, feature_test):
    from time import time    
    from sklearn.linear_model import LinearRegression
    
    reg = LinearRegression()
    
    train_time = time()
    reg.fit(feature_train, target_train)
    train_time = time() - train_time    
    
    pred_time = time()
    pred = reg.predict(feature_test)
    pred_time = time() - pred_time    
    
    return reg, train_time, pred_time
