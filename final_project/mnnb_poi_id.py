# -*- coding: utf-8 -*-

def classifyMNNB(features_train, labels_train, features_test, labels_test):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score
    from time import time    
    
    clf = MultinomialNB()

    train_time = time()
    clf.fit(features_train, labels_train)
    train_time = time() - train_time    
    
    pred_time = time()
    pred = clf.predict(features_test)
    pred_prob = clf.predict_proba(features_test)
    pred_time = time() - pred_time    
    
    accuracy = accuracy_score(labels_test, pred)
    
    return clf, accuracy, train_time, pred_time, pred, pred_prob
