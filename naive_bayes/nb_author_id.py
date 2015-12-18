#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

# classify with Naive Bayes and return accuracy on test set
def classifyNB(features_train, labels_train, features_test, labels_test):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    from time import time    
    
    clf = GaussianNB()

    train_time = time()
    clf.fit(features_train, labels_train)
    train_time = time() - train_time    
    
    pred_time = time()
    pred = clf.predict(features_test)
    pred_time = time() - pred_time    
    
    accuracy = accuracy_score(labels_test, pred)
    
    return clf, accuracy, train_time, pred_time, pred


#########################################################

if __name__ == "__main__":
    
    accuracy, train_time, pred_time = classifyNB(features_train, labels_train, features_test, labels_test)
    
    print "accuracy:", round(accuracy, 3)
    print "training time:", round(train_time, 3), "s"
    print "prediction time:", round(pred_time, 3), "s"