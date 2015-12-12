#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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

### control size of training data
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100]

#########################################################
### your code goes here ###
def classifySVM(features_train, labels_train, features_test, labels_test, c_val):
    from sklearn.metrics import accuracy_score
    from time import time    
    from sklearn.svm import SVC
    
    clf = SVC(kernel="rbf", C = c_val)
    
    train_time = time()
    clf.fit(features_train, labels_train)
    train_time = time() - train_time    
    
    pred_time = time()
    pred = clf.predict(features_test)
    pred_time = time() - pred_time    
    
    accuracy = accuracy_score(labels_test, pred)
    
    return accuracy, train_time, pred_time, pred 

#########################################################

if __name__ == "__main__":
    
    c_vals = [10000]
    
    for c_val in c_vals: 
        accuracy, train_time, pred_time, pred = classifySVM(features_train, labels_train, features_test, labels_test, c_val)
        
        print "\nc_val:", round(c_val, 0)
        print "accuracy:", round(accuracy, 3)
        print "training time:", round(train_time, 3), "s"
        print "prediction time:", round(pred_time, 3), "s"
        print "preds:", pred[10], pred[26], pred[50]
