#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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
def classifyDT(features_train, labels_train, features_test, labels_test, min_smp_split):
    from sklearn.metrics import accuracy_score
    from time import time    
    from sklearn.tree import DecisionTreeClassifier
    
    clf = DecisionTreeClassifier(min_samples_split = min_smp_split)
    
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
    
    min_smp_splits = [40]
    
    for min_smp_split in min_smp_splits: 
        accuracy, train_time, pred_time, pred = classifyDT(features_train, labels_train, features_test, labels_test, min_smp_split)
        
        print "\nmin_smp_split:", round(min_smp_split, 0)
        print "accuracy:", round(accuracy, 3)
        print "training time:", round(train_time, 3), "s"
        print "prediction time:", round(pred_time, 3), "s"
        print "preds:", pred[10], pred[26], pred[50]
