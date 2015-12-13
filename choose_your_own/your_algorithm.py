#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from classifyKNN import classifyKNN

neighs = [2,3,5,8,12]

for neigh in neighs:
    clf, accuracy, train_time, pred_time = classifyKNN(features_train, labels_train, features_test, labels_test, neigh)
    
    print "\nKNN Classifier"
    print "\nneighbors:", neigh
    print "accuracy:", round(accuracy, 3)
    print "training time:", round(train_time, 3), "s"
    print "prediction time:", round(pred_time, 3), "s"
    try:
        prettyPicture(clf, features_test, labels_test)
    except NameError:
        pass


from classifyAdaBoost import classifyAdaBoost

n_ests = [5,25,125,625,3125]

for n_est in n_ests:
    clf, accuracy, train_time, pred_time = classifyAdaBoost(features_train, labels_train, features_test, labels_test, n_est)
    
    print "\nAdaBoost Classifier"
    print "estimators:", n_est
    print "accuracy:", round(accuracy, 3)
    print "training time:", round(train_time, 3), "s"
    print "prediction time:", round(pred_time, 3), "s"
    try:
        prettyPicture(clf, features_test, labels_test)
    except NameError:
        pass
    

from classifyRF import classifyRF

n_ests = [5,25,125,625,3125]

for n_est in n_ests:
    clf, accuracy, train_time, pred_time = classifyRF(features_train, labels_train, features_test, labels_test, n_est)
    
    print "\nRandomForest Classifier"
    print "estimators:", n_est
    print "accuracy:", round(accuracy, 3)
    print "training time:", round(train_time, 3), "s"
    print "prediction time:", round(pred_time, 3), "s"
    try:
        prettyPicture(clf, features_test, labels_test)
    except NameError:
        pass