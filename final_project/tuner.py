# -*- coding: utf-8 -*-
from sklearn.feature_selection.univariate_selection import SelectKBest, f_classif
from sklearn import svm, cross_validation, datasets, metrics
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
import numpy as np
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

def get_score(clf, data):
    '''Allows several of different kinds of classifiers,
    interchangably. Some (like random forests, SVMs, and logistic
    regression) have the method decision_function and some (like naive
    bayes) have predict_proba.
    '''
    try:
        out = clf.decision_function(data).ravel()
    except AttributeError:
        try:
            out = clf.predict_proba(data)[:,1]
        except AttributeError:
            out = clf.predict(data)
    return out
   
def roc_score(clf, data, labels):
    from sklearn.metrics import roc_auc_score    
    predictions = get_score(clf, data)
    return roc_auc_score(labels, predictions)


def tune(data,labels, clf=None):
    clf = Pipeline([('num_features', 
               SelectKBest(f_classif,k=100)),
                    ('svm', svm.LinearSVC(C=.01))])
    param_grid = [{
        'num_features__k':range(500,2000,500),
        'svm__C':10.**np.arange(-3,4)
    }]
    grid_search = GridSearchCV(clf, 
                               param_grid,
                               cv=3,
                               scoring="roc_auc",
                               n_jobs=4)
    grid_search.fit(data,labels)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for p in param_grid.keys():
        print p, best_parameters[p]

    #plot_cs(grid_search)

    return grid_search

def plot_cs(grid_search):
    for name,params in grid_search.param_grid.items():
        plt.plot(params,
                 [c.mean_validation_score 
                  for c in grid_search.grid_scores_], 
                 label="validation score")
        plt.xticks(params[::len(params)/6])
        plt.xlabel(name)
        plt.xlim(min(params),max(params))
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        title = ( get_name(grid_search.best_estimator_) +
                  name.split("__")[-1] )
        plt.title(title)
        plt.savefig("%s.png"%(title))

def evaluate(data,labels, num_trials=100):
    header = ("name","ROC_score","var","max")
    df = pandas.DataFrame()


