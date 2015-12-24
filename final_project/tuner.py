# -*- coding: utf-8 -*-
from sklearn.feature_selection.univariate_selection import SelectPercentile, f_classif
from sklearn import svm, cross_validation, datasets, metrics
from sklearn.pipeline import Pipeline
from sklearn.grid_search import ParameterGrid, GridSearchCV, RandomizedSearchCV
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
   
def roc_score(labels, predictions):
    from sklearn.metrics import roc_auc_score    
    #predictions = get_score(clf, data)
    score = roc_auc_score(labels.ravel(), predictions.ravel())    
    return score


def tune(data,labels, clf=None):
    from sklearn.cross_validation import StratifiedShuffleSplit 
    sss = StratifiedShuffleSplit(labels, n_iter = 10, test_size = .1, random_state = 42) 
    clf = Pipeline([('num_features', 
               SelectKBest(f_classif,k=100)),
                    ('svm', svm.SVC(C=.01, kernel = 'linear', probability = True, random_state = 11))])
    param_grid = {
        'num_features__k':range(250,2500,250),
        'svm__C':10.**np.arange(-3,4),
        #'svm__loss':['hinge','squared_hinge'],
        'svm__class_weight':['balanced',None]
    }
    grid_search = RandomizedSearchCV(clf, 
                               param_grid,
                               n_iter = 100,
                               cv=sss,
                               scoring='f1',
                               n_jobs=-1,
                               pre_dispatch = '2*n_jobs',
                               random_state = 42)
    grid_search.fit(data,labels)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for p in param_grid.keys():
        print (p, best_parameters[p])

    #plot_cs(grid_search)

    return grid_search

def tuneSGD(data,labels, clf=None):
    from sklearn.cross_validation import StratifiedShuffleSplit 
    from sklearn.linear_model import SGDClassifier
    sss = StratifiedShuffleSplit(labels, n_iter = 10, test_size = .2, random_state = 42) 
    clf = Pipeline([#('num_features',SelectPercentile(f_classif,percentile = 5)),
                    ('sgd', SGDClassifier(random_state = 11, penalty = 'elasticnet', n_jobs = 1, alpha = 10**-4))])
    param_grid = {
        #'num_features__percentile': list(range(1,101)),
        'sgd__loss':['modified_huber','squared_hinge'],#,'hinge','log'],
        'sgd__class_weight':['balanced',None],
        'sgd__l1_ratio': list(np.arange(0,1.0,.01)),
        'sgd__alpha': list(10.**np.arange(-6,-3,.1))

    }
    
    grid_search = RandomizedSearchCV(clf, 
                               param_grid,
                               n_iter = 250,
                               random_state = 42,
                               cv=sss,
                               scoring = 'roc_auc',#roc_score,
                               n_jobs= -2,
                               pre_dispatch = '2*n_jobs')
    grid_search.fit(data,labels)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for p in param_grid.keys():
        print (p, best_parameters[p])
    
    return grid_search
    plot_cs(grid_search)

def cv_dataframe(grid_search):
    import pandas as pd    
    grid_dict = {}
    grid_dict['mean_cv_score'] = []
    for param in grid_search.grid_scores_[0][0].keys():
        grid_dict[param] = []
    for item in grid_search.grid_scores_:
        grid_dict['mean_cv_score'].append(item[1])
        for param in item[0].keys():
            grid_dict[param].append(item[0][param])
    
    df = pd.DataFrame(grid_dict)
    
    return df
#grid_search.grid_scores_


def plot_cs(grid_search):
    for name,params in grid_search.grid_scores_.items():
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


