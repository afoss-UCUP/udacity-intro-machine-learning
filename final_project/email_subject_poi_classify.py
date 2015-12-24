# -*- coding: utf-8 -*-
#if __name__ == "__main__":
#    import multiprocessing as mp; mp.set_start_method('forkserver')
import pickle
import numpy
import sys
sys.path.append( "../naive_bayes/" )
from mongo_related import *
import pandas as pd

numpy.random.seed(11)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
#words_file = "subject_word_data.pkl" 
#authors_file = "subject_email_authors.pkl"
#word_data = pickle.load( open(words_file, "r"))
#authors = pickle.load( open(authors_file, "r") )
db = get_db()
data = pd.DataFrame(list(db.enron_emails.find()))
data = data.drop_duplicates(subset=['poi_flag','subject','body'])

poi = data['poi_flag'].values
word_data = data['body'].values
### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime

### your code goes here

#def code(features, labels):
#    from tuner import tune
#    out = tune(features,labels)
#    return out

def tunedSGD(features, labels):
    from tuner import tuneSGD
    out = tuneSGD(features,labels)
    return out


if __name__ == '__main__':
    from sklearn import cross_validation
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, poi,stratify = poi, test_size=0.2, random_state=11)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, 
                                 max_df=0.5,
                                 max_features=100000,
                                 min_df = 5,
                                 ngram_range = (1,2),
                                 stop_words='english')
    features_train = vectorizer.fit_transform(features_train)
    features_test  = vectorizer.transform(features_test)
   # features_train = pd.SparseDataFrame([ pd.SparseSeries(features_train[i].toarray().ravel())
    #     for i in numpy.arange(features_train.shape[0]) ])

    out = tunedSGD(features_train, labels_train)


from tuner import cv_dataframe
cv_df = cv_dataframe(out)
cv_df['sgd__class_weight'][cv_df['sgd__class_weight'].isnull()] = 'None'
cv_df['sgd__l1_class'] = 'NaN'
cv_df['sgd__l1_class'][cv_df['sgd__l1_ratio'] <= .5] = 'L2ish'
cv_df['sgd__l1_class'][cv_df['sgd__l1_ratio'] > .5] = 'L1ish'

cv_df.boxplot('mean_cv_score',by = ['sgd__loss','sgd__alpha','sgd__class_weight'], rot = 90)

from ggplot import *

qplot(y = 'mean_cv_score',x = 'num_features__k', data = cv_df, color = 'sgd__loss')
qplot(y = 'mean_cv_score',x = 'sgd__alpha', data = cv_df, color = 'sgd__l1_class') +\
    facet_wrap('sgd__loss','sgd__class_weight', scales = 'fixed') +\
    stat_smooth() +\
    ylim(.8,1) +\
    xlim(0,.001)

qplot(y = 'mean_cv_score',x = 'num_features__k', data = cv_df, color = 'sgd__loss')


fig1 = cv_df[['mean_cv_score','num_features__k','sgd__loss']]
fig1.plot(colormap = 'Greens')
# from dt_author_id import classifyDT

# clf, accuracy, train_time, pred_time, pred = classifyDT(features_train, labels_train, features_test, labels_test,10)

from mnnb_poi_id import classifyMNNB

clf, accuracy, train_time, pred_time, pred, pred_prob = classifyMNNB(features_train, labels_train, features_test, labels_test)

from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectKBest, f_classif

clf = LinearSVC(C = 10, loss = 'hinge')
fclassif = SelectKBest(f_classif, k=1250)
features_train = fclassif.fit_transform(features_train, labels_train)
clf.fit(features_train, labels_train)
features_test = fclassif.transform(features_test)
pred = clf.predict(features_test)


cv = cross_validation.StratifiedShuffleSplit(labels_train, n_iter=3, test_size=0.1, random_state=0)

from tuner import roc_score
vals = cross_validation.cross_val_score(clf, features_train, labels_train, cv=cv, scoring = 'roc_auc')

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

    
print ('precision: ',precision_score(labels_test, pred))
print ('recall: ',recall_score(labels_test, pred))
print ('AUC: ',roc_auc_score(labels_test, pred))
print ('F1: ',f1_score(labels_test, pred))


features_total = word_data
labels_total = poi

features_total = vectorizer.transform(features_total)
features_total = fclassif.transform(features_total) 

from sklearn.cross_validation import StratifiedShuffleSplit 
sss = StratifiedShuffleSplit(labels_total, n_iter = 3, test_size = .5, random_state = 42) 
from sklearn.cross_validation import cross_val_predict
pred = cross_val_predict(clf, features_total, y=labels_total, cv = 3, n_jobs = 6)
pred = clf.predict(features_total)
data['predicted_poi'] = pred

data.groupby('address').mean().sort(['poi_flag','predicted_poi'])


for i in range(0,len(pred)):
    if pred[i] == 1:
        print word_data[i]
       
for i in range(0,len(clf.feature_importances_)):
    if clf.feature_importances_[i] >= .005:
        print clf.feature_importances_[i], vectorizer.get_feature_names()[i]
        

from sklearn.linear_model import SGDClassifier

from sklearn.feature_selection import SelectPercentile, f_classif

clf = SGDClassifier(alpha = .0000015, loss = 'squared_hinge', class_weight = 'balanced', l1_ratio = .31, penalty = 'elasticnet', random_state = 42 )
#fclassif = SelectKBest(f_classif, k=21500)
#features_train = fclassif.fit_transform(features_train, labels_train)
clf.fit(features_train, labels_train)
#features_test = fclassif.transform(features_test)
pred = clf.predict(features_test)

   
print ('precision: ',precision_score(labels_test, pred))
print ('recall: ',recall_score(labels_test, pred))
print ('AUC: ',roc_auc_score(labels_test, pred))
print ('F1: ',f1_score(labels_test, pred))


clf = SGDClassifier(alpha = .6*10**-5, loss = 'modified_huber', 
                    class_weight = 'balanced', l1_ratio = .03, penalty = 'elasticnet', random_state = 42)
#fclassif = SelectPercentile(f_classif, percentile = 80)
#features_train = fclassif.fit_transform(features_train, labels_train)
clf.fit(features_train, labels_train)
#features_test = fclassif.transform(features_test)
pred = clf.predict(features_test)
  
print ('precision: ',precision_score(labels_test, pred))
print ('recall: ',recall_score(labels_test, pred))
print ('AUC: ',roc_auc_score(labels_test, pred))
print ('F1: ',f1_score(labels_test, pred))
