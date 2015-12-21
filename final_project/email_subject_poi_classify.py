# -*- coding: utf-8 -*-

import pickle
import numpy
import sys
sys.path.append( "../naive_bayes/" )
from mongo_related import *
import pandas as pd

numpy.random.seed(42)


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

poi = data['poi_flag'].as_matrix()
word_data = data['subject'].as_matrix()
### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime


### your code goes here

def code(features, labels):
    from tuner import tune
    out = tune(features,labels)
    return out

if __name__ == '__main__':
    from sklearn import cross_validation
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, poi,stratify = poi, test_size=0.1, random_state=42)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, 
                                 max_df=0.5,
                                 #max_features=10000,
                                 min_df = 2,
                                 ngram_range = (1,2),
                                 stop_words='english')
    features_train = vectorizer.fit_transform(features_train)
    features_test  = vectorizer.transform(features_test)
    

    out = code(features_train, labels_train)

# from dt_author_id import classifyDT

# clf, accuracy, train_time, pred_time, pred = classifyDT(features_train, labels_train, features_test, labels_test,10)

from mnnb_poi_id import classifyMNNB

clf, accuracy, train_time, pred_time, pred, pred_prob = classifyMNNB(features_train, labels_train, features_test, labels_test)

from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectKBest, f_classif

clf = LinearSVC(C = 1)
fclassif = SelectKBest(f_classif, k=1500)
features_train = fclassif.fit_transform(features_train, labels_train)
clf.fit(features_train, labels_train)
features_test = fclassif.transform(features_test)
pred = clf.predict(features_test)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
    
print 'precision: ',precision_score(labels_test, pred)
print 'recall: ',recall_score(labels_test, pred)

features_total = word_data
labels_total = poi

features_total = vectorizer.transform(features_total)
features_total = fclassif.transform(features_total) 
pred = clf.predict(features_total)
data['predicted_poi'] = pred


for i in range(0,len(pred)):
    if pred[i] == 1:
        print word_data[i]
       
for i in range(0,len(clf.feature_importances_)):
    if clf.feature_importances_[i] >= .005:
        print clf.feature_importances_[i], vectorizer.get_feature_names()[i]
        
