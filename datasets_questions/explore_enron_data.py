#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import sys
import os

os
if '..\\final_project' not in sys.path:
    sys.path.append('..\\final_project')
    
if '..\\datasets_questions' not in sys.path:
    sys.path.append('..\\datasets_questions')


import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print len(enron_data.keys())

print len(enron_data[enron_data.keys()[0]].keys())

poi_count = 0
for person in enron_data:
    if enron_data[person]['poi'] == True:
        poi_count += 1
    else:
        pass
    
print poi_count

from poi_email_addresses import poiEmails

print len(poiEmails())

print enron_data['PRENTICE JAMES']['total_stock_value']

print enron_data['COLWELL WESLEY']['from_this_person_to_poi']

print enron_data['SKILLING JEFFREY K']['exercised_stock_options']

print enron_data['FASTOW ANDREW S']['total_payments']
print enron_data['LAY KENNETH L']['total_payments']
print enron_data['SKILLING JEFFREY K']['total_payments']
type(poi_count)

salary_count = 0
email_address_count = 0
for person in enron_data:
    if enron_data[person]['salary'] != 'NaN':
        salary_count += 1
    if enron_data[person]['email_address'] != 'NaN':
        email_address_count += 1
    
print salary_count,' salaries, ',email_address_count,' emails' 

total_payments_count = 0
for person in enron_data:
    if enron_data[person]['total_payments'] != 'NaN':
        total_payments_count += 1

missing_payments = len(enron_data.keys())-total_payments_count
percent_missing_payments = float(missing_payments) / len(enron_data.keys())        
print missing_payments,' people, ',percent_missing_payments*100,' percent' 


poi_count = 0
poi_nan_count = 0
for person in enron_data:
    if enron_data[person]['poi'] == True:
        poi_count += 1        
        if enron_data[person]['total_payments'] == 'NaN':
            poi_nan_count += 1
    else:
        pass
    
print (35-(poi_count-poi_nan_count)) / float(35)

