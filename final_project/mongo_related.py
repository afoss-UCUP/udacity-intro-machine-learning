# -*- coding: utf-8 -*-


import json

#get mongo database
def get_db():
    from pymongo import MongoClient
    client = MongoClient('localhost:27017')
    db = client.enron_emails
    return db

#aggregate function for mongo queries
def aggregate(db, pipeline):
    result = db.enron_emails.aggregate(pipeline)
    return result

#add denver_boulder line data from json
def insert_data(data, db):
    
    db.insert(data)

    pass

#imports json documents iteratively from a file
def import_to_db(json_file):
    
    db = get_db()
    db = db.enron_emails
    
    with open(json_file) as f:
        for line in f:
           data = json.loads(line)
           insert_data(data, db)
