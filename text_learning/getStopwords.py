# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:00:39 2015

@author: Aaron Foss
"""

def getStopwords():
    
    from nltk.corpus import stopwords
    sw = stopwords.words("english")
    return sw