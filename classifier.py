#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 17:37:36 2018

@author: nparti
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

class Classifier():
    
    def __init__(self):
        self.model = None
    
    def fit(self, x, y):
        self.model = GaussianNB()
        self.model.fit(X=x, y=y)
        return self.model
        
    def optimized_fit(self, x, y, model: GridSearchCV):
        self.model = model
        self.model.fit(x,y)
        return self.model
    
def sclale_data():
    pass

def create_grid_cv():
    pass

def split_data():
    pass