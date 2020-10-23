

#Importing necessary files
import sys
sys.path.append('../')
import pandas as pd
import csv as cv
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump, load


def func_main():
    #Reading the dataset
    df = pd.read_csv('Datasets/GermanCredit.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                           class_weight=None, random_state=None, solver='lbfgs', max_iter=5000, multi_class='auto',
                           verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

    #Fitting the model with the dataset
    model = model.fit(X, Y)
    dump(model, 'LogRegCredit.joblib')
    
    return model

