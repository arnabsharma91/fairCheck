

#Importing necessary files
import sys
sys.path.append('../')
import pandas as pd
import csv as cv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from joblib import dump, load


def func_main():
    #Reading the dataset
    df = pd.read_csv('Datasets/Adult.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    model = BernoulliNB()
    #Fitting the model with the dataset
    model = model.fit(X, Y)
    dump(model, 'NBAdult.joblib')
    
    return model

