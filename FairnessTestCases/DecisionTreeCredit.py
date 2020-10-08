

#Importing necessary files
import sys
sys.path.append('../')
import pandas as pd
import csv as cv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load


def func_main():
    

    #Reading the dataset
    df = pd.read_csv('Datasets/GermanCredit.csv') 

    data = df.values

    X = data[:, :-1]
    Y = data[:, -1]

    model = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=20, min_samples_split=2, 
                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=10)
    #Fitting the model with the dataset
    model = model.fit(X, Y)

    #dump(model, 'DecTreeCredit.joblib')

    return model

