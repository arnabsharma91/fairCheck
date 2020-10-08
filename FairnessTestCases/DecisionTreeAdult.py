

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
    df = pd.read_csv('Datasets/Adult.csv') 

    data = df.values

    X = data[:, :-1]
    Y = data[:, -1]

    model = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0,
                       random_state=None, splitter='best')

    #Fitting the model with the dataset
    model = model.fit(X, Y)
    
    
    dump(model, 'DecTreeAdult.joblib')

    return model



