

#Importing necessary files
import sys
sys.path.append('../')
import pandas as pd
import csv as cv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

def func_main():
    

    #Reading the dataset
    df = pd.read_csv('Datasets/Adult.csv') 

    data = df.values

    X = data[:, :-1]
    Y = data[:, -1]

    model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=5, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                       oob_score=False, random_state=None, verbose=0,
                       warm_start=False)

    #Fitting the model with the dataset
    model = model.fit(X, Y)
    #dump(model, 'RanForestAdult.joblib')

    return model


