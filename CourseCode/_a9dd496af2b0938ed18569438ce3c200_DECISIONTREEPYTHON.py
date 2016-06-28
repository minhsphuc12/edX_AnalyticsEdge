# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:12:54 2015

@author: ldierker
"""

# -*- coding: utf-8 -*-

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

os.chdir("C:\TREES")

"""
Data Engineering and Analysis
"""
#Load the dataset

AH_data = pd.read_csv("tree_addhealth.csv")
#read data
data_clean = AH_data.dropna()
#drop all NA in dataset
data_clean.dtypes
data_clean.describe()
#examine and summary data

"""
Modeling and Prediction
"""
#Split into training and testing sets

predictors = data_clean[['BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN',
'age','ALCEVR1','ALCPROBS1','marever1','cocever1','inhever1','cigavail','DEP1',
'ESTEEM1','VIOL1','PASSIST','DEVIANT1','SCHCONN1','GPA1','EXPEL1','FAMCONCT','PARACTV',
'PARPRES']]
#define many data fields as predictors,

targets = data_clean.TREG1
# and one for target (dependent variables)

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.4)
# with the order of train-test and pred-target, this train_test_spit() function cut predictior set and target set into two: train subset and test subset. 
#test_size determine the relative size of test subset

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape
#check the shape (what it means??) of four subset created 

#Build model on training data
classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)
 #make predictions
predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())




