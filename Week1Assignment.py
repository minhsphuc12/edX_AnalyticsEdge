# -*- coding: utf-8 -*-
"""
Spyder Editor

This is my submission to Machine Learning Course on Coursera. 
This containts heavily sample code provided in the course. 
I changed the dataset, test ratio 0.2, and max_depth = 3
I also have to use write_pdf instead of create_png to create output
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

os.chdir("C:\Machine_learning")

"""
Data Engineering and Analysis
"""
#Load the dataset

AH_data = pd.read_csv("addhealth_pds.csv")
#read data
data_clean = AH_data.dropna()
#drop all NA in dataset

#data_clean.dtypes

# data_clean.describe()
#examine and summary data

"""
Modeling and Prediction
"""
#Split into training and testing sets

#predictors = data_clean[['BIO_SEX','CORE1','CORE2','HIEDBLK',
#'age','H1GI1M','H1GI1Y','H1GI2','H1GI6E','H1GI7F',
#'H1GI14','SMP07','CLUSTER1','CLUSTER2','H2GI1M']]

predictors = data_clean[['CLUSTER2','CLUSTER1']]


#predictors.dtypes

#define many data fields as predictors,

targets = data_clean.SCH_YR
# and one for target (dependent variables)

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, 
                                                targets, test_size=.2)
# with the order of train-test and pred-target, this train_test_spit() 
# function cut predictior set and target set into two: train subset and test subset. 
# test_size determine the relative size of test subset

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
tree.export_graphviz(classifier, out_file=out,max_depth=3)



import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.write_pdf('love2.pdf'))

