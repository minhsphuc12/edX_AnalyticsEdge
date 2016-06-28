# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 23:58:17 2016

@author: phucn_000
"""

"""
Spyder Editor

This is my submission to Machine Learning Course on Coursera. 
This containts heavily sample code provided in the course. 
I changed the dataset, number of tree 30
I also have to use write_pdf instead of create_png to create output
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
import sklearn.metrics
 # Feature Importance
from sklearn.ensemble import ExtraTreesClassifier

os.chdir("C:\Machine_learning")

#Load the dataset

AH_data = pd.read_csv("addhealth_pds.csv")
data_clean = AH_data.dropna()

data_clean.dtypes
data_clean.describe()

#Split into training and testing sets

predictors = data_clean[['BIO_SEX','H1GI4','H1GI5A','H1GI5B']]
predictors.dtypes
targets = data_clean.SCH_YR

pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, targets, test_size=.3)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=30)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)


# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)



print(model.feature_importances_)
""" [ 0.47544108  0.16881172  0.16961701  0.18613019]
display the relative importance of each attribute, correspondingly BIO_SEX, H1GI4, H1GI5, H1GI5B
"""

"""
Running a different number of trees and see the effect
 of that on the accuracy of the prediction
"""

trees=range(30)
accuracy=np.zeros(30)

for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
   
plt.cla()
graph = plt.plot(trees, accuracy)

"""
This plot show that for a small number of trees, the accuracy reaches its top
But after that, generally more trees is just equal less accurate (higher test error)
"""
