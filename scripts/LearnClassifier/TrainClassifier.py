# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:15:02 2016

@author: konop
"""

import numpy as np
import pickle
import sys
import os
from sklearn.externals import joblib
from sklearn.utils import shuffle
sys.path.append('../')
import config as C
from VolumesToXy import Read_xy

def SerializeClassifier(path2save,file_name,clf):
    filepath=path2save+file_name
    joblib.dump(clf, filepath)
    print("saved to: "+filepath)
    return None
    
def TrainClassifier(name_c,numberOfAtoms,scales,classifier,X_train,y_train):
    y = y_train.reshape(len(y_train))
    X = X_train[:,0:numberOfAtoms*scales]
    X, y = shuffle(X, y, random_state=1) # change the random state
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
    print("classif_rate for %s : %f " % (name_c, classif_rate))
    return classifier

if __name__ == '__main__':
    # set parameters
    Param = C.ReadParameters()
    numberOfAtoms = Param.numOfAtoms
    scales = Param.fMapS
    X,y=Read_xy(Param.path2Xy,preffix=Param.dicoName+'_')
    classifiers = Param.classifiers
    clf2save = TrainClassifier(Param.clf2use,numberOfAtoms,scales,classifiers[Param.clf2use],X,y)
    SerializeClassifier(Param.path2Clfs,Param.clfName,clf2save)
