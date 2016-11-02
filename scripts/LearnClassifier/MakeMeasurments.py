# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:15:02 2016

@author: konop
"""

import numpy as np
import pickle
import sys
import os
import random
# choose classifiers
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from VolumesToXy import Read_xy
import config as C
sys.path.append('../')
import warnings
warnings.filterwarnings("ignore")

def SerializeMeasure(path2save,file_name,measure):
    filepath=path2save+file_name+".npy"
    output = open(filepath, 'wb')
    np.save(output, measure, allow_pickle=True, fix_imports=True)
    output.close()
    print("saved to: "+filepath)
    return None

def MakeAvgMeasure(numberOfAtoms, ratio_set, scales, number_of_evaluation,X,y):
    # Compute an average accuracy
    # Define classifiers
    classifiers = Param.classifiers
    stacked_measure={}
    
    y_raw = y.reshape(len(y))
    for scale in scales:
        X_raw = X[:,0:numberOfAtoms*scale]
        stacked_avg={}
        for ratio in ratio_set:
            avg = {
                'CV_Logit_newton_L2': [],
                'CV_Logit_lbfgs_L2': []}
            for i in range(number_of_evaluation):
                print("scale: "+str(scale)+" i: "+str(i))
                X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_raw, y_raw, test_size=ratio, random_state=random.randrange(8899999))
                X=X_train
                y=y_train
                print("y_pred : y_train")
                for index, (name_c, classifier) in enumerate(classifiers.items()):
                    classifier.fit(X, y)
                    y_pred = classifier.predict(X)
                    classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
                    print("classif_rate for %s : %f " % (name_c, classif_rate))
                print("y_pred : y_test")
                #
                # Check on test data
                X=X_test
                y=y_test
                for index, (name_c, classifier) in enumerate(classifiers.items()):
                    y_pred = classifier.predict(X)
                    classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
                    print("classif_rate for %s : %f " % (name_c, classif_rate))
                    avg[name_c].append(classif_rate)
            stacked_avg[str(ratio)]=avg
        stacked_measure[str(scale)]=stacked_avg
    return stacked_measure

if __name__ == '__main__':
    # set parameters
    Param = C.ReadParameters()
    # set measurments parameters
    ratio_set = Param.ratio
    scales    = Param.StoTry # you can try many e.g. [2,3,4]
    number_of_evaluations= Param.numOfEval
    print(Param.dicoName)
    X,y=Read_xy(Param.path2Xy,preffix=Param.dicoName+'_')
    y = y.reshape(len(y))
    measure = MakeAvgMeasure(Param.numOfAtoms, ratio_set, scales, number_of_evaluations,X,y)
    print("#################################################")
    for ratio in ratio_set:
        for numberOfScales in scales:
            print "numberOfScales: "+str(numberOfScales) + " ratio: "+str(ratio) + \
                  " number of eval: "+str(number_of_evaluations) + " numberOfAtoms: " + str(Param.numOfAtoms)
            print("Avg lbfgs L2:     "+str(np.mean(measure[str(numberOfScales)][str(ratio)]['CV_Logit_lbfgs_L2'])))
            print("std lbfgs L2:     "+str(np.std(measure[str(numberOfScales)][str(ratio)]['CV_Logit_lbfgs_L2'])))
            print("Avg newton L2:    "+str(np.mean(measure[str(numberOfScales)][str(ratio)]['CV_Logit_newton_L2'])))
            print("std newton L2:    "+str(np.std(measure[str(numberOfScales)][str(ratio)]['CV_Logit_newton_L2'])))
    SerializeMeasure(Param.path2Measures,Param.dicoName,measure)
