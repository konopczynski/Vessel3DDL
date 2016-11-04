# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:15:02 2016

@author: konop
"""

import numpy as np
import sys
import random
import warnings
from sklearn import cross_validation
sys.path.append('../')
import config
sys.path.append('../utils')
from VolumesToXy import read_xy
warnings.filterwarnings("ignore")


def serialize_measurements(path2save, file_name, measure):
    path2file = path2save+file_name+".npy"
    output = open(path2file, 'wb')
    np.save(output, measure, allow_pickle=True, fix_imports=True)
    output.close()
    print("saved to: "+path2file)
    return None


def averaged_measurement(param, ratio_set, scales, number_of_evaluation, X, y):
    # Compute an average accuracy
    # Define classifiers
    classifiers = param.classifiers
    stacked_measure={}
    
    y_raw = y.reshape(len(y))
    for scale in scales:
        X_raw = X[:, 0:param.numOfAtoms*scale]
        stacked_avg = {}
        for ratio in ratio_set:
            avg = {
                'CV_Logit_newton_L2': [],
                'CV_Logit_lbfgs_L2': []}
            for i in range(number_of_evaluation):
                print("scale: "+str(scale)+" i: "+str(i))
                X_train, X_test, y_train, y_test = \
                    cross_validation.train_test_split(X_raw, y_raw,
                                                      test_size=ratio, random_state=random.randrange(8899999))
                X = X_train
                y = y_train
                print("y_prediction : y_train")
                for index, (name_c, classifier) in enumerate(classifiers.items()):
                    classifier.fit(X, y)
                    y_prediction = classifier.predict(X)
                    classification_rate = np.mean(y_prediction.ravel() == y.ravel()) * 100
                    print("classification_rate for %s : %f " % (name_c, classification_rate))
                print("y_prediction : y_test")
                #
                # Check on test data
                X = X_test
                y = y_test
                for index, (name_c, classifier) in enumerate(classifiers.items()):
                    y_prediction = classifier.predict(X)
                    classification_rate = np.mean(y_prediction.ravel() == y.ravel()) * 100
                    print("classification_rate for %s : %f " % (name_c, classification_rate))
                    avg[name_c].append(classification_rate)
            stacked_avg[str(ratio)] = avg
        stacked_measure[str(scale)] = stacked_avg
    return stacked_measure


def main():
    # set parameters
    param = config.read_parameters()
    # set measurements parameters
    ratio_set = param.ratio
    scales    = param.StoTry  # you can try many e.g. [2,3,4]
    number_of_evaluations = param.numOfEval
    print(param.dictionaryName)
    X, y = read_xy(param.path2Xy, preffix=param.dictionaryName+'_')
    y = y.reshape(len(y))
    measure = averaged_measurement(param, ratio_set, scales, number_of_evaluations,X,y)
    print("#################################################")
    for ratio in ratio_set:
        for numberOfScales in scales:
            print "numberOfScales: "+str(numberOfScales) + " ratio: "+str(ratio) + \
                  " number of eval: "+str(number_of_evaluations) + " numberOfAtoms: " + str(param.numOfAtoms)
            print("Avg lbfgs L2:     "+str(np.mean(measure[str(numberOfScales)][str(ratio)]['CV_Logit_lbfgs_L2'])))
            print("std lbfgs L2:     "+str(np.std(measure[str(numberOfScales)][str(ratio)]['CV_Logit_lbfgs_L2'])))
            print("Avg newton L2:    "+str(np.mean(measure[str(numberOfScales)][str(ratio)]['CV_Logit_newton_L2'])))
            print("std newton L2:    "+str(np.std(measure[str(numberOfScales)][str(ratio)]['CV_Logit_newton_L2'])))
    serialize_measurements(param.path2Measures, param.dictionaryName, measure)
    return None

if __name__ == '__main__':
    main()
