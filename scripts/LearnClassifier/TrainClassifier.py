# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:15:02 2016

@author: konopczynski
"""

import numpy as np
import sys
from sklearn.externals import joblib
from sklearn.utils import shuffle
sys.path.append('../')
sys.path.append('../utils')
import config
from VolumesToXy import read_xy


def serialize_classifier(path2save, file_name, clf):
    path2file = path2save+file_name
    joblib.dump(clf, path2file)
    print("saved to: "+path2file)
    return None


def train_classifier(name_c, number_of_atoms, scales, classifier, X_train, y_train):
    y = y_train.reshape(len(y_train))
    X = X_train[:, 0:number_of_atoms*scales]
    X, y = shuffle(X, y, random_state=1)  # change the random state
    classifier.fit(X, y)
    y_predicted = classifier.predict(X)
    classification_rate = np.mean(y_predicted.ravel() == y.ravel()) * 100
    print("classif_rate for %s : %f " % (name_c, classification_rate))
    return classifier


def main():
    # set parameters
    param = config.read_parameters()
    scales = param.fMapS
    X, y = read_xy(param.path2Xy, preffix=param.dictionaryName+'_')
    classifiers = param.classifiers
    clf2save = train_classifier(param.clf2use, param.numOfAtoms, scales, classifiers[param.clf2use], X, y)
    serialize_classifier(param.path2classifier, param.clfName, clf2save)

if __name__ == '__main__':
    main()
