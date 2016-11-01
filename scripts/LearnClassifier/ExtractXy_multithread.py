# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:56:11 2016

@author: konop
Computes the Feature Maps and extract feature vectors
for the annotated voxels.
The number of threads is hardcoded
"""

import pickle
import sys
import os
sys.path.append('../')
import config as C
from VolumesToXy import ApplyFilters, Serialize_Xy
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

def applyF(Param,p):
    # d is a number of threads
    d=Param.threads
    Dictpath = Param.path2dicts
    path2saveXandY_temp = Param.path2Xy_temp
    # Read dictionary
    dicoName = Param.dicoName
    inDict = dicoName+'.pkl'
    preffix= dicoName
    inputFile = open(Dictpath+inDict,'rb')
    D = pickle.load(inputFile) # load the dictionary
    l = len(D)/d   
    D = D[(p)*l:(p+1)*l] # consider only the d-th number of atoms
    inputFile.close()
    # Apply filters
    XX,yy=ApplyFilters(Param,D)
    # Serialize
    Serialize_Xy(path2saveXandY_temp,XX=XX,yy=yy,preffix=preffix,suffix='_'+str(p))
    return None

if __name__ == '__main__':
    Param = C.ReadParameters()
    parts = range(0,Param.threads,1)
    partial_applyF = partial(applyF,Param)
    pool = ThreadPool(Param.threads)
    pool.map(partial_applyF, parts)
    pool.close()
    pool.join()