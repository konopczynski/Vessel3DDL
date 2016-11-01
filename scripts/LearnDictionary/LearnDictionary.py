# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:56:11 2016

@author: konop

Perform the dictionary learning for a given settings,
on the provided patches
"""

import numpy as np
import pickle
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import DictionaryLearning
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append('../')
import config as C

#@profile
def dico_mini(nP, n_c=512, a=1, n_i=800, n_j = 3, b_s=3, es=5, fit_algorithm='lars'):
    """
    nP  - patches to learn on (nP stands for normalized patches)
    n_c - number of components (atoms) e.g. 512
    a   - alpha sparsity controlling parameter
    n_i - total number of iterations to perform
    b_s - batch size: number of samples in each mini-batch
    fit_algorithm - {‘lars’, ‘cd’} 
    n_j - number of parallel jobs to run (number of threads)
    e_s - size of each element in the dictionary
    """
    dic = MiniBatchDictionaryLearning(n_components=n_c, alpha=a,   n_iter=n_i, n_jobs = n_j, batch_size=b_s,fit_algorithm=fit_algorithm)
    print ("Start learning dico_mini: n_c: "+str(n_c)+", a: " +str(a) +", n_i: " +str(n_i)+ ", n_j: "+str(n_j)+", es: "+str(es)+", b_s: "+str(b_s))
    V1 = dic.fit(nP).components_
    D1 = V1.reshape(n_c,es,es,es) # 512x5x5x5
    return D1

#@profile
def dico(nP, n_c=512, a=1, n_i=100, n_j = 3, es=5):
    dic =          DictionaryLearning(n_components=n_c, alpha=a, max_iter=n_i, n_jobs = n_j,fit_algorithm=fit_algorithm)
    print ("Start learning dico:      n_c: "+str(n_c)+", a: " +str(a) +", n_i: " +str(n_i)+", es: "+str(es)+ ", n_j: "+str(n_j))
    V2 = dic.fit(nP).components_
    D2 = V2.reshape(n_c,es,es,es) # 512x5x5x5
    return D2

def serialize_dico(D,path2saveDict,file_name):
    tosave = path2saveDict+file_name
    output = open(tosave, 'wb')
    pickle.dump(D, output)
    output.close()
    print("saved at: "+tosave)
    return None

if __name__ == '__main__':
    P = C.ReadParameters()
    path2patches  = P.path2patches
    path2saveDict = P.path2dicts
    nA = P.numOfAtoms
    eS = P.eS
    bS = P.bS
    nI = P.nI
    aC = P.aC

    FileWithPatches = P.path2patches+P.Fpatches+'.npy'
    nP = np.load(FileWithPatches)
    dicoName = P.dicoName
    # Learn the dictionary
    D1 = dico_mini(nP, n_c=nA, a=aC, n_i=nI, n_j = 4 , b_s=bS, es=eS, fit_algorithm='lars')
    # Serialize the dictionary
    file_name = dicoName+'.pkl'
    serialize_dico(D1,path2saveDict,file_name)
