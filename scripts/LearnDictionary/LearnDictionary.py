# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:56:11 2016

@author: konopczynski

Perform the dictionary learning for a given settings,
on the provided patches
"""

import numpy as np
import pickle
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import DictionaryLearning
import warnings
import sys
sys.path.append('../')
import config
warnings.filterwarnings("ignore")


def learn_dictionary_mini(patches, n_c=512, a=1, n_i=800, n_j=3, b_s=3, es=5, fit_algorithm='lars'):
    """
    patches  - patches to learn on (should be normalized before)
    n_c - number of components (atoms) e.g. 512
    a   - alpha sparsity controlling parameter
    n_i - total number of iterations to perform
    b_s - batch size: number of samples in each mini-batch
    fit_algorithm - {‘lars’, ‘cd’}
    n_j - number of parallel jobs to run (number of threads)
    e_s - size of each element in the dictionary
    """
    dic = MiniBatchDictionaryLearning(n_components=n_c, alpha=a, n_iter=n_i,
                                      n_jobs=n_j, batch_size=b_s, fit_algorithm=fit_algorithm)
    print ("Start learning dictionary_mini: n_c: "+str(n_c)+", alpha: "+str(a)+", n_i: " +
           str(n_i)+", n_j: "+str(n_j)+", es: "+str(es)+", b_s: "+str(b_s))
    v1 = dic.fit(patches).components_
    d1 = v1.reshape(n_c, es, es, es)  # e.g. 512x5x5x5
    return d1


def learn_dictionary(patches, n_c=512, a=1, n_i=100, n_j=3, es=5, fit_algorithm='lars'):
    dic = DictionaryLearning(n_components=n_c, alpha=a, max_iter=n_i,
                             n_jobs=n_j, fit_algorithm=fit_algorithm)
    print ("Start learning dictionary: n_c: "+str(n_c)+", alpha: "+str(a)+", n_i: " +
           str(n_i)+", es: "+str(es)+", n_j: "+str(n_j))
    v2 = dic.fit(patches).components_
    d2 = v2.reshape(n_c, es, es, es)  # e.g. 512x5x5x5
    return d2


def serialize_dictionary(d, path2save):
    full_saving_path = path2save
    output = open(full_saving_path, 'wb')
    pickle.dump(d, output)
    output.close()
    print("saved at: "+full_saving_path)
    return None


def main():
    param = config.read_parameters()
    na = param.numOfAtoms
    es = param.eS
    bs = param.bS
    ni = param.nI
    ac = param.aC

    file_with_patches = param.path2patches+param.fileWithPatches+'.npy'
    patches = np.load(file_with_patches)
    # Learn the dictionary
    dictionary = learn_dictionary_mini(patches, n_c=na, a=ac, n_i=ni,
                                       n_j=4, b_s=bs, es=es, fit_algorithm='lars')
    # Serialize the dictionary
    path2save_dictionary = param.path2dicts+param.dictionaryName+'.pkl'
    serialize_dictionary(dictionary, path2save_dictionary)
    return None

if __name__ == '__main__':
    main()
