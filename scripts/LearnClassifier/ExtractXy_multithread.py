# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:56:11 2016

@author: konopczynski
Computes the Feature Maps and extract feature vectors
for the annotated voxels.
The number of threads is hardcoded
"""

import pickle
import sys
sys.path.append('../')
sys.path.append('../utils')
import config
from VolumesToXy import apply_filters, serialize_xy
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial


def apply_feature_maps(param, p):
    # d is a number of threads
    d = param.threads
    # Read dictionary
    input_file = open(param.path2dicts+param.dictionaryName+'.pkl', 'rb')
    dictionary = pickle.load(input_file)  # load the dictionary
    l = len(dictionary)/d
    dictionary = dictionary[p*l:(p+1)*l]  # consider only the d-th number of atoms
    input_file.close()
    # Apply filters
    XX, yy = apply_filters(param, dictionary)
    # Serialize
    serialize_xy(param.path2Xy_temp, XX=XX, yy=yy, preffix=param.dictionaryName, suffix='_'+str(p))
    return None


def main():
    param = config.read_parameters()
    parts = range(0, param.threads, 1)
    partial_apply_feature_maps = partial(apply_feature_maps, param)
    pool = ThreadPool(param.threads)
    pool.map(partial_apply_feature_maps, parts)
    pool.close()
    pool.join()
    return None

if __name__ == '__main__':
    main()
