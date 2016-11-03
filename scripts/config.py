# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:01:54 2016

@author: konopczynski

This is a config file for the VESSEL12 data.
The data should be placed at the ../../Data/
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV


def read_parameters():
    class Params:
        pass
    param = Params()
    # 0. Paths to data and serialization
    param.path            = '../../Data/'
    param.path2examples   = '../../Data/VESSEL12/VESSEL12_ExampleScans/'
    param.path2patches    = '../../Data/Serialized/saved_patches/'
    param.path2dicts      = '../../Data/Serialized/saved_dicts/'
    param.path2Xy_temp    = '../../Data/Serialized/saved_xy/Parallel/'
    param.path2Xy         = '../../Data/Serialized/saved_xy/'
    param.path2classifier = '../../Data/Serialized/saved_classifiers/'
    param.path2Measures   = '../../Data/Serialized/saved_measures/'
    param.path2Output     = '../../Data/Serialized/Output/'
    # create the paths if they don't exist
    create_paths([param.path2patches, param.path2dicts, param.path2Xy_temp,
                 param.path2Xy, param.path2classifier, param.path2Measures, param.path2Output])
    # 1. Extraction of patches
    param.number_of_patches = 100000
    param.paths2volumes_unannotated, param.paths2masks_unannotated = get_v_and_m_paths(param.path)
    param.patch_size = (5, 5, 5)
    param.sliceDim = (512, 512)
    param.patchSliceNum = (  # SliceNum - number of slices in a volume stacks from which patches are extracted
                    355, 415, 534, 426, 424,
                    375, 461, 442, 543, 426,
                    421, 446, 471, 386, 378,
                    451, 429, 408, 396, 406)
    param.patchScale = 5  # number of scales from which patches are extracted

    # 2. Dictionary learning
    param.numOfAtoms = 24  # numOfAtoms e.g. 512
    param.eS = 5    # elementSize e.g. 5x5x5
    param.bS = 5    # batch size
    param.nI = 300  # number of iterations
    param.aC = 1    # alpha constant

    param.fileWithPatches = (str(param.number_of_patches)+'patches_'+str(param.patch_size[0]) +
                             'ps_'+str(len(param.paths2volumes_unannotated))+'stacks')
    param.dictionaryName  = (str(param.nI)+'iter_'+str(param.numOfAtoms) +
                             'atoms_'+str(param.bS)+'bs_'+str(param.aC)+'a_'+str(param.eS)+'es_'+param.fileWithPatches)
    # 3. Feature map extraction #########################################

    param.paths2volumes, param.paths2masks, param.paths2annotations = get_annotated_v_and_m_paths(param.path2examples)
    param.numOfVolumes = 3
    param.sliceNum = (459, 448, 418)
    param.dim = ((459, 512, 512), (448, 512, 512), (418, 512, 512))
    param.anno = load_annotations(param.path2examples, param.paths2annotations)  # Read the annotations in csv format
    param.clfS    = 2  # Scales to use for the classifier
    param.fMapS   = 2  # num of scales used to create the feature maps
    param.threads = 4  # the more memory you have the higher it can be
    # 4. Training the classifier
    param.classifiers = {  # C=1
        'CV_Logit_newton_L2': LogisticRegressionCV(penalty='l2', solver='newton-cg', n_jobs=1, max_iter=300),
        'CV_Logit_lbfgs_L2':  LogisticRegressionCV(penalty='l2', solver='lbfgs',     n_jobs=1, max_iter=300)}
    param.clf2use = 'CV_Logit_newton_L2'  # classifier to use
    param.clfName = param.clf2use+'_'+param.dictionaryName
    # 5. Measurements
    param.ratio     = [0.256]
    param.StoTry    = [2]  # scales to try. one can try many e.g. [2,3,4]
    param.numOfEval = 100  # number of evaluations
    # 7. Usage: Segment vessels #########################################
    param.volumeToProcess = 'Scans/VESSEL12_21.raw'
    param.outputFn = 'OutputVolume.npy'
    return param


def get_v_and_m_paths(path):
    paths2volumes = []
    paths2masks = []
    # Scans: VESSEL12_01-05
    paths2volumes.append(path+'VESSEL12/VESSEL12_01-05/VESSEL12_01.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_01-05/VESSEL12_02.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_01-05/VESSEL12_03.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_01-05/VESSEL12_04.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_01-05/VESSEL12_05.raw')

    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_01.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_02.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_03.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_04.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_05.raw')
    
    # Scans: VESSEL12_06-10
    paths2volumes.append(path+'VESSEL12/VESSEL12_06-10/VESSEL12_06.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_06-10/VESSEL12_07.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_06-10/VESSEL12_08.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_06-10/VESSEL12_09.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_06-10/VESSEL12_10.raw')

    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_06.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_07.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_08.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_09.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_10.raw')
    
    # Scans: VESSEL12_11-15
    paths2volumes.append(path+'VESSEL12/VESSEL12_11-15/VESSEL12_11.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_11-15/VESSEL12_12.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_11-15/VESSEL12_13.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_11-15/VESSEL12_14.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_11-15/VESSEL12_15.raw')

    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_11.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_12.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_13.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_14.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_15.raw')

    # Scans: VESSEL12_16-20
    paths2volumes.append(path+'VESSEL12/VESSEL12_16-20/VESSEL12_16.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_16-20/VESSEL12_17.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_16-20/VESSEL12_18.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_16-20/VESSEL12_19.raw')
    paths2volumes.append(path+'VESSEL12/VESSEL12_16-20/VESSEL12_20.raw')

    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_16.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_17.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_18.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_19.raw')
    paths2masks.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_20.raw')
    return paths2volumes, paths2masks


def get_annotated_v_and_m_paths(path):
    paths2volumes = []
    paths2masks = []
    paths2annotations = []
    paths2volumes.append(path+'Scans/VESSEL12_21.raw')
    paths2masks.append(path+'Lungmasks/VESSEL12_21.raw')
    paths2annotations.append('Annotations/VESSEL12_21_Annotations.csv')
    
    paths2volumes.append(path+'Scans/VESSEL12_22.raw')
    paths2masks.append(path+'Lungmasks/VESSEL12_22.raw')
    paths2annotations.append('Annotations/VESSEL12_22_Annotations.csv')

    paths2volumes.append(path+'Scans/VESSEL12_23.raw')
    paths2masks.append(path+'Lungmasks/VESSEL12_23.raw')
    paths2annotations.append('Annotations/VESSEL12_23_Annotations.csv')
    return paths2volumes, paths2masks, paths2annotations


def load_annotations(path, paths2annotations):
    loaded_annotation = []
    for i in range(len(paths2annotations)):
        if paths2annotations[i][-3:] == 'csv':
            df = pd.read_csv(path+paths2annotations[i], sep=',', header=None)
            loaded_annotation.append(df.values)
        else:
            input_file_a = paths2annotations[i]  # open(paths2annotations[i], 'rb')
            a = np.load(input_file_a)
            loaded_annotation.append(a)
    return loaded_annotation


def create_paths(directories):
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)
    return None
