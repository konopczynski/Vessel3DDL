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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
sys.path.append('../')

def ReadParameters():
    class R_params():
        pass
    Param=R_params()
    # 0. Paths to data and serialiation
    Param.path          = '../../Data/'
    Param.path2examples = '../../Data/VESSEL12/VESSEL12_ExampleScans/'
    Param.path2patches  = '../../Data/Serialized/saved_patches/'
    Param.path2dicts    = '../../Data/Serialized/saved_dicts/'
    Param.path2Xy_temp  = '../../Data/Serialized/saved_xy/Parallel/'
    Param.path2Xy       = '../../Data/Serialized/saved_xy/'
    Param.path2Clfs     = '../../Data/Serialized/saved_classifiers/'
    Param.path2Measures = '../../Data/Serialized/saved_measures/'
    Param.path2Output   = '../../Data/Serialized/Output/'
    # create the paths if they dont exist
    CreatePaths([Param.path2patches,Param.path2dicts,Param.path2Xy_temp,Param.path2Xy,Param.path2Clfs,Param.path2Measures,Param.path2Output])
    # 1. Extraction of patches
    Param.nP = 1000
    Param.patchVpaths, Param.patchMpaths=getVandMpaths(Param.path)
    Param.patch_size=(5,5,5)
    Param.sliceDim=(512,512)
    Param.patchSliceNum =(# SliceNum - number of slices in a volume stacks from which patches are extracted
                    355,415,534,426,424,
                    375,461,442,543,426,
                    421,446,471,386,378,
                    451,429,408,396,406)
    Param.patchScale = 5 # number of scales from which patches are extracted

    # 2. Dicionary learning 
    Param.numOfAtoms = 24 # numOfAtoms e.g. 512
    Param.eS = 5 # elementSize e.g. 5x5x5
    Param.bS = 5 # batch size
    Param.nI = 300 # number of iterations
    Param.aC = 1 # alpha constant

    Param.Fpatches = str(Param.nP)+'patches_'+str(Param.patch_size[0])+'ps_'+str(len(Param.patchVpaths))+'stacks'
    Param.dicoName = str(Param.nI)+'iter_'+str(Param.numOfAtoms)+'atoms_'+str(Param.bS)+'bs_'+str(Param.aC)+'a_'+str(Param.eS)+'es_'+Param.Fpatches
    # 3. Feature map extraction #########################################

    Param.AVpaths,Param.AMpaths,Param.Apaths = getA_VandMpaths(Param.path2examples)
    Param.numOfVolumes = 3
    Param.sliceNum=(459,448,418)
    Param.dim = ((459,512,512),(448,512,512),(418,512,512))
    Param.anno = LoadAnnotations(Param.path2examples,Param.Apaths) # Read the annotations in the csv format
    Param.clfS    = 2 # Scales to use for the classifier
    Param.fMapS   = 2 # num of scales used to create the feature maps
    Param.threads = 4 # the more memory you have the higher it can be
    # 4. Training the classifier
    Param.classifiers ={ # C=1
        'CV_Logit_newton_L2': LogisticRegressionCV(penalty='l2',solver='newton-cg', n_jobs=1,max_iter=300),
        'CV_Logit_lbfgs_L2':  LogisticRegressionCV(penalty='l2',solver='lbfgs',     n_jobs=1,max_iter=300)}
    Param.clf2use = 'CV_Logit_newton_L2' # classifier to use
    Param.clfName = Param.clf2use+'_'+Param.dicoName
    # 5. Measurments
    Param.ratio     = [0.256]
    Param.StoTry    = [2] # scales to try. one can try many e.g. [2,3,4]
    Param.numOfEval = 1000 # number of evaluations
    # 7. Usage: Segment vessels #########################################
    Param.volumeToProcess = 'Scans/VESSEL12_21.raw'
    Param.outputFn = 'OutputVolume.npy'
    return Param

def getVandMpaths(path):
    Vpath=[]
    Mpath=[]
    # Scans: VESSEL12_01-05
    Vpath.append(path+'VESSEL12/VESSEL12_01-05/VESSEL12_01.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_01-05/VESSEL12_02.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_01-05/VESSEL12_03.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_01-05/VESSEL12_04.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_01-05/VESSEL12_05.raw')
    
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_01.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_02.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_03.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_04.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_05.raw')
    
    # Scans: VESSEL12_06-10
    Vpath.append(path+'VESSEL12/VESSEL12_06-10/VESSEL12_06.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_06-10/VESSEL12_07.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_06-10/VESSEL12_08.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_06-10/VESSEL12_09.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_06-10/VESSEL12_10.raw')
    
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_06.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_07.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_08.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_09.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_10.raw')
    
    # Scans: VESSEL12_11-15
    Vpath.append(path+'VESSEL12/VESSEL12_11-15/VESSEL12_11.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_11-15/VESSEL12_12.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_11-15/VESSEL12_13.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_11-15/VESSEL12_14.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_11-15/VESSEL12_15.raw')
    
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_11.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_12.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_13.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_14.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_15.raw')
    
    
    # Scans: VESSEL12_16-20
    Vpath.append(path+'VESSEL12/VESSEL12_16-20/VESSEL12_16.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_16-20/VESSEL12_17.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_16-20/VESSEL12_18.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_16-20/VESSEL12_19.raw')
    Vpath.append(path+'VESSEL12/VESSEL12_16-20/VESSEL12_20.raw')
    
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_16.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_17.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_18.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_19.raw')
    Mpath.append(path+'VESSEL12/VESSEL12_01-20_Lungmasks/VESSEL12_20.raw')
    return Vpath,Mpath

def getA_VandMpaths(path):
    Vpath = []
    Mpath = []
    Apath = []
    Vpath.append(path+'Scans/VESSEL12_21.raw')
    Mpath.append(path+'Lungmasks/VESSEL12_21.raw')
    Apath.append('Annotations/VESSEL12_21_Annotations.csv')
    
    Vpath.append(path+'Scans/VESSEL12_22.raw')
    Mpath.append(path+'Lungmasks/VESSEL12_22.raw')
    Apath.append('Annotations/VESSEL12_22_Annotations.csv')

    Vpath.append(path+'Scans/VESSEL12_23.raw')
    Mpath.append(path+'Lungmasks/VESSEL12_23.raw')
    Apath.append('Annotations/VESSEL12_23_Annotations.csv')
    return (Vpath,Mpath,Apath)

def LoadAnnotations(path,Apaths):
    LoadedAdnotation = []
    for i in range(len(Apaths)):
        if Apaths[i][-3:] == 'csv':
            df = pd.read_csv(path+Apaths[i], sep=',',header=None)
            LoadedAdnotation.append(df.values)
        else:
            inputFile_A = Apaths[i] #open(Apaths[i], 'rb')
            A = np.load(inputFile_A)
            LoadedAdnotation.append(A)
    return LoadedAdnotation

def CreatePaths(directories):
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)
    return None
