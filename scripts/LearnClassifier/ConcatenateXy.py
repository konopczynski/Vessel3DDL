# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:15:02 2016

@author: konopczynski
"""

import numpy as np
import pickle
import sys
sys.path.append('../')
import config as C
sys.path.append('../utils')
from VolumesToXy import serialize_xy


def read_temp_xy(path2xy, nmbrOfThreads, preffix):
    ArrayOf_Xtemp=[]
    ArrayOf_ytemp=[]
    for i in xrange(nmbrOfThreads):
        XfileName = preffix+'_X_'+str(i)+'.pkl' # +'_i.pkl'
        yfileName = preffix+'_y_'+str(i)+'.pkl' # +'_i.pkl'
        inputFile = open(path2xy+XfileName,'rb')
        Xtemp = pickle.load(inputFile)
        ArrayOf_Xtemp.append(Xtemp)
        inputFile.close()
        inputFile = open(path2xy+yfileName,'rb')
        ytemp = pickle.load(inputFile)
        ArrayOf_ytemp.append(ytemp)
        inputFile.close()
    return ArrayOf_Xtemp, ArrayOf_ytemp


def get_y(ytemp,nmbrOfScalesApplied,nmbrOfVolumes):
    """
    insert the yArray and extract the y labels
    Example:
    if nmbrOfScales = 5, nmbrOfVolumes=3
    we consider only first scale for each volume,
    ie ytemp[0],ytemp[5],ytemp[10]
    because the other scales have the same y labels
    Later concatenate labels from each volume
    ie y=np.concatenate((ytemp[0],ytemp[5],ytemp[10]),axis=0)
    and return y
    """
    
    ytempArray=[] 
    for i in xrange(nmbrOfVolumes):
        ytempArray.append(ytemp[0][i*nmbrOfScalesApplied])
    y = ytempArray[0]  # initialize y with 0
    for i in xrange(len(ytempArray)-1):
        y = np.concatenate((y,ytempArray[i+1]),axis=0)
    return y


def get_X(Xtemp,nmbrOfScalesApplied,nmbrOfVolumes,nmbrOfScales4Classifier, nmbrOfThreads):
    """
    Example:
    for nmbrOfScalesApplied = 5, nmbrOfVolumes = 3, 
    nmbrOfScales4Classifier = 5, nmbrOfThreads = 4
    
    #first part
    X00 = np.concatenate((X0[ 0],X1[ 0],X2[ 0],X3[ 0]),axis=1)
    X01 = np.concatenate((X0[ 1],X1[ 1],X2[ 1],X3[ 1]),axis=1)
    X02 = np.concatenate((X0[ 2],X1[ 2],X2[ 2],X3[ 2]),axis=1)
    X03 = np.concatenate((X0[ 3],X1[ 3],X2[ 3],X3[ 3]),axis=1)
    X04 = np.concatenate((X0[ 4],X1[ 4],X2[ 4],X3[ 4]),axis=1)
    X05 = np.concatenate((X0[ 5],X1[ 5],X2[ 5],X3[ 5]),axis=1)
    X06 = np.concatenate((X0[ 6],X1[ 6],X2[ 6],X3[ 6]),axis=1)
    X07 = np.concatenate((X0[ 7],X1[ 7],X2[ 7],X3[ 7]),axis=1)
    X08 = np.concatenate((X0[ 8],X1[ 8],X2[ 8],X3[ 8]),axis=1)
    X09 = np.concatenate((X0[ 9],X1[ 9],X2[ 9],X3[ 9]),axis=1)
    X10 = np.concatenate((X0[10],X1[10],X2[10],X3[10]),axis=1)
    X11 = np.concatenate((X0[11],X1[11],X2[11],X3[11]),axis=1)
    X12 = np.concatenate((X0[12],X1[12],X2[12],X3[12]),axis=1)
    X13 = np.concatenate((X0[13],X1[13],X2[13],X3[13]),axis=1)
    X14 = np.concatenate((X0[14],X1[14],X2[14],X3[14]),axis=1)

    #second part
    X_21=np.concatenate((X00,X01,X02,X03,X04),axis=1)
    X_22=np.concatenate((X05,X06,X07,X08,X09),axis=1)
    X_23=np.concatenate((X10,X11,X12,X13,X14),axis=1)
    
    #third part
    X = np.concatenate((X_21,X_22,X_23),axis=0)
    """
    
    XtempArray = [] #first part
    for i in xrange(nmbrOfVolumes*nmbrOfScalesApplied):
        Xcon = Xtemp[0][i] # initialize with 0
        for j in xrange(nmbrOfThreads-1):
            Xcon = np.concatenate((Xcon,Xtemp[j+1][ i]),axis=1)
        XtempArray.append(Xcon)
    
    XtempArray_volume = [] #second part
    for i in xrange(nmbrOfVolumes):
        Xcon = XtempArray[i*nmbrOfScalesApplied] # initialize with 0
        for j in xrange(nmbrOfScales4Classifier-1):
            Xcon = np.concatenate((Xcon,XtempArray[i*nmbrOfScalesApplied+1+j]),axis=1)
        XtempArray_volume.append(Xcon)
    
    #third part
    X = XtempArray_volume[0] # initialize with 0
    for i in xrange(nmbrOfVolumes-1):
        X = np.concatenate((X,XtempArray_volume[i+1]),axis=0)
    return X


def main():
    # Set the parameters:
    Param = C.read_parameters()
    number_of_volumes = Param.numOfVolumes
    number_of_parallel_threads  = Param.threads   
    number_of_scales_fmaps = Param.fMapS   
    number_of_scales_classifier = Param.clfS
    
    Xtemp, ytemp = read_temp_xy(Param.path2Xy_temp,number_of_parallel_threads,Param.dictionaryName)
    y = get_y(ytemp,number_of_scales_fmaps,number_of_volumes ) # i.e. y=np.concatenate((y_21,y_22,y_23),axis=0)
    X = get_X(Xtemp,number_of_scales_fmaps,number_of_volumes,number_of_scales_classifier,number_of_parallel_threads)
    serialize_xy(Param.path2Xy,X,y,Param.dictionaryName,"")
    return None

if __name__ == '__main__':
    main()
