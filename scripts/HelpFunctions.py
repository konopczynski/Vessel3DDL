import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction import image
from sklearn.decomposition import MiniBatchDictionaryLearning
from scipy import ndimage
import pyqtgraph as pg
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from skimage.transform import pyramid_gaussian
from skimage.transform import pyramid_expand
from sklearn.decomposition import MiniBatchDictionaryLearning
import sys
sys.path.append('./')
import patches_3d  as p
import pyramids_3d as pyr
import os
import time

# TBD: Most of those imports are unncessary

# some help functions
def ReadVolumes(SliceDim,SliceNum, paths):
    # Read int16 as specified in README of vessel12 database
    z,x,y = np.sum(SliceNum), SliceDim[0], SliceDim[1]
    volume2read = np.empty((z,x,y), np.int16)
    print('ok')
    int16 = 2**19
    j = 0
    for i in range(len(paths)):
        print(j)
        print(j+SliceNum[i])
        volume2read.data[j*int16: (j+SliceNum[i]) *int16] = open(paths[i]).read()
        j = j+SliceNum[i]
    return volume2read

def ReadVolume(SliceDim,SliceNum, path):
    # Read int16 as specified in README of vessel12 database
    z,x,y = SliceNum, SliceDim[0], SliceDim[1]
    volume2read = np.empty((z,x,y), np.int16)
    int16 = 2**19
    volume2read.data[0: (SliceNum) *int16] = open(path).read()
    return volume2read

def MAP_TO_255_float(rI):
    I=rI.astype('float')
    HIGH = np.max(I)
    LOW  = np.min(I)
    I255 = 255.0*((I-LOW)/(HIGH-LOW))
    return I255

def MAP_TO_255_float_32(rI):
    I=rI.astype('float32')
    HIGH = np.max(I)
    LOW  = np.min(I)
    I255 = 255.0*((I-LOW)/(HIGH-LOW))
    return I255

def NgNormalization2(Pin,g=10.0):
    """
    variance and mean normalization
     parameter g=10 set by Ng and Coates 2011a
     g parameter is scale-dependent and assumes
     each pixel intensity remains betwwen 0 and 255.
    """
    Pmean = np.mean(Pin,axis=1,keepdims=True)    
    Pstd  = np.sqrt(np.var(Pin,axis=1,keepdims=True)+g ) # g = 10 for images of brightness 0...255 
    O = (Pin - Pmean) / Pstd
    return O
    
def MeanSubtraction(Pin):
    """
    Mean subtraction patch wise
    Pin - input patches
    Po  - output patches
    """
    Pmean = np.mean(Pin,axis=1,keepdims=True)    
    Po = Pin - Pmean
    return Po

def Visualize_D(Dict):
    # Visualize atoms in the dictionary D
    # uses matplotlib
    for j, comp in enumerate(D[:np.shape(D)[0]]):
        plt.subplot(4, 8, j + 1)
        plt.imshow(comp.reshape(Para.patch_size), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    pass

def Visualize_Annotations():
    # Just a default one
    v_1=Para.anno
    V = TS[0].copy()
    V[v_1[v_1[:,2]==155][:,0],v_1[v_1[:,2]==155][:,1]] = 300
    pg.image(V)
    pass

def ExtractAnnotations(Stack, A, Coor):
    X = []
    y = []
    for i in range(len(Coor)):
        # cooridnates for slice number Coor[i]
        X.append(Stack[i][:,A[A[:,2]==Coor[i]][:,0],A[A[:,2]==Coor[i]][:,1]])
        X[i]=X[i].T
        # targets
        y.append(A[A[:,2]==Coor[i]][:,3])
    rX = np.concatenate((X[:]),axis=0)
    ry = np.concatenate((y[:]),axis=0)
    return (rX,ry)

def zca_whitening(inputs,epsilon=0.00001):
    # e is an epsilon, a whitening constat
    # it protects from dividing by zero if S is extremly small
    # usually e~10^-5
    
    #Correlation matrix
    print('Computing Correlation matrix')
    start_time = time.clock()
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1]
    print(start_time - time.clock())
    #Singular Value Decomposition
    print('Computing Singular Value Decomposition')
    start_time = time.clock()
    U,S,V = np.linalg.svd(sigma)
    print(start_time - time.clock())
    #ZCA Whitening matrix
    print('Computing ZCA')
    start_time = time.clock()
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)
    print(start_time - time.clock())
    #Data whitening
    return np.dot(ZCAMatrix, inputs)

def ExtractCube_3d(volume, point, cubeShape):
    # z,y,x because the annotations are given
    # for transposed volume 
    x,y,z = point
    shift_x = cubeShape[0]/2
    shift_y = cubeShape[1]/2
    shift_z = cubeShape[2]/2
    Cube = volume[x-shift_x-1:x+shift_x,
                  y-shift_y-1:y+shift_y,
                  z-shift_z-1:z+shift_z]
    return Cube
    
#TBD: check the annotation
def RearangePoinAnnotation(point,PaddZAxis=0,PaddYAxis=0,PaddXAxis=0):
    """
    z,y,x = point
    x = x ^ y
    y = y ^ x
    x = x ^ y
    """
    point[:,0] = point[:,0] ^ point[:,2]
    point[:,2] = point[:,2] ^ point[:,0]
    point[:,0] = point[:,0] ^ point[:,2]
    if PaddZAxis!=0:
        point[:,0]+=PaddZAxis
    if PaddYAxis!=0:
        point[:,1]+=PaddYAxis
    if PaddXAxis!=0:
        point[:,2]+=PaddXAxis
    return(point)

def CenterPoint(volume):
    x,y,z=np.shape(volume)
    return volume[x/2,y/2,z/2]