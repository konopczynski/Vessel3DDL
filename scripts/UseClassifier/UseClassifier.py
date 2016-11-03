# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 08:30:17 2016

@author: konopczynski

This script is using prelearned dictionary and classifier.
It is hardcoded for VESSEL12 data.
"""

import pickle
import sys
from sklearn.externals import joblib
import numpy as np
from scipy import ndimage as nd
import scipy
import time
sys.path.append('../')
import config
sys.path.append('../utils')
import HelpFunctions as HLP
import pyramids_3d as pyr

def PreprocessVolume():
    #empty, no preprocessing has been applied so far
    return None

def LoadClassifier(path2clf,file_name):
    filepath=path2clf+file_name
    clf = joblib.load(filepath) 
    return clf

def LoadVolume(sliceDim,sliceNum,Vpath):
    V = HLP.ReadVolume(sliceDim,sliceNum, Vpath)
    return V

def LoadDictionary(Dpath,Dname):
    inputFile = open(Dpath+Dname,'rb')
    D = pickle.load(inputFile) # load the dictionary
    inputFile.close()
    return D

def ExtractPatch(volume, xdim, ydim, zdim):
    Cube = volume[xdim[0]:xdim[1],
                  ydim[0]:ydim[1],
                  zdim[0]:zdim[1]]
    return Cube

def ApplyAtoms(V,D,scale):
    out=[]
    for s in xrange(scale):
        if s!=0:
            print('scale='+str(s))
            V = pyr.pyramid_reduce_3d(V,downscale=2) # reduce the volume. e.g. from 512^3 to 256^3
        else: print('scale=0')
        for i in xrange(len(D)):
            print('s:'+str(s)+' i:'+str(i))
            conv = nd.convolve(V, D[i], mode='constant', cval=0.0)
            if s==0:
                out.append(conv)
            else:
                upscaled = pyr.pyramid_expand_3d(conv, upscale=2**s)
                out.append(upscaled)
    out=np.array(out)
    return out

def CutTheEdges():
    # TBD
    # Think how much should be cutted, and therefor how much the patches should overlap.
    # it depends on the atom width and number of scales
    return None

def Divide(L,n,w):
    # L - length of the array e.g. 512
    # n - size of patch to be cutted e.g. 128
    # w - size of the window e.g. 4
    # k - number of possible steps e.g. 4 in this case
    # R - rest e.g. 12 (R=k*w)
    k = ((L-n)/(n-w)) + 1
    R=k*w
    Lv=[] # left value
    Rv=[] #right value
    for i in xrange(k):
        Lv.append(i*(n-w))
        Rv.append(i*(n-w)+n+w)
    Lv.append(L-R)
    Rv.append(L)
    return Lv,Rv

def divideIntoBlocks(V,Lv,Rv):
    # works only for 3d V
    NV = []
    L = len(Lv)
    for i in xrange(L):
        for j in xrange(L):
            for k in xrange(L):
                xdim=[Lv[i],Rv[i]] 
                ydim=[Lv[j],Rv[j]]
                zdim=[Lv[k],Rv[k]]
                NV.append(ExtractPatch(V, xdim, ydim, zdim))
    return NV

def connectAllBlocks(DV,dim,w):
    number_of_patches = len(DV)
    L = int(scipy.special.cbrt(number_of_patches))
    Lsqr = L**2
    buff_x=[]
    buff_y=[]
    buff_z=[]
    for i in xrange(L):
        for j in xrange(L):
            index = i*Lsqr+j*L
            print(index)
            buff_x = DV[index]
            for k in xrange(L-1):                 
                index = i*Lsqr+j*L+k+1
                print(index)
                nextV=DV[index]
                buff_x=connectBlocks(buff_x,nextV,2,w)
            if j==0: buff_y= buff_x
            else:    buff_y= connectBlocks(buff_y, buff_x,1,w)
        if i==0: buff_z= buff_y
        else:    buff_z= connectBlocks(buff_z, buff_y,0,w)
    return buff_z

def connectBlocks(B1,B2,dim,w=0):
    # dim = 0 - x
    # dim = 1 - y
    # dim = 2 - z    
    s1 = np.shape(B1)
    s2 = np.shape(B2)
    if dim == 0:
        O = np.zeros((s1[0]+s2[0]-2*w,s1[1],s1[2]))
        O[0:s1[0]-w,0:s1[1],0:s1[2]]=B1[:s1[0]-w,:,:]
        O[s1[0]-w:,:,:]=B2[w:,:,:]
    elif dim == 1:
        O = np.zeros((s1[0],s1[1]+s2[1]-2*w,s1[2]))
        O[0:s1[0],0:s1[1]-w,0:s1[2]]=B1[:,:s1[1]-w,:]
        O[:,s1[1]-w:,:]=B2[:,w:,:]
    elif dim == 2:
        O = np.zeros((s1[0],s1[1],s1[2]+s2[2]-2*w))
        O[0:s1[0],0:s1[1],0:s1[2]-w]=B1[:,:,:s1[2]-w]
        O[:,:,s1[2]-w:]=B2[:,:,w:]
    return O

def Padd(V):
    ## works only for one type of volume
    ##TBD: make it more general
    shp = np.shape(V)
    if shp[0]!=512:
        PaddedVolume = np.lib.pad(V, np.array([[26,27],[0,0],[0,0]]), 'edge')
    return PaddedVolume

def SerializeOutput(Output,path2save,file_name):
    outputFile = open(path2save+file_name+".npy", 'wb')
    np.save(outputFile, Output,allow_pickle=True, fix_imports=True)
    outputFile.close()
    return None

def main():
    param = config.read_parameters()
    sliceNum = param.sliceNum[0]
    path2volume = param.path2examples+param.volumeToProcess
    scale = param.clfS

    inDict = param.dictionaryName+'.pkl'
    inClass= param.clf2use+'_'+param.dictionaryName

    C = LoadClassifier(param.path2classifier,inClass)
    V = LoadVolume(param.sliceDim,sliceNum,path2volume)
    V = Padd(V)
    D = LoadDictionary(param.path2dicts,inDict)

    print ('start')
    t0=time.time()
    window = 8
    Lv,Rv=Divide(512,128,window)
    NV = divideIntoBlocks(V,Lv,Rv)
    # Perform the operation on each block in the list NV
    # before division: P = block
    # P = ExtractPatch(V, xdim, ydim, zdim)
    # P = P.astype('float32')
    # Next, connect everything back.
    Out = []
    for i in xrange(len(NV)):
        P=NV[i]
        P=P.astype('float32')
        O=ApplyAtoms(P,D,scale)
        shp = np.shape(P)
        O = O.reshape(scale*param.numOfAtoms,shp[0]*shp[1]*shp[2])
        O = O.T
        y_pred = C.predict(O)
        Out.append(y_pred.reshape(shp[0],shp[1],shp[2]))
    ####
    O=connectAllBlocks(Out,(512,512,512),window)
    SerializeOutput(O,param.path2Output,param.dictionaryName+'_OutputVolume')
    print(time.time()-t0)
    print("Done")

if __name__ == '__main__':
    main()