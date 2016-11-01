# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:56:11 2016

@author: konop
It's a handy set of functions to
apply 3D filters at many scales
and retreaving the X and y for the data

Hardcoded for the VESSEL12 data
"""

import pickle
import numpy as np
import sys
import os
import time
from scipy import ndimage as nd
sys.path.append('../')
import pyramids_3d as pyr
import HelpFunctions as HLP


def ApplyFilters(Param,D):
    """
    this function needs to be rewritten
    right now it's a mess but works
    """
    XX=[] # feature vector
    yy=[] # labels
    AVpaths,AMpaths,Apaths = Param.AVpaths,Param.AMpaths,Param.Apaths
    start_time = time.time()
    # Extract cubes from each volume and scale
    for v_indx in range(len(AVpaths)): # for each volume
        start_reading_volume_time=time.time()
        if   AVpaths[v_indx][-3:] == 'raw': # if the volume is in a raw format
            mask = np.empty(Param.dim[v_indx], np.bool_)
            mask.data[:] = open(AMpaths[v_indx]).read()
            rV = HLP.ReadVolume(Param.sliceDim,Param.sliceNum[v_indx], Param.AVpaths[v_indx])
        elif AVpaths[v_indx][-3:] == 'npy': # else, if its in npy format
            inputFile_V = AVpaths[v_indx] #open(AVpaths[v_indx], 'rb')
            inputFile_M = AMpaths[v_indx] #open(AMpaths[v_indx], 'rb')
            rV   = np.load(inputFile_V) #pickle.load(inputFile_V)
            mask = np.load(inputFile_M) #pickle.load(inputFile_M)
            mask = mask.astype(bool)
        # Dont apply masks
        volume = rV.astype('float32') #HLP.MAP_TO_255_float_32(rV)
        
        # Rearange Annotations for padding, and apply padding to the volume
        points=Param.anno[v_indx][:,0:3].copy()
        shp = np.shape(volume)    
        if v_indx == 0:
            if shp[0]!=512:
                points = HLP.RearangePoinAnnotation(points,PaddZAxis=26)
                volume = np.lib.pad(volume, np.array([[26,27],[0,0],[0,0]]), 'edge')
            elif shp[1]!=512:
                print('size1:'+str(np.shape(volume)))
                points = HLP.RearangePoinAnnotation(points,PaddYAxis=26)
                volume = np.lib.pad(volume, np.array([[0,0],[26,27],[0,0]]), 'edge')
            elif shp[2]!=512:
                points = HLP.RearangePoinAnnotation(points,PaddXAxis=26)
                volume = np.lib.pad(volume, np.array([[0,0],[0,0],[26,27]]), 'edge')
        if v_indx == 1:
            if shp[0]!=512:
                points = HLP.RearangePoinAnnotation(points,PaddZAxis=32)
                volume = np.lib.pad(volume, np.array([[32,32],[0,0],[0,0]]), 'edge')
            elif shp[1]!=512:
                points = HLP.RearangePoinAnnotation(points,PaddYAxis=32)
                volume = np.lib.pad(volume, np.array([[0,0],[32,32],[0,0]]), 'edge')
            elif shp[2]!=512:
                points = HLP.RearangePoinAnnotation(points,PaddXAxis=32)
                volume = np.lib.pad(volume, np.array([[0,0],[0,0],[32,32]]), 'edge')
        if v_indx == 2:
            if shp[0]!=512:
                points = HLP.RearangePoinAnnotation(points,PaddZAxis=47)
                volume = np.lib.pad(volume, np.array([[47,47],[0,0],[0,0]]), 'edge')
            elif shp[1]!=512:
                points = HLP.RearangePoinAnnotation(points,PaddYAxis=47)
                volume = np.lib.pad(volume, np.array([[0,0],[47,47],[0,0]]), 'edge')
            elif shp[2]!=512:
                points = HLP.RearangePoinAnnotation(points,PaddXAxis=47)
                volume = np.lib.pad(volume, np.array([[0,0],[0,0],[47,47]]), 'edge')            
        
        print ('Data reading time:'+ str(time.time()-start_reading_volume_time))
        print('np.shape(volume): ' + str(np.shape(volume)))
        for s_indx in range(Param.fMapS): # at each scale
            print("Start of Volume: "+str(v_indx)+" Scale:"+str(s_indx))
            if s_indx==0: # at scale 0 apply conv only at the annotated points
                # volume=volume
                # apply convolutions at small cubes centered at annotated points
                x, y, z= np.shape(volume)
                # Extract annotated cubes
                cubes=np.empty((len(Param.anno[v_indx]),Param.patch_size[0],
                                                        Param.patch_size[1],
                                                        Param.patch_size[2]))
                for i in range(len(Param.anno[v_indx])):
                    cubes[i]=HLP.ExtractCube_3d(volume,
                                                points[i],
                                                Param.patch_size)
                # Apply convolution and save in X
                X=np.empty((len(cubes),len(D)))
                for i in range(len(cubes)):
                    for j in range(len(D)):
                        conv = nd.convolve(cubes[i], D[j], mode='constant', cval=0.0)
                        X[i,j] = HLP.CenterPoint(conv)
                y = Param.anno[v_indx][:,3:4]
                XX.append(X)
                yy.append(y)
            else:
                # Gaussian Scaling
                start_pyramid_time=time.time()
                volume = pyr.pyramid_reduce_3d(volume,downscale=2) # reduce the volume. e.g. from 512^3 to 256^3
                print ('pyramid reduce time:'+str(time.time()-start_pyramid_time))
                x, y, z = np.shape(volume)
                # Extract annotated cubes
                cubes=np.empty((len(Param.anno[v_indx]),Param.patch_size[0],
                                                        Param.patch_size[1],
                                                        Param.patch_size[2]))
                X=np.empty((len(cubes),len(D)))
                #for each filter
                for j in range(len(D)): # for each atom j in the dictionary
                    print("filter:"+str(j)+" Volume:"+str(v_indx)+" scale:"+str(s_indx))
                    start_filter_time = time.time()
                    conv = nd.convolve(volume, D[j], mode='constant', cval=0.0) # convolove volume with the atom j
                    # Pyramids are almost 2x faster but use 2x more memory
                    # Zoom use less memory but is slower
                    upscaled = pyr.pyramid_expand_3d(conv, upscale=2**s_indx) # upscale the volume to the initial size
                    #upscaled = nd.interpolation.zoom(conv,zoom=2**s_indx) 
                    for i in range(len(Param.anno[v_indx])): # extract cubes from the upscaled volume
                        cubes[i]=HLP.ExtractCube_3d(upscaled,
                                                    points[i],
                                                    Param.patch_size)
                    for i in range(len(cubes)): # save the middle voxels from i extracted cubes to the X[i,j]
                        X[i,j] = HLP.CenterPoint(cubes[i])
                    print ('upscaling+convolution time:'+str(time.time()-start_filter_time))
                y = Param.anno[v_indx][:,3:4]
                XX.append(X)
                yy.append(y)
    end_time = time.time()
    print ("Total time:")
    print str(end_time-start_time)
    return XX,yy

def Serialize_Xy(path2save,XX,yy,preffix="",suffix=""):
    X_to_save = path2save+preffix+"_X"+suffix+".pkl"
    output = open(X_to_save, 'wb')
    pickle.dump(XX, output)
    output.close()
    print ("Saved: "+X_to_save)
    y_to_save = path2save+preffix+"_y"+suffix+".pkl"
    output = open(y_to_save, 'wb')
    pickle.dump(yy, output)
    output.close()
    print("Saved: "+y_to_save)
    return None

def Read_xy(path2xy,preffix="" ,suffix=""):
    XfileName = preffix+'X'+suffix+'.pkl'
    yfileName = preffix+'y'+suffix+'.pkl'
    inputFile = open(path2xy+XfileName,'rb')
    X = pickle.load(inputFile)
    inputFile.close()
    inputFile = open(path2xy+yfileName,'rb')
    y = pickle.load(inputFile)
    inputFile.close()
    return X, y
