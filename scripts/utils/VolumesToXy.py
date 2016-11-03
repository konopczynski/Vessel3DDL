# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:56:11 2016

@author: konopczynski
It's a handy set of functions to
apply 3D filters at many scales
and retrieving the X and y for the data

Hardcoded for the VESSEL12 data
"""

import pickle
import numpy as np
import sys
import time
from scipy import ndimage as nd
sys.path.append('../')
import pyramids_3d as pyr
import HelpFunctions as HLP


def apply_filters(param, dictionary):
    """
    this function needs to be rewritten
    right now it's a mess but works
    """
    XX = []  # feature vector
    yy = []  # labels
    paths2volumes, paths2masks = param.paths2volumes, param.paths2masks
    start_time = time.time()
    raw_volume = 0  # init
    # Extract cubes from each volume and scale
    for v_index in range(len(paths2volumes)):  # for each volume
        start_reading_volume_time = time.time()
        if paths2volumes[v_index][-3:] == 'raw':  # if the volume is in a raw format
            mask = np.empty(param.dim[v_index], np.bool_)
            mask.data[:] = open(paths2masks[v_index]).read()
            raw_volume = HLP.ReadVolume(param.sliceDim, param.sliceNum[v_index], param.paths2volumes[v_index])
        elif paths2volumes[v_index][-3:] == 'npy':  # else, if its in npy format
            inputFile_V = paths2volumes[v_index]  # open(paths2volumes[v_index], 'rb')
            inputFile_M = paths2masks[v_index]    # open(paths2masks[v_index], 'rb')
            raw_volume   = np.load(inputFile_V)   # pickle.load(inputFile_V)
            mask = np.load(inputFile_M)  # pickle.load(inputFile_M)
            mask = mask.astype(bool)
        # Don't apply masks
        volume = raw_volume.astype('float32')  # HLP.MAP_TO_255_float_32(raw_volume)
        
        # Rearrange Annotations for padding, and apply padding to the volume
        points = param.anno[v_index][:,0:3].copy()
        shp = np.shape(volume)    
        if v_index == 0:
            if shp[0] != 512:
                points = HLP.RearangePoinAnnotation(points, PaddZAxis=26)
                volume = np.lib.pad(volume, np.array([[26, 27], [0, 0], [0, 0]]), 'edge')
            elif shp[1] != 512:
                print('size1:'+str(np.shape(volume)))
                points = HLP.RearangePoinAnnotation(points, PaddYAxis=26)
                volume = np.lib.pad(volume, np.array([[0, 0], [26, 27], [0, 0]]), 'edge')
            elif shp[2] != 512:
                points = HLP.RearangePoinAnnotation(points, PaddXAxis=26)
                volume = np.lib.pad(volume, np.array([[0, 0], [0, 0], [26, 27]]), 'edge')
        if v_index == 1:
            if shp[0] != 512:
                points = HLP.RearangePoinAnnotation(points, PaddZAxis=32)
                volume = np.lib.pad(volume, np.array([[32, 32], [0, 0], [0, 0]]), 'edge')
            elif shp[1] != 512:
                points = HLP.RearangePoinAnnotation(points, PaddYAxis=32)
                volume = np.lib.pad(volume, np.array([[0, 0], [32, 32], [0, 0]]), 'edge')
            elif shp[2] != 512:
                points = HLP.RearangePoinAnnotation(points, PaddXAxis=32)
                volume = np.lib.pad(volume, np.array([[0, 0], [0, 0], [32, 32]]), 'edge')
        if v_index == 2:
            if shp[0] != 512:
                points = HLP.RearangePoinAnnotation(points, PaddZAxis=47)
                volume = np.lib.pad(volume, np.array([[47, 47], [0, 0], [0, 0]]), 'edge')
            elif shp[1] != 512:
                points = HLP.RearangePoinAnnotation(points, PaddYAxis=47)
                volume = np.lib.pad(volume, np.array([[0, 0], [47, 47], [0, 0]]), 'edge')
            elif shp[2] != 512:
                points = HLP.RearangePoinAnnotation(points, PaddXAxis=47)
                volume = np.lib.pad(volume, np.array([[0, 0], [0, 0], [47, 47]]), 'edge')
        
        print ('Data reading time:' + str(time.time()-start_reading_volume_time))
        print('np.shape(volume): '  + str(np.shape(volume)))
        for s_index in range(param.fMapS):  # at each scale
            print("Start of Volume: "+str(v_index)+" Scale:"+str(s_index))
            if s_index == 0:  # at scale 0 apply conv only at the annotated points
                # volume=volume
                # apply convolutions at small cubes centered at annotated points
                # Extract annotated cubes
                cubes = np.empty((len(param.anno[v_index]), param.patch_size[0],
                                  param.patch_size[1], param.patch_size[2]))
                for i in range(len(param.anno[v_index])):
                    cubes[i] = HLP.ExtractCube_3d(volume, points[i], param.patch_size)
                # Apply convolution and save in X
                X = np.empty((len(cubes), len(dictionary)))
                for i in range(len(cubes)):
                    for j in range(len(dictionary)):
                        conv = nd.convolve(cubes[i], dictionary[j], mode='constant', cval=0.0)
                        X[i, j] = HLP.CenterPoint(conv)
                y = param.anno[v_index][:, 3:4]
                XX.append(X)
                yy.append(y)
            else:
                # Gaussian Scaling
                start_pyramid_time=time.time()
                volume = pyr.pyramid_reduce_3d(volume, downscale=2)  # reduce the volume. e.g. from 512^3 to 256^3
                print ('pyramid reduce time:'+str(time.time()-start_pyramid_time))
                # Extract annotated cubes
                cubes = np.empty((len(param.anno[v_index]), param.patch_size[0],
                                  param.patch_size[1], param.patch_size[2]))
                X = np.empty((len(cubes), len(dictionary)))
                # for each filter
                for j in range(len(dictionary)):  # for each atom j in the dictionary
                    print("filter:"+str(j)+" Volume:"+str(v_index)+" scale:"+str(s_index))
                    start_filter_time = time.time()
                    conv = nd.convolve(volume, dictionary[j], mode='constant', cval=0.0)  # convolve volume with atom j
                    # Pyramids are almost 2x faster but use 2x more memory
                    # Zoom use less memory but is slower
                    upscaled = pyr.pyramid_expand_3d(conv, upscale=2**s_index)  # upscale the volume to the initial size
                    # upscaled = nd.interpolation.zoom(conv,zoom=2**s_index)
                    for i in range(len(param.anno[v_index])):  # extract cubes from the upscaled volume
                        cubes[i] = HLP.ExtractCube_3d(upscaled, points[i], param.patch_size)
                    for i in range(len(cubes)):  # save the middle voxels from i extracted cubes to the X[i,j]
                        X[i, j] = HLP.CenterPoint(cubes[i])
                    print ('upscaling+convolution time:'+str(time.time()-start_filter_time))
                y = param.anno[v_index][:, 3:4]
                XX.append(X)
                yy.append(y)
    end_time = time.time()
    print ("Total time:")
    print str(end_time-start_time)
    return XX, yy


def serialize_xy(path2save, XX, yy, preffix="", suffix=""):
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


def read_xy(path2xy, preffix="", suffix=""):
    Xfile_name = preffix+'X'+suffix+'.pkl'
    yfile_name = preffix+'y'+suffix+'.pkl'
    input_file = open(path2xy+Xfile_name, 'rb')
    X = pickle.load(input_file)
    input_file.close()
    input_file = open(path2xy+yfile_name, 'rb')
    y = pickle.load(input_file)
    input_file.close()
    return X, y
