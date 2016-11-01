# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:56:11 2016

@author: konop

This script extracts random patches at many scales from provided data.
At scale 0 it extracts only the patches from the masked region.
At higher scales the extracted number of patches is lower. 
"""

import numpy as np
import pickle
import sys
import os

sys.path.append('../')

import patches_3d  as p
import pyramids_3d as pyr
import HelpFunctions as HLP
import config as C


def ExtractPatches(Param,numOfPatches=100000):
    # Extract patches from each volume and scale
    patches=[]
    for v_indx in range(len(Param.patchVpaths)):
        print('volume: ' + str(v_indx))
        # prepare volume: apply mask and rescale
        if   Param.patchVpaths[v_indx][-3:] == 'raw':
            print('raw')
            mask = np.empty((Param.patchSliceNum[v_indx],Param.SliceDim[0],Param.SliceDim[1]), np.bool_)
            mask.data[:] = open(Param.patchMpaths[v_indx]).read()
            rV = HLP.ReadVolume(Param.SliceDim,Param.patchSliceNum[v_indx], Param.patchVpaths[v_indx])
        elif Param.patchVpaths[v_indx][-3:] == 'pkl':
            print('pkl')
            inputFile_V = open(Param.patchVpaths[v_indx], 'rb')
            inputFile_M = open(Param.patchMpaths[v_indx], 'rb')
            rV = pickle.load(inputFile_V)
            mask = pickle.load(inputFile_M)
            inputFile_V.close()
            inputFile_M.close()
            mask=mask.astype(bool) # we will use this mask to sample from the masked regions at scale 0
        # DONT MAP TO 255 - JUST MAKE IT FLOAT32 #
        volume = rV.astype('float32') #HLP.MAP_TO_255_float_32(rV)
        # APPLY SCALES
        volume = tuple(pyr.pyramid_gaussian_3d((volume), downscale=2, max_layer=Param.patchScale))
        for s_indx in range(Param.patchScale):
            x, y, z= np.shape(volume[s_indx])
            patch_size = Param.patch_size
            print("     scale:"+str(s_indx))
            if s_indx==0:
                data = p.extract_patches_3d_fromMask(volume[s_indx], mask, patch_size, max_patches=numOfPatches,random_state=2)
            else:
                data = p.extract_patches_3d(volume[s_indx], patch_size, max_patches=numOfPatches/(8*s_indx),random_state=2)        
            data = data.reshape(data.shape[0], -1)  
            normalizedPatches = HLP.MeanSubtraction(data)
            patches.append(normalizedPatches)

    # concatenate patches together
    for i in range(len(patches)-1):
        normalizedPatches = np.concatenate((normalizedPatches,patches[i]),axis=0).copy()
    print ("Number of patches: "+str(len(normalizedPatches)))
    return normalizedPatches

def SerializePatches(patches,file_name,path2save):
    path2save = "../../Data/Serialized/saved_patches/"
    f2save = path2save+file_name+".npy"
    output = open(f2save, 'wb')
    np.save(output, patches,allow_pickle=True, fix_imports=True)
    output.close()
    print "saved to: "+f2save
    return None


if __name__ == '__main__':
    P = C.ReadParameters()
    normalizedPatches = ExtractPatches(P,P.nP) # 100000
    SerializePatches(normalizedPatches,P.Fpatches,P.path2patches)
    print('done')
