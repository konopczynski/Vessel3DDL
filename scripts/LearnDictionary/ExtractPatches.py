# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:56:11 2016

@author: konopczynski

This script extracts random patches at many scales from provided data.
At scale 0 it extracts only the patches from the masked region.
At higher scales the extracted number of patches is lower. 
"""

import numpy as np
import pickle
import sys
sys.path.append('../')
import config
sys.path.append('../utils')
import patches_3d as p
import pyramids_3d as pyr
import HelpFunctions as HLP


def extract_patches(param, numofpatches=100000):
    # Extract patches from each volume and scale
    patches = []
    raw_volume, normalized_patches, mask = 0, 0, 0  # init

    for v_index in range(len(param.paths2volumes_unannotated)):
        print('volume: ' + str(v_index))
        # prepare volume: apply mask and rescale
        if param.paths2volumes_unannotated[v_index][-3:] == 'raw':
            print('raw')
            mask = np.empty((param.patchSliceNum[v_index], param.sliceDim[0], param.sliceDim[1]), np.bool_)
            mask.data[:] = open(param.paths2masks_unannotated[v_index]).read()
            raw_volume = HLP.ReadVolume(param.sliceDim, param.patchSliceNum[v_index],
                                        param.paths2volumes_unannotated[v_index])
        elif param.paths2volumes_unannotated[v_index][-3:] == 'pkl':
            print('pkl')
            path2input_volume = open(param.paths2volumes_unannotated[v_index], 'rb')
            path2input_mask   = open(param.paths2masks_unannotated[v_index], 'rb')
            raw_volume = pickle.load(path2input_volume)
            mask = pickle.load(path2input_mask)
            path2input_volume.close()
            path2input_mask.close()
            mask = mask.astype(bool)  # use this mask to sample from the masked regions at scale 0
        # DON'T MAP TO 255 - JUST MAKE IT FLOAT32 #
        volume = raw_volume.astype('float32')  # HLP.MAP_TO_255_float_32(raw_volume)
        # APPLY SCALES
        volume = tuple(pyr.pyramid_gaussian_3d(volume, downscale=2, max_layer=param.patchScale))
        for s_index in range(param.patchScale):
            patch_size = param.patch_size
            print("     scale:"+str(s_index))
            if s_index == 0:
                data = p.extract_patches_3d_fromMask(volume[s_index], mask, patch_size,
                                                     max_patches=numofpatches, random_state=2)
            else:
                data = p.extract_patches_3d(volume[s_index], patch_size,
                                            max_patches=numofpatches/(8*s_index), random_state=2)
            data = data.reshape(data.shape[0], -1)
            normalized_patches = HLP.MeanSubtraction(data)
            patches.append(normalized_patches)

    # concatenate patches together
    for i in range(len(patches)-1):
        normalized_patches = np.concatenate((normalized_patches, patches[i]), axis=0).copy()
    print ("Number of patches: "+str(len(normalized_patches)))
    return normalized_patches


def serialize_patches(patches, file_name, path2save):
    f2save = path2save+file_name+".npy"
    output = open(f2save, 'wb')
    np.save(output, patches, allow_pickle=True, fix_imports=True)
    output.close()
    print "saved to: "+f2save
    return None


def main():
    param = config.read_parameters()
    normalized_patches = extract_patches(param, param.number_of_patches)  # 100000
    serialize_patches(normalized_patches, param.fileWithPatches, param.path2patches)
    print('done')

if __name__ == '__main__':
    main()
