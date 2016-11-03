# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:12:11 2016

@author: konop

Support for extracting 3d patches similar to extract patches 2d from scikit-learn
"""


import numbers
import numpy as np
from sklearn.utils import check_array, check_random_state
from numpy.lib.stride_tricks import as_strided
from itertools import product

"""
Arrays used for testing

a = np.array([[[ 1, 2, 3, 4],[ 5, 6, 7, 8],[ 9,10,11,12],[13,14,15,16]],
              [[17,18,19,20],[21,22,23,24],[25,26,27,28],[29,30,31,32]],
              [[33,34,35,36],[37,38,39,40],[41,42,43,44],[45,46,47,48]],
              [[49,50,51,52],[53,54,55,56],[57,58,59,60],[61,62,63,64]]])

i = np.array([[ 1, 2, 3, 4],
              [ 5, 6, 7, 8],
              [ 9,10,11,12],
              [13,14,15,16]])
              
i_modal = np.array([[[ 1, 2, 3],[ 4, 5, 6],[ 7, 8, 9],[10,11,12]],
                    [[13,14,15],[16,17,18],[19,20,21],[22,23,24]],
                    [[25,26,27],[28,29,30],[31,32,33],[34,35,36]],
                    [[37,38,39],[40,41,42],[43,44,45],[46,46,48]]])
             
a_modal = np.array([[[[  1,  2,  3],[  4,  5,  6],[  7,  8,  9],[ 10, 11, 12]],
                     [[ 13, 14, 15],[ 16, 17, 18],[ 19, 20, 21],[ 22, 23, 24]],
                     [[ 25, 26, 27],[ 28, 29, 30],[ 31, 32, 33],[ 34, 35, 36]],
                     [[ 37, 38, 39],[ 40, 41, 42],[ 43, 44, 45],[ 46, 47, 48]]],
                    [[[ 49, 50, 51],[ 52, 53, 54],[ 55, 56, 57],[ 58, 59, 60]],
                     [[ 61, 62, 63],[ 64, 65, 66],[ 67, 68, 69],[ 70, 71, 72]],
                     [[ 73, 74, 75],[ 76, 77, 78],[ 79, 80, 81],[ 82, 83, 84]],
                     [[ 85, 86, 87],[ 88, 89, 90],[ 91, 92, 93],[ 94, 95, 96]]],
                    [[[ 97, 98, 99],[100,101,102],[103,104,105],[106,107,108]],
                     [[109,110,111],[112,113,114],[115,116,117],[118,119,120]],
                     [[121,122,123],[124,125,126],[127,128,129],[130,131,132]],
                     [[133,134,135],[136,137,138],[139,140,141],[142,143,144]]],
                    [[[145,146,147],[148,149,150],[151,152,153],[154,155,156]],
                     [[157,158,159],[160,161,162],[163,164,165],[166,167,168]],
                     [[169,170,171],[172,173,174],[175,176,177],[178,179,180]],
                     [[181,182,183],[184,185,186],[187,188,189],[190,191,192]]]])
                     
i_modal = np.array([[[ 1, 2],[ 4, 5],[ 7, 8],[10,11]],
                    [[13,14],[16,17],[19,20],[22,23]],
                    [[25,26],[28,29],[31,32],[34,35]],
                    [[37,38],[40,41],[43,44],[46,46]]])
             
a_modal = np.array([[[[  1,  2],[  4,  5],[  7,  8],[ 10, 11]],
                     [[ 13, 14],[ 16, 17],[ 19, 20],[ 22, 23]],
                     [[ 25, 26],[ 28, 29],[ 31, 32],[ 34, 35]],
                     [[ 37, 38],[ 40, 41],[ 43, 44],[ 46, 47]]],
                    [[[ 49, 50],[ 52, 53],[ 55, 56],[ 58, 59]],
                     [[ 61, 62],[ 64, 65],[ 67, 68],[ 70, 71]],
                     [[ 73, 74],[ 76, 77],[ 79, 80],[ 82, 83]],
                     [[ 85, 86],[ 88, 89],[ 91, 92],[ 94, 95]]],
                    [[[ 97, 98],[100,101],[103,104],[106,107]],
                     [[109,110],[112,113],[115,116],[118,119]],
                     [[121,122],[124,125],[127,128],[130,131]],
                     [[133,134],[136,137],[139,140],[142,143]]],
                    [[[145,146],[148,149],[151,152],[154,155]],
                     [[157,158],[160,161],[163,164],[166,167]],
                     [[169,170],[172,173],[175,176],[178,179]],
                     [[181,182],[184,185],[187,188],[190,191]]]])
"""

def _compute_n_patches(i_h, i_w, p_h, p_w, max_patches=None):
    """Compute the number of patches that will be extracted in an image.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    i_h : int
        The image height
    i_w : int
        The image with
    p_h : int
        The height of a patch
    p_w : int
        The width of a patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    """
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    all_patches = n_h * n_w

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Real))
                and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches

def _compute_n_patches_3d(i_x, i_y, i_z, p_x, p_y, p_z, max_patches=None):
    """Compute the number of patches that will be extracted in a volume.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    i_x : int
        The number of voxels in x dimension
    i_y : int
        The number of voxels in y dimension
    i_z : int
        The number of voxels in z dimension
    p_x : int
        The number of voxels in x dimension of a patch
    p_y : int
        The number of voxels in y dimension of a patch
    p_z : int
        The number of voxels in z dimension of a patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    """
    n_x = i_x - p_x + 1
    n_y = i_y - p_y + 1
    n_z = i_z - p_z + 1
    all_patches = n_x * n_y * n_z

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Real))
                and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches

def extract_patches(arr, patch_shape=8, extraction_step=1):
    """Extracts patches of any n-dimensional array in place using strides.
    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted
    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.
    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = [slice(None, None, st) for st in extraction_step]
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches


def extract_patches_2d(image, patch_size, max_patches=None, random_state=None):
    """Reshape a 2D image into a collection of patches
    The resulting patches are allocated in a dedicated array.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    image : array, shape = (image_height, image_width) or
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.
    patch_size : tuple of ints (patch_height, patch_width)
        the dimensions of one patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    random_state : int or RandomState
        Pseudo number generator state used for random sampling to use if
        `max_patches` is not None.
    Returns
    -------
    patches : array, shape = (n_patches, patch_height, patch_width) or
         (n_patches, patch_height, patch_width, n_channels)
         The collection of patches extracted from the image, where `n_patches`
         is either `max_patches` or the total number of patches that can be
         extracted.
    Examples
    --------
    >>> from sklearn.feature_extraction import image
    >>> one_image = np.arange(16).reshape((4, 4))
    >>> one_image
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> patches = image.extract_patches_2d(one_image, (2, 2))
    >>> print(patches.shape)
    (9, 2, 2)
    >>> patches[0]
    array([[0, 1],
           [4, 5]])
    >>> patches[1]
    array([[1, 2],
           [5, 6]])
    >>> patches[8]
    array([[10, 11],
           [14, 15]])
    """
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    extracted_patches = extract_patches(image, patch_shape=(p_h, p_w, n_colors), extraction_step=1)

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches

def extract_patches_3d(volume, patch_size, max_patches=None, random_state=None):
    """Reshape a 3D volume into a collection of patches
    The resulting patches are allocated in a dedicated array.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    volume : array, shape = (volume_x, volume_y, volume_z)
        No channels are allowed
    patch_size : tuple of ints (patch_x, patch_y, patch_z)
        the dimensions of one patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    random_state : int or RandomState
        Pseudo number generator state used for random sampling to use if
        `max_patches` is not None.
    Returns
    -------
    patches : array, shape = (n_patches, patch_x, patch_y, patch_z)
         The collection of patches extracted from the volume, where `n_patches`
         is either `max_patches` or the total number of patches that can be
         extracted.
    Examples
    --------
    TBD
    """
    v_x, v_y, v_z = volume.shape[:3]
    p_x, p_y, p_z = patch_size

    if p_x > v_x:
        raise ValueError("Height of the patch should be less than the height"
                         " of the volume.")

    if p_y > v_y:
        raise ValueError("Width of the patch should be less than the width"
                         " of the volume.")

    if p_z > v_z:
        raise ValueError("z of the patch should be less than the z"
                         " of the volume.")
                         
    volume = check_array(volume, allow_nd=True)
    volume = volume.reshape((v_x, v_y, v_z, -1))
    n_colors = volume.shape[-1]

    extracted_patches = extract_patches(volume, patch_shape=(p_x, p_y, p_z, n_colors), extraction_step=1)

    n_patches = _compute_n_patches_3d(v_x, v_y, v_z, p_x, p_y, p_z, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(v_x - p_x + 1, size=n_patches)
        j_s = rng.randint(v_y - p_y + 1, size=n_patches)
        k_s = rng.randint(v_z - p_z + 1, size=n_patches)
        
        patches = extracted_patches[i_s, j_s, k_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_x, p_y, p_z, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_x, p_y, p_z))
    else:
        return patches

def extract_patches_3d_fromMask(volume, mask, patch_size, max_patches=None, random_state=None):
    """Reshape a 3D volume into a collection of patches
    The resulting patches are allocated in a dedicated array.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    volume : array, shape = (volume_x, volume_y, volume_z)
        No channels are allowed
    patch_size : tuple of ints (patch_x, patch_y, patch_z)
        the dimensions of one patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    random_state : int or RandomState
        Pseudo number generator state used for random sampling to use if
        `max_patches` is not None.
    Returns
    -------
    patches : array, shape = (n_patches, patch_x, patch_y, patch_z)
         The collection of patches extracted from the volume, where `n_patches`
         is either `max_patches` or the total number of patches that can be
         extracted.
    Examples
    --------
    TBD
    """
    v_x, v_y, v_z = volume.shape[:3]
    p_x, p_y, p_z = patch_size

    if p_x > v_x:
        raise ValueError("Height of the patch should be less than the height"
                         " of the volume.")

    if p_y > v_y:
        raise ValueError("Width of the patch should be less than the width"
                         " of the volume.")

    if p_z > v_z:
        raise ValueError("z of the patch should be less than the z"
                         " of the volume.")
                         
    volume = check_array(volume, allow_nd=True)
    volume = volume.reshape((v_x, v_y, v_z, -1))
    n_colors = volume.shape[-1]

    extracted_patches = extract_patches(volume, patch_shape=(p_x, p_y, p_z, n_colors), extraction_step=1)

    n_patches = _compute_n_patches_3d(v_x, v_y, v_z, p_x, p_y, p_z, max_patches)
    # check the indexes where mask is True
    M=np.array(np.where(mask[p_x/2:v_x-p_x/2,
                             p_y/2:v_y-p_y/2,
                             p_z/2:v_z-p_z/2]==True)).T
    if max_patches:
        rng = check_random_state(random_state)
        indx = rng.randint(len(M), size=n_patches)
        i_s = M[indx][:,0]
        j_s = M[indx][:,1]
        k_s = M[indx][:,2]        
        patches = extracted_patches[i_s, j_s, k_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_x, p_y, p_z, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_x, p_y, p_z))
    else:
        return patches

def reconstruct_from_patches_2d(patches, image_size):
    """Reconstruct the image from all of its patches.
    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    patches : array, shape = (n_patches, patch_height, patch_width) or
        (n_patches, patch_height, patch_width, n_channels)
        The complete set of patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.
    image_size : tuple of ints (image_height, image_width) or
        (image_height, image_width, n_channels)
        the size of the image that will be reconstructed
    Returns
    -------
    image : array, shape = image_size
        the reconstructed image
    """
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i:i + p_h, j:j + p_w] += p

    for i in range(i_h):
        for j in range(i_w):
            # divide by the amount of overlap
            # XXX: is this the most efficient way? memory-wise yes, cpu wise?
            img[i, j] /= float(min(i + 1, p_h, i_h - i) *
                               min(j + 1, p_w, i_w - j))
    return img

def reconstruct_from_patches_3d(patches, volume_size):
    """Reconstruct the volume from all of its patches.
    Patches are assumed to overlap and the volume is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    patches : array, shape = (n_patches, patch_x, patch_y, patch_z)
    volume_size : tuple of ints (volume_x, volume_y, volume_z)
        the size of the image that will be reconstructed
    Returns
    -------
    volume : array, shape = volume_size
        the reconstructed volume
    """
    v_x, v_y, v_z = volume_size[:3]
    p_x, p_y, p_z = patches.shape[1:4]
    vol = np.zeros(volume_size)
    # compute the dimensions of the patches array
    n_x = v_x - p_x + 1
    n_y = v_y - p_y + 1
    n_z = v_z - p_z + 1
    for p, (i, j, k) in zip(patches, product(range(n_x), range(n_y), range(n_z))):
        vol[i:i + p_x, j:j + p_y, k:k + p_z] += p

    for i in range(v_x):
        for j in range(v_y):
            for k in range(v_z):
                # divide by the amount of overlap
                # XXX: is this the most efficient way? memory-wise yes, cpu wise?
                vol[i, j, k] /= float(min(i + 1, p_x, v_x - i) *
                                      min(j + 1, p_y, v_y - j) *
                                      min(k + 1, p_z, v_z - k))
    return vol

def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True, preserve_range=False):
    """Resize image to match a certain size.
    Performs interpolation to up-size or down-size images. For down-sampling
    N-dimensional images by applying the arithmetic sum or mean, see
    `skimage.measure.local_sum` and `skimage.transform.downscale_local_mean`,
    respectively.
    Parameters
    ----------
    image : ndarray
        Input image.
    output_shape : tuple or ndarray
        Size of the generated output image `(rows, cols[, dim])`. If `dim` is
        not provided, the number of channels is preserved. In case the number
        of input channels does not equal the number of output channels a
        3-dimensional interpolation is applied.
    Returns
    -------
    resized : ndarray
        Resized version of the input.
    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to
        be in the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
    Note
    ----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].
    Examples
    --------
    >>> from skimage import data
    >>> from skimage.transform import resize
    >>> image = data.camera()
    >>> resize(image, (100, 100)).shape
    (100, 100)
    """

    rows, cols = output_shape[0], output_shape[1]
    orig_rows, orig_cols = image.shape[0], image.shape[1]

    row_scale = float(orig_rows) / rows
    col_scale = float(orig_cols) / cols

    # 3-dimensional interpolation
    if len(output_shape) == 3 and (image.ndim == 2
                                   or output_shape[2] != image.shape[2]):
        ndi_mode = _to_ndimage_mode(mode)
        dim = output_shape[2]
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        orig_dim = image.shape[2]
        dim_scale = float(orig_dim) / dim

        map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
        map_rows = row_scale * (map_rows + 0.5) - 0.5
        map_cols = col_scale * (map_cols + 0.5) - 0.5
        map_dims = dim_scale * (map_dims + 0.5) - 0.5

        coord_map = np.array([map_rows, map_cols, map_dims])

        image = _convert_warp_input(image, preserve_range)

        out = ndi.map_coordinates(image, coord_map, order=order,
                                  mode=ndi_mode, cval=cval)

        _clip_warp_output(image, out, order, mode, cval, clip)

    else:  # 2-dimensional interpolation

        if rows == 1 and cols == 1:
            tform = AffineTransform(translation=(orig_cols / 2.0 - 0.5,
                                                 orig_rows / 2.0 - 0.5))
        else:
            # 3 control points necessary to estimate exact AffineTransform
            src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
            dst_corners = np.zeros(src_corners.shape, dtype=np.double)
            # take into account that 0th pixel is at position (0.5, 0.5)
            dst_corners[:, 0] = col_scale * (src_corners[:, 0] + 0.5) - 0.5
            dst_corners[:, 1] = row_scale * (src_corners[:, 1] + 0.5) - 0.5

            tform = AffineTransform()
            tform.estimate(src_corners, dst_corners)

        out = warp(image, tform, output_shape=output_shape, order=order,
                   mode=mode, cval=cval, clip=clip,
                   preserve_range=preserve_range)

    return out