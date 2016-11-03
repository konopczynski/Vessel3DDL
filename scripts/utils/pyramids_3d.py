# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:12:11 2016

@author: konop

This extends the skimage pyramids to work on 3D volumes
"""

import math
import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize
from skimage.util import img_as_float
import time

"""
arrays used for testing

a = np.array([[[ 1, 2, 3, 4],[ 5, 6, 7, 8],[ 9,10,11,12],[13,14,15,16]],
              [[17,18,19,20],[21,22,23,24],[25,26,27,28],[29,30,31,32]],
              [[33,34,35,36],[37,38,39,40],[41,42,43,44],[45,46,47,48]],
              [[49,50,51,52],[53,54,55,56],[57,58,59,60],[61,62,63,64]]])
              
w = np.array([[0,1],[0,1],[0,1]])

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
                     [[181,182],[184,185],[187,188],[190,191]]]], dtype='float32')
"""

def _smooth(image, sigma, mode, cval):
    """Return image with each channel smoothed by the Gaussian filter."""
    smoothed = np.empty(image.shape, dtype=np.double)
    # apply Gaussian filter to all dimensions independently
    if image.ndim == 3:
        for dim in range(image.shape[2]):
            ndi.gaussian_filter(image[..., dim], sigma,
                                output=smoothed[..., dim],
                                mode=mode, cval=cval)
    else:
        ndi.gaussian_filter(image, sigma, output=smoothed,
                            mode=mode, cval=cval)
    return smoothed

def _smooth_3d(volume, sigma, mode, cval):
    """Return volume with each channel smoothed by the Gaussian filter."""
    smoothed = np.empty(volume.shape, dtype=np.double)
    # apply Gaussian filter to all dimensions independently
    # volume.ndim == 4 means the volume is multimodal
    if volume.ndim == 4:
        # compute 3d convolution for each modality, dim is a modality
        for dim in range(volume.shape[3]):
            ndi.gaussian_filter(volume[..., dim], sigma,
                                output=smoothed[..., dim],
                                mode=mode, cval=cval)
    else:
        ndi.gaussian_filter(volume, sigma, output=smoothed, mode=mode, cval=cval)
    return smoothed

def _check_factor(factor):
    if factor <= 1:
        raise ValueError('scale factor must be greater than 1')

def _check_float32(volume):
    if np.dtype(volume[0][0][0]) != 'float32':
        raise ValueError('volume must be float32')

def pyramid_gaussian(image, max_layer=-1, downscale=2, sigma=None, order=1,
                     mode='reflect', cval=0):
    """Yield images of the Gaussian pyramid formed by the input image.
    Recursively applies the `pyramid_reduce` function to the image, and yields
    the downscaled images.
    Note that the first image of the pyramid will be the original, unscaled
    image. The total number of images is `max_layer + 1`. In case all layers
    are computed, the last image is either a one-pixel image or the image where
    the reduction does not change its shape.
    Parameters
    ----------
    image : array
        Input image.
    max_layer : int
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers as float images.
    References
    ----------
    .. [1] http://web.mit.edu/persci/people/adelson/pub_pdfs/pyramid83.pdf
    """

    _check_factor(downscale)

    # cast to float for consistent data type in pyramid
    image = img_as_float(image)

    layer = 0
    rows = image.shape[0]
    cols = image.shape[1]

    prev_layer_image = image
    yield image

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    while layer != max_layer:
        layer += 1

        layer_image = pyramid_reduce(prev_layer_image, downscale, sigma, order,
                                     mode, cval)

        prev_rows = rows
        prev_cols = cols
        prev_layer_image = layer_image
        rows = layer_image.shape[0]
        cols = layer_image.shape[1]

        # no change to previous pyramid layer
        if prev_rows == rows and prev_cols == cols:
            break

        yield layer_image
    
def pyramid_gaussian_3d(volume, max_layer=-1, downscale=2, sigma=None, order=1, mode='reflect', cval=0):
    """Yield images of the 3d Gaussian pyramid formed by the input volume.
    Recursively applies the `pyramid_reduce` function to the image, and yields
    the downscaled images.
    Note that the first image of the pyramid will be the original, unscaled
    volume. The total number of images is `max_layer + 1`. In case all layers
    are computed, the last image is either a one-pixel image or the image where
    the reduction does not change its shape.
    Parameters
    ----------
    volume : array
        Input volume.
    max_layer : int
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers as float volumes.
    References
    ----------
    .. [1] http://web.mit.edu/persci/people/adelson/pub_pdfs/pyramid83.pdf
    """

    _check_factor(downscale)

    # XXX: cast to float for consistent data type in pyramid
    # volume = img_as_float(volume)
    _check_float32(volume)
    
    layer = 0
    x = volume.shape[0]
    y = volume.shape[1]
    z = volume.shape[2]

    prev_layer_volume = volume
    yield volume

    # build downsampled volumes until max_layer is reached or downscale process
    # does not change volume size
    while layer != max_layer:
        layer += 1

        layer_volume = pyramid_reduce_3d(prev_layer_volume, downscale, sigma, order, mode, cval)

        prev_x = x
        prev_y = y
        prev_z = z
        prev_layer_volume = layer_volume
        x = layer_volume.shape[0]
        y = layer_volume.shape[1]
        z = layer_volume.shape[2]

        # no change to previous pyramid layer
        if prev_x == x and prev_y == y and prev_z == z:
            break

        yield layer_volume

def pyramid_reduce_3d(volume, downscale=2, sigma=None, order=1, mode='reflect', cval=0):
    """Smooth and then downsample volume.
    Parameters
    ----------
    volume : array
        Input image.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    Returns
    -------
    out : array
        Smoothed and downsampled float volume.
    References
    ----------
    .. [1] http://web.mit.edu/persci/people/adelson/pub_pdfs/pyramid83.pdf
    """

    _check_factor(downscale)

    _check_float32(volume)

    x = volume.shape[0]
    y = volume.shape[1]
    z = volume.shape[2]
    out_x = math.ceil(x / float(downscale))
    out_y = math.ceil(y / float(downscale))
    out_z = math.ceil(z / float(downscale))

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    smoothed = _smooth_3d(volume, sigma, mode, cval)
    out = resize(smoothed, (out_x, out_y, out_z), order=order, mode=mode, cval=cval)
    # I want it to be float32
    out=out.astype('float32')
    return out

def pyramid_3d(volume, downscale=2, sigma=None, order=1, mode='reflect', cval=0):
    _check_factor(downscale)
    _check_float32(volume)
    
    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0
    smoothed = _smooth_3d(volume, sigma, mode, cval)
    # I want it to be float32
    smoothed=smoothed.astype('float32')
    return smoothed
    
def pyramid_expand(image, upscale=2, sigma=None, order=1, mode='reflect', cval=0):
    """Upsample and then smooth image.
    Parameters
    ----------
    image : array
        Input image.
    upscale : float, optional
        Upscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * upscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of upsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    Returns
    -------
    out : array
        Upsampled and smoothed float image.
    References
    ----------
    .. [1] http://web.mit.edu/persci/people/adelson/pub_pdfs/pyramid83.pdf
    """

    _check_factor(upscale)

    image = img_as_float(image)

    rows = image.shape[0]
    cols = image.shape[1]
    out_rows = math.ceil(upscale * rows)
    out_cols = math.ceil(upscale * cols)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * upscale / 6.0

    resized = resize(image, (out_rows, out_cols), order=order, mode=mode, cval=cval)
    out = _smooth(resized, sigma, mode, cval)

    return out

def pyramid_expand_3d(volume, upscale=2, sigma=None, order=1, mode='reflect', cval=0):
    _check_factor(upscale)
    #_check_float32(volume)
    #volume=img_as_float(volume)
    volume=volume.astype('float64') # /(12641.6)  #
    x = volume.shape[0]
    y = volume.shape[1]
    z = volume.shape[2]
    out_x = math.ceil(upscale * x)
    out_y = math.ceil(upscale * y)
    out_z = math.ceil(upscale * z)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * upscale / 6.0
    
    start_time = time.time()
    resized = resize(volume, (out_x, out_y, out_z), order=order, mode=mode, cval=cval)
    start_time = time.time()
    out = _smooth_3d(resized, sigma, mode, cval)
    # I want it to be float32
    start_time = time.time()    
    out=out.astype('float32')
    return out
