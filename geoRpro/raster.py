import os
import sys
import copy
import math
import logging
import collections
from contextlib import contextmanager
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
from rasterio.warp import Resampling
from geoRpro.sent2 import Sentinel2

import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load(src, window=False, masked=False, bands=None, **kwargs):
    """
    Load a 3D numpy array in memory

    *********

    params:
    --------

    window : a pair (tuple) of pairs of ints or rasterio.windows.Window obj
             For example ((0, 2), (0, 2)) defines a 2x2 window at the upper
             left of the raster dataset.

    masked : bool (default=False)
             if True exclude nodata values

    bands: list
           number of bands to be loaded

    -------

    return: tuple (array, meta)
               array: 3D numpy arr (bands, rows, columns)
               meta: dict (metadata associated with the array)
    """
    meta = src.profile

    if not bands:
        bands = list(src.indexes)

    meta.update({
        'count': len(bands)})

    if window:
        if isinstance(window, tuple):
            height = window[0][1] -window[0][0]
            width = window[1][1] -window[1][0]
        else:
            height = window.height
            width = window.width
        meta.update({
            'height': height,
            'width': width,
            'transform': rasterio.windows.transform(window, src.transform)})

    arr = src.read(bands, window=window, masked=masked, **kwargs)

    return arr, meta

def load_resample(src, scale=2, method=rasterio.enums.Resampling.nearest):
    """
    Change the cell size of an existing raster object.

    Can be used for both:

    Upsampling; converting to higher resolution/smaller cells
    Downsampling converting to lower resolution/larger cells

    a raster object.

    params:
    --------

        src : rasterio.DatasetReader object

        scale : int (default:2)
                 scaling factor to change the cell size with.
                 scale = 2 -> Upsampling e.g from 10m to 20m resolution
                 scale = 0.5 -> Downsampling e.g from 20m to 10m resolution

    return:

        tuple: array, metadata
    """
    t = src.transform

    # rescale the metadata
    transform = rasterio.Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
    height = src.height * scale
    width = src.width * scale

    meta = src.profile
    meta.update(transform=transform, driver='GTiff', height=height,
                    width=width)

    # resampling
    arr = src.read(out_shape=(src.count, int(height), int(width),),
                   resampling=method)
    return arr, meta

def load_polygon():
    pass

def mask_vals(arr, meta, vals):
    """
    Create a masked array (data, mask) and metadata based on a list of values

    (data) masked values where the condition (cond) is True
    (mask) boolean mask, TRUE for masked values FALSE everywhere else

    params:
    --------

        arr :  nd numpy array

        meta : dict
               metadata for the new raster

        vals : list
               values that should be masked

    return:
        tuple: masked array, metadata
    """
    # update metadata
    new_meta = meta.copy()
    new_meta.update({
        'nbits': 1})
    # create a masked array
    arr = np.ma.MaskedArray(arr, np.in1d(arr, vals))
    return arr, new_meta

def apply_mask(arr, mask, fill_value=0):
    """
    Apply a mask to a target array and replace masked values with a
    fill_value


    params:
    --------

        mask : numpy boolean mask arr
               e.g. mask_arr.mask

        fill_value : int (default:0)
                     value used to fill in the masked values

    return:
           numpy array with values equal to fill_value where mask_arr
           is equal to 1
    """
    # check arr and mask_arr have the same dimensions
    assert (arr.shape == mask.shape),\
        "Array and mask must have the same dimensions!"

    # get masked array
    masked_arr = np.ma.array(arr, mask=mask)

    # Fill masked vales with fill_value
    arr_filled = np.ma.filled(masked_arr, fill_value=fill_value)
    return arr_filled
