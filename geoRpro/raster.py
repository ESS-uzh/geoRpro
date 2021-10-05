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

class Indexes:
    """
    Provide methods to calculate different raster indexes

    Attributes
    ----------

     metadata: dict
                profile information associated with one of the raster
                used to calculate the indexes

     scale_factor: int (default:1000)
               Calculated indexes array are scaled for one of those facors:
               1 -> Return a float array
               1000 -> Return an int array
    """

    def __init__(self, metadata, scale_factor=1000):

        self.metadata = metadata
        self.scale_factor = scale_factor

    @property
    def scale_factor(self):
        return self._scale_factor


    @scale_factor.setter
    def scale_factor(self, value):
        if value == 1000:
            self.metadata.update({
            'driver': 'GTiff',
            'count': 1})
        elif value == 1:
            self.metadata.update({
            'driver': 'GTiff',
            'dtype': 'float32',
            'count': 1})
        else:
            raise ValueError("Allowed scale factors are: 1 or 1000")
        self._scale_factor = value


    def _scale_and_round(self, arr):
        array = arr * self.scale_factor
        if self.scale_factor == 1000:
            array = array.astype(int)
        return array, self.metadata


    def ndvi(self, red_src, nir_src):
        # to do: check for rasters to be (1, width, height)
        redB = red_src.read()
        nirB = nir_src.read()
        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nirB.astype(np.float32)-redB.astype(np.float32))/ \
                   (nirB.astype(np.float32)+redB.astype(np.float32))
        # replace nan with 0
        where_are_NaNs = np.isnan(ndvi)
        ndvi[where_are_NaNs] = 0

        return self._scale_and_round(ndvi)


    def nbr(self, nir_src, swir_src):
        """ Normalized Burn Ratio """
        nirB = nir_src.read()
        swirB = swir_src.read()
        np.seterr(divide='ignore', invalid='ignore')
        nbr = (nirB.astype(np.float32)-swirB.astype(np.float32))/ \
                   (nirB.astype(np.float32)+swirB.astype(np.float32))
        # replace nan with 0
        where_are_NaNs = np.isnan(nbr)
        nbr[where_are_NaNs] = 0
        return self._scale_and_round(nbr)


    def bsi(self, blue_src, red_src, nir_src, swir_src):
        """ Bare Soil Index (BSI) """
        blueB = blue_src.read()
        redB = red_src.read()
        nirB = nir_src.read()
        swirB = swir_src.read()
        np.seterr(divide='ignore', invalid='ignore')
        bsi = ((swirB.astype(np.float32)+redB.astype(np.float32))-(nirB.astype(np.float32)+blueB.astype(np.float32))) / \
            ((swirB.astype(np.float32)+redB.astype(np.float32))+(nirB.astype(np.float32)+blueB.astype(np.float32)))
        # replace nan with 0
        where_are_NaNs = np.isnan(bsi)
        bsi[where_are_NaNs] = 0
        return self._scale_and_round(bsi)


    def ndwi(self, green_src, nir_src):
        """ Normalized Difference Water Index (NDWI)  """
        greenB = green_src.read()
        nirB = nir_src.read()
        np.seterr(divide='ignore', invalid='ignore')
        ndwi = (greenB.astype(np.float32)-nirB.astype(np.float32))/ \
                   (greenB.astype(np.float32)+nirB.astype(np.float32))
        # replace nan with 0
        where_are_NaNs = np.isnan(ndwi)
        ndwi[where_are_NaNs] = 0
        return self._scale_and_round(ndwi)
