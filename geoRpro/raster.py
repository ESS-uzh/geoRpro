import os
import copy
import logging
from contextlib import contextmanager
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
import shapely

from geoRpro.sent2 import Sentinel2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# - utils for processing array and metadata


def mask_vals(arr, meta, vals):
    """
    Create a masked array (data, mask, fill_value) based on a list of values
    and update metadata

    data: Data array with masked value
    mask: Boolean array with True: masked and False: not masked

    *********
    params:
        vals -> a list of values that shold be masked

    return:
        tuple: masked array, metadata
    """
    # update metadata
    new_meta = meta.copy()
    new_meta.update({
        'driver': 'GTiff',
        'nbits': 1})
    # create a masked array
    arr = np.ma.MaskedArray(arr, np.in1d(arr, vals))
    return arr, new_meta


def mask_cond(arr, meta, cond):
    """
    Create a masked array (data, mask, fill_value) based on a condition
    and update metadata

    data: Data array with masked value
    mask: Boolean array with True: masked and False: not masked

    *********
    params:
        t_arr -> numpy arrays (bands, width, heigh)
        cond -> masking condition
        meta -> metadata associated with the target arrays

    return:
        tuple: masked array, metadata
    """
    # update metadata
    new_meta = meta.copy()
    new_meta.update({
        'driver': 'GTiff',
        'nbits': 1})
    # create a masked array
    arr = np.ma.masked_where(cond, arr)
    return arr, new_meta


def apply_mask(arr, mask_arr, fill_value=999999):
    """
    Apply a mask array on a target array

    *********

    params:
        mask_arr -> numpy mask arr (binary or boolean)
        fill_value -> value used to fill in the masked values
    return:
        tuple: masked filled array, metadata
    """
    # check arr and mask have the same dim
    assert (arr.shape == mask_arr.shape),\
        "Array and mask must have the same dimensions!"
    # get masked array
    masked_arr = np.ma.array(arr, mask=mask_arr)

    # Fill masked vales with zero !! maybe to be changed
    arr_filled = np.ma.filled(masked_arr, fill_value=fill_value)
    return arr_filled


@contextmanager
def to_src(arr, metadata):

    with ExitStack() as stack:
        memfile = stack.enter_context(rasterio.MemoryFile())
        with memfile.open(**metadata) as data: # Open as DatasetWriter
            if arr.ndim == 2:
                data.write_band(1, arr)
                logger.debug(f"Saved band and metadata as DataSetWriter")
            else:
                data.write(arr.astype(metadata['dtype']))
                logger.debug(f"Saved array and metadata as DataSetWriter")
        with memfile.open() as data:  # Reopen as DatasetReader
          yield data


# - utils for writing array/raster to disk

def write_array_as_raster(arr, meta, fpath):
    """
    Save a numpy array as geo-raster to disk

    params:
    --------

        arr :  nd numpy array
               must be 3D array (bands, height, width)

        meta : dict
               metadata for the new raster

        fname : string
               full path of the new raster file
    return:

        full path of the new raster
    """
    assert (meta['driver'] == 'GTiff'),\
        "Please use GTiff driver to write to disk. \
    Passed {} instead.".format(meta['driver'])

    assert (arr.ndim == 3),\
        "np_array must have ndim = 3. \
    Passed np_array of dimension {} instead.".format(arr.ndim)

    with rasterio.open(fpath, 'w', **meta) as dst:
        for layer_idx in range(arr.shape[0]):
            # follow gdal convention, start indexing from 1 -> layer_idx+1
            dst.write_band(layer_idx+1, arr[layer_idx,:,:].astype(meta['dtype']))
    return fpath


def write_raster(srcs, meta, fpath, mask=False):
    """
    Save a geo-raster to disk

    params:
    --------

        srcs : list
               collection of rasterio.DatasetReader objects

        meta : dict
               metadata af the final geo-raster

        fname : string
               full path of the new raster file

        mask : bool (dafault=False)
               if True, return a mask array
    return:

        full path of the new raster
    """
    with rasterio.open(fpath, 'w', **meta) as dst:
        for _id, src in enumerate(srcs, start=1):
            print(f"Writing to disk src with res: {src.res}")
            arr = src.read(masked=mask)
            dst.write_band(_id, arr[0, :, :].astype(meta['dtype']))
    return fpath


# - utils for processing src: rasterio.DataSetReader


def load(src, masked=False):
    "load array in memory"
    arr = src.read(masked=masked)
    metadata = src.profile
    return arr, metadata


def load_band(src, index, masked=False):
    "load array in memory"
    arr = src.read(index, masked=masked)
    metadata = src.profile
    return arr, metadata


def load_window(src, window, masked=False):
    """
    Return an area of interest (aoi)
    *********

    params:
        window -> rasterio.windows.Window

    return:
        tuple: array, metadata
    """
    arr = src.read(window=window, masked=masked)
    metadata = src.profile.copy()
    metadata.update({
        'driver': 'GTiff',
        'height': window.height,
        'width': window.width,
        'transform': rasterio.windows.transform(window, src.transform)})
    return arr, metadata


def load_resample(src, scale=2):
    """
    Change the cell size of an existing raster object.

    Can be used for both:

    Upsampling; converting to higher resolution/smaller cells
    Downsampling converting to lower resolution/larger cells

    a raster object.

    Save the new raster directly to disk.

    ************

    params:
        src -> rasterio.DatasetReader
        scale -> scaling factor to change the cell size with.
                 scale = 2 -> Upsampling e.g from 10m to 20m resolution
                 scale = 0.5 -> Downsampling e.g from 20m to 10m resolution

    yield:
        src -> resampled rasterio.DatasetReader
    """
    t = src.transform

    # rescale the metadata
    transform = rasterio.Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
    height = src.height * scale
    width = src.width * scale

    metadata = copy.deepcopy(src.profile)
    metadata.update(transform=transform, driver='GTiff', height=height,
                    width=width)

    # resampling
    arr = src.read(out_shape=(src.count, int(height), int(width),),
                    resampling=rasterio.enums.Resampling.nearest)
    return arr, metadata


def load_ndvi(cls, src_red, src_nir):
    """
    Calc ndvi array
    *********

    params:
        nir_arr -> 2D numpy arrays (width, heigh)
        meta -> metadata associated with one of the two arrays

    return:
        tuple of new numpy arr and relative metadata
    """
    red = src_red.read(1)
    nir = src_nir.read(1)
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = (nir.astype(np.float32)-red.astype(np.float32))/ \
               (nir.astype(np.float32)+red.astype(np.float32))

    # updata metadata
    new_meta = src_red.meta.copy()
    new_meta.update({
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1})
    arr = np.expand_dims(ndvi, axis=0)
    return arr, new_meta





class Rstack:
    """
    Stack of geo rasters

    Attributes
    ----------

    items: list of rasterio.DatasetReader objects
    """
    def __init__(self, items=None):

        if items is None:
            items = []
        self.items = items
        self.metadata_collect = None
        self._gen_metadata()

    def __check_for_crs(self):
        """
        Check for CRS consistency for the collection
        """
        if not all(r.crs.to_epsg() == self.items[0].crs.to_epsg()
                   for r in self.items[1:]):
            raise ValueError("CRS of all rasters should be the same")

    def __check_for_dimensions(self):
        """
        Check for dimension consistency for the collection
        """
        if not all(r.width == self.items[0].width and r.height == self.items[0].height
                   for r in self.items[1:]):
            raise ValueError("height and width of all rasters should be the same")

    def __check_for_resolution(self):
        """
        Check for dimension consistency for the collection
        """
        if not all(r.res == self.items[0].res for r in self.items[1:]):
            raise ValueError("Cannot stack rasters with different spacial resolution")

    def _gen_metadata(self):
        """
        Generate metadata for the collection
        """
        if self.items:
            # copy metadata of the first item
            self.metadata_collect = self.items[0].profile
            self.__check_for_crs()
            self.__check_for_dimensions()
            self.metadata_collect.update(count=len(self.items))
            self.metadata_collect.update(driver='GTiff')

    def set_metadata_param(self, param, value):
        self.metadata_collect[param] = value

    def add_item(self, item):
        """
        Add a new item to the item collection and update metadata_collect

        params:
        ----------

             item :  rasterio.DatasetReader
        """
        self.items.append(item)
        if len(self.items) == 1:
            self._gen_metadata()
        else:
            self.metadata_collect.update(count=len(self.items))
            # ! check for other meta of the item
            self.__check_for_resolution()
            self.__check_for_crs()
            self.__check_for_dimensions()
            if item.meta['dtype'] != self.metadata_collect['dtype']:
                self.metadata_collect.update(dtype=item.meta['dtype'])

    def extend_items(self, items):
        """
        Extend the item collection and update metadata_collect

        params:
        ----------

             item :  list
                     list of rasterio.DatasetReader objects
        """
        # ! TO DO:
        # ! check for meta of new items, e.g. dtype, dimensions, crs
        self.items.extend(items)
        self.metadata_collect.update(count=len(self.items))

    def reorder_items(self, new_order):
        """
        Change the order of the items

        params:
        ----------

             new_order :  list
                          list of indexes defining the new order of the items
                          e.g., [3, 2, 0, 1, 4]
        """
        self.items = [self.items[i] for i in new_order]

    def gen_windows(self, approx_patch_size):
        """
        Yields patches of the entire stack as numpy array

        params:
        ----------

             patch_size :  int
                           size of the patch, e.g 500 pixels
        """
        v_split = self.items[0].height // approx_patch_size
        h_split = self.items[0].width // approx_patch_size

        # get patch arrays
        v_arrays = np.array_split(np.arange(self.items[0].height), v_split)
        h_arrays = np.array_split(np.arange(self.items[0].width), h_split)
        new_meta = self.metadata_collect
        for v_arr in v_arrays:
            v_start = v_arr[0]
            for h_arr in h_arrays:
                h_start = h_arr[0]
                yield Window(h_start, v_start, len(h_arr), len(v_arr))


    def get_window(self, window):
        new_meta = self.metadata_collect
        new_meta.update({
        'driver': 'GTiff',
        'height': window.height,
        'width': window.width,
        'transform': rasterio.windows.transform(window, self.metadata_collect['transform'])})

        arr = np.array([src.read(1, window=window) for src in self.items])
        #if not mask.size == 0:
        #    print("mask")
        #    mask_win = mask[0][window.row_off: window.row_off+window.height,
        #                    window.col_off:window.col_off+window.width]
        #    masked_arr = np.ma.masked_array(*np.broadcast_arrays(arr, mask_win))
        #    arr = np.ma.filled(masked_arr, fill_value=fill_value)
        return arr, new_meta



if __name__ == "__main__":


    from rasterio.windows import Window
    from contextlib import ExitStack
    import pdb

    INDIR = "/home/diego/work/dev/data"
    s10 = Sentinel2(os.path.join(INDIR, "amazon/S2B_MSIL2A_20200803T142739_N0214_R053_T20MPA_20200803T165642.SAFE/GRANULE/L2A_T20MPA_A017811_20200803T142734/IMG_DATA/R10m/"))
    win = Window(0, 0, 1000, 1000)

    with ExitStack() as stack_files:
        rstack = Rstack([stack_files.enter_context(rasterio.open(fp))
            for fp in s10.get_fpaths('B02_10m', 'B03_10m', 'B04_10m', 'B08_10m')])
        arr, meta = rstack.get_window(win)
        write_array_as_raster(arr, meta, os.path.join(s10.dirpath, "S2B_T20MPA_20200803_Subset.tif"))


