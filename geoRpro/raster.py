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

# * - Series of raster operations yielding a rasterio.DatasetReader


class RArr:
    """
    Array and metadata operation for geo-Rasters

    Each method updates the instance attributes; self.arr and self meta.

    Classmethods can be used as alterative constructors

    """
    def __init__(self, arr, meta):
            self.arr = arr
            self.meta = meta

    def write_array_as_raster(self, fpath):
        """
        Save a numpy array as geo-raster to disk
        *********
        params:
            arr ->  3D numpy array to save as raster
            meta -> metadata for the new raster
            fname -> name of the file (without ext)
        return:
            fpath -> full path of the new raster
        """
        assert (self.meta['driver'] == 'GTiff'),\
            "Please use GTiff driver to write to disk. \
        Passed {} instead.".format(self.meta['driver'])

        assert (self.arr.ndim == 3),\
            "np_array must have ndim = 3. \
        Passed np_array of dimension {} instead.".format(self.arr.ndim)

        with rasterio.open(fpath, 'w', **self.meta) as dst:
            dst.write(self.arr.astype(self.meta['dtype']))
        return fpath



    def mask_vals(self, vals):
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
        new_meta = self.meta.copy()
        new_meta.update({
            'driver': 'GTiff',
            'nbits': 1})
        # create a masked array
        self.arr = np.ma.MaskedArray(self.arr, np.in1d(self.arr, vals))
        self.meta = new_meta
        return self


    def mask_cond(self, cond):
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
        new_meta = self.meta.copy()
        new_meta.update({
            'driver': 'GTiff',
            'nbits': 1})
        # create a masked array
        self.arr = np.ma.masked_where(cond, self.arr)
        self.meta = new_meta
        return self


    def apply_mask(self, mask_arr, fill_value=-9999):
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
        assert (self.arr.shape == mask_arr.shape),\
            "Array and mask must have the same dimensions!"
        # get masked array
        masked_arr = np.ma.array(self.arr, mask=mask_arr)

        # Fill masked vales with zero !! maybe to be changed
        m_filled = np.ma.filled(masked_arr, fill_value=fill_value)
        self.arr = m_filled
        return self


    @classmethod
    def load(cls, src, index=None, masked=False):
        "load array in memory"

        if index:
            return RArr(src.read(index, masked=masked), src.meta)
        return RArr(src.read(masked=masked), src.meta)



    @classmethod
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
        return RArr(np.expand_dims(ndvi, axis=0), new_meta)


    @classmethod
    def load_window(cls, src, window):
        """
        Return an area of interest (aoi)
        *********

        params:
            window -> rasterio.windows.Window

        return:
            tuple: array, metadata
        """
        new_meta = src.meta.copy()
        new_meta.update({
            'driver': 'GTiff',
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, src.transform)})
        return RArr(src.read(window=window), new_meta)


    @classmethod
    def load_resample(cls, src, scale=2):
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

        new_meta = copy.deepcopy(src.meta)
        new_meta.update(transform=transform, driver='GTiff', height=height,
                        width=width)

        # resampling
        arr = src.read(out_shape=(src.count, int(height), int(width),),
                        resampling=rasterio.enums.Resampling.nearest)
        return RArr(arr, new_meta)

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

def write_raster(src, fpath):
    logger.debug(f"Writing to disk..")
    with rasterio.open(fpath, 'w', **src.meta) as dst:
        dst.write(src.read())
    logger.debug(f"{fname} saved to disk")
    return fpath


if __name__ == "__main__":


    from rasterio.windows import Window
    from contextlib import ExitStack
    import pdb

    INDIR = "/home/diego/work/dev/data"
    s20 = Sentinel2(os.path.join(INDIR, "geoRpro_inp/S2A_MSIL2A_20190628T073621_N9999_R092_T37MBN_20191121T145522.SAFE/GRANULE/L2A_T37MBN_A020967_20190628T075427/IMG_DATA/R20m"))
    s10 = Sentinel2(os.path.join(INDIR, "geoRpro_inp/S2A_MSIL2A_20190628T073621_N9999_R092_T37MBN_20191121T145522.SAFE/GRANULE/L2A_T37MBN_A020967_20190628T075427/IMG_DATA/R10m"))
    fpath_scl = s20.get_fpaths('SCL_20m')[0]
    fpath_nir = s10.get_fpaths('B08_10m')[0]
    fpath_red = s10.get_fpaths('B04_10m')[0]

    #s10 = Sentinel2(os.path.join(INDIR, "amazon/S2A_MSIL2A_20200729T142741_N0214_R053_T20MNC_20200729T165425.SAFE/GRANULE/L2A_T20MNC_A026648_20200729T142736/IMG_DATA/R10m"))
    #fpath_tcl = s10.get_fpaths('TCI_10m')[0]

    with ExitStack() as stack_files:
        scl_src = stack_files.enter_context(rasterio.open(fpath_scl))
        red_src = stack_files.enter_context(rasterio.open(fpath_red))
        nir_src = stack_files.enter_context(rasterio.open(fpath_nir))
        print(nir_src)
        with ExitStack() as stack_action:
            r_sc = RArr.load(scl_src)
            print(r_sc.arr)
            print(r_sc.meta)
            src = stack_action.enter_context(to_src(r_sc.arr, r_sc.meta))
            print(r.arr)
            print(r.meta)
            r.mask_vals([3,5])
            print(r.arr)
            print(r.meta)
    print(nir_src)
