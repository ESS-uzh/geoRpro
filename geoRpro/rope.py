import copy

import numpy as np
import rasterio
from contextlib import contextmanager
import pdb


def write_raster(src, fname):
    with rasterio.open(fname, 'w', **src.meta) as dst:
        dst.write(src.read())
    return fname


def mem_file(arr, metadata, *to_del_arr):
    """
    Create and yield a Datareader using a numpy array
    and metadata
    *********

    params:
        arr -> np.array (bands, rows, columns)
        metadata -> dict of metadata
        to_del_arr -> names of np.arrays to be deleted

    yield:
        src -> rasterio.DatasetReader
    """
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**metadata) as src: # Open as DatasetWriter
            if arr.ndim == 2:
                src.write_band(1, arr)
            else:
                src.write(arr.astype(metadata['dtype']))
            for arr in to_del_arr:
                del arr

        with memfile.open() as src:  # Reopen as DatasetReader
            yield src


@contextmanager
def calc_ndvi(src_red, src_nir):
    """
    *********

    params:
        src_red -> rasterio.DatasetReader
        src_nir -> rasterio.DatasetReader

    yield:
        src_ndvi -> rasterio.DatasetReader
    """
    red_arr = src_red.read(1)
    nir_arr = src_nir.read(1)
    np.seterr(divide='ignore', invalid='ignore')
    ndvi_arr = (nir_arr.astype(np.float32)-red_arr.astype(np.float32))/ \
               (nir_arr.astype(np.float32)+red_arr.astype(np.float32))

    # grab and copy metadata of one of the two array
    ndvi_meta = copy.deepcopy(src_red.meta)
    ndvi_meta.update(count=1, dtype="float32", driver='GTiff')
    return mem_file(ndvi_arr, ndvi_meta, red_arr, nir_arr)


@contextmanager
def resample_raster(src, scale=2):
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
    arr = src.read(resampling=rasterio.enums.Resampling.nearest)
    return mem_file(arr, new_meta, arr)



@contextmanager
def create_raster_mask(src, vals):
    """
    Mask the values of an array in vals list.

    *********

    params:
        arr -> numpy array
        vals -> a list of values
    return:
        numpy mask array

    """
    arr = src.read()
    mask_arr = np.ma.MaskedArray(arr, np.in1d(arr, vals))
    new_meta = copy.deepcopy(src.meta)
    new_meta.update(driver='GTiff', nbits=1)
    return mem_file(mask_arr.mask, new_meta, arr, mask_arr)



def mask_and_fill(arr, mask, fill_value=0):
    """
    Mask an array using a mask array and fill it with
    a fill value.

    *********

    params:
        arr -> numpy array to be masked
        mask -> numpy boolean mask array

    return:
        numpy masked array
    """
    # check arr and mask have the same dim
    assert (arr.shape == mask.shape),\
        "Array and mask must have the same dimensions!"
    masked_arr = np.ma.array(arr, mask=mask)

    # Fill masked vales with zero !! maybe to be changed
    return np.ma.filled(masked_arr, fill_value=fill_value)


def build_task(oper):
    try:
        src = rasterio.open("tests/T37MBN_20190628T073621_SCL_20m.jp2")
        print(src.meta)
        #pdb.set_trace()
        for o, params in oper:
            new_src = o(src, params)
            print(new_src.meta)
    finally:
        src.close()


class Chain():
    def __init__(self, kind=None):
        self.kind = kind
    def my_print(self):
        print (self.kind)
        return self
    def line(self):
        self.kind = 'line'
        return self
    def bar(self):
        self.kind='bar'
        return self

if __name__ == "__main__":
#    build_task([(resample_raster2, 2), (create_raster_mask2,[3,7,8,9,10])])
    with rasterio.open("tests/T37MBN_20190628T073621_SCL_20m.jp2") as src:
        with resample_raster(src) as resampled:
            write_raster(resampled, 'T37MBN_20190628T073621_SCL_res_10m.tif')
            print(resampled.meta)
            with create_raster_mask(resampled, [3,7,8,9,10]) as mask:
                unique, counts = np.unique(mask.read(), return_counts=True)
                print(dict(zip(unique, counts)))
                write_raster(mask, 'mask.tif')
