import os
import copy
import math
import logging
from contextlib import contextmanager
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window

import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# - utils for processing array and metadata


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
        'driver': 'GTiff',
        'nbits': 1})
    # create a masked array
    arr = np.ma.MaskedArray(arr, np.in1d(arr, vals))
    return arr, new_meta


def mask_cond(arr, meta, cond):
    """
    Create a masked array (data, mask) and metadata based on condition

    (data) masked values where the condition (cond) is True
    (mask) boolean mask, TRUE for masked values FALSE everywhere else

    params:
    --------

        arr :  nd numpy array

        meta : dict
               metadata for the new raster

        cond : array-like
               masking condition; e.g. masks values greater then 2
               example: mask_arr, meta_bin = rst.mask_cond(arr, meta, arr > 2)
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

    # Register GDAL format drivers and configuration options with a
    # context manager.
    with rasterio.Env():
        with rasterio.open(fpath, 'w', **meta) as dst:
            for layer_idx in range(arr.shape[0]):
                # follow gdal convention, start indexing from 1 -> layer_idx+1
                dst.write_band(layer_idx+1, arr[layer_idx, :, :].astype(meta['dtype']))
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

        mask : bool (dafault:False)
               if True, return a mask array
    return:

        full path of the new raster
    """

    # Register GDAL format drivers and configuration options with a
    # context manager.
    with rasterio.Env():

        with rasterio.open(fpath, 'w', **meta) as dst:
            for _id, src in enumerate(srcs, start=1):
                print(f"Writing to disk src with res: {src.res}")
                arr = src.read(masked=mask)
                dst.write_band(_id, arr[0, :, :].astype(meta['dtype']))

    return fpath


def mosaic_rasters(srcs, fpath):
    """
    Mosaic raster files band by band and save it to disk

    *********

    params:
    ---------

        srcs : list
               collection of rasterio.DatasetReader objects

        fpath : full path of the final raster mosaic
    """

    first_src = srcs[0]
    first_res = first_src.res
    dtype = first_src.dtypes[0]
    # Determine output band count
    output_count = first_src.count


    # Extent of all inputs
    # scan input files
    xs = []
    ys = []
    for src in srcs:
        left, bottom, right, top = src.bounds
        xs.extend([left, right])
        ys.extend([bottom, top])
    dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)

    out_transform = rasterio.Affine.translation(dst_w, dst_n)

    # Resolution/pixel size
    res = first_res
    out_transform *= rasterio.Affine.scale(res[0], -res[1])

    # Compute output array shape. We guarantee it will cover the output
    # bounds completely
    output_width = int(math.ceil((dst_e - dst_w) / res[0]))
    output_height = int(math.ceil((dst_n - dst_s) / res[1]))

    # Adjust bounds to fit
    dst_e, dst_s = out_transform * (output_width, output_height)
    # create destination array
    # destination array shape
    shape = (output_height, output_width)
    # dest = np.zeros((output_count, output_height, output_width), dtype=dtype)
    # Using numpy.memmap to create arrays directly mapped into a file
    from tempfile import mkdtemp
    memmap_file = os.path.join(mkdtemp(), 'test.mymemmap')
    dest_array = np.memmap(memmap_file, dtype=dtype, mode='w+', shape=shape)

    dest_profile = {
            "driver": 'GTiff',
            "height": dest_array.shape[0],
            "width": dest_array.shape[1],
            "count": output_count,
            "dtype": dest_array.dtype,
            "crs": '+proj=latlong',
            "transform": out_transform
    }

    # open output file in write/read mode and fill with destination mosaic array
    with rasterio.open(fpath, 'w+', **dest_profile) as mosaic_raster:
        for src in srcs:
            for ji, src_window in src.block_windows():
                print(ji)
                arr = src.read(window=src_window)
                # store raster nodata value
                nodata = src.nodatavals[0]
                # replace zeros with nan
                #arr[arr == nodata] = np.nan
                # convert relative input window location to relative output # windowlocation
                # using real world coordinates (bounds)
                src_bounds = rasterio.windows.bounds(src_window, transform=src.profile["transform"])
                dst_window = rasterio.windows.from_bounds(*src_bounds, transform=mosaic_raster.profile["transform"])

                # round the values of dest_window as they can be float
                dst_window = rasterio.windows.Window(round(dst_window.col_off), round(dst_window.row_off), round(dst_window.width), round(dst_window.height))
                # before writing the window, replace source nodata with dest
                # nodataas it can already have been written (e.g. another adjacent # country)
                # https://stackoverflow.com/a/43590909/1979665
                dest_pre = mosaic_raster.read(window=dst_window)
                mask = (arr == nodata)
                r_mod = np.copy(arr)
                r_mod[mask] = dest_pre[mask]
                mosaic_raster.write(r_mod, window=dst_window)

    os.remove(memmap_file)
    return fpath

# - utils for processing src: rasterio.DataSetReader


def load(src, masked=False):
    """
    Load a raster array

    *********

    params:
    --------

        src : rasterio.DatasetReader object

        masked : bool (default=False)
                 if True exclude nodata values


    return:
        tuple: array, metadata
  """
    arr = src.read(masked=masked)
    metadata = src.profile
    return arr, metadata


def load_bands(src, indexes, masked=False):
    """
    Load selected bands of a raster as array

    *********

    params:
    --------

        src : rasterio.DatasetReader object

        indexes : list
                  list of bands to load, e.g. [1,2,3]

        masked : bool (default:False)
                 if True exclude nodata values


    return:
        tuple: array, metadata
    """
    arr = src.read(indexes, masked=masked)
    metadata = src.profile
    metadata.update({
        'driver': 'GTiff',
        'count': len(indexes)})
    return arr, metadata


def load_window(src, window, masked=False):
    """
    Load a raster array from a window

    *********

    params:
    --------

        src : rasterio.DatasetReader object

        window : rasterio.windows.Window

        masked : bool (default:False)
                 if True exclude nodata values

    return:
        tuple: array, metadata
  """
    arr = src.read(window=window, masked=masked)
    metadata = src.profile
    metadata.update({
        'driver': 'GTiff',
        'height': window.height,
        'width': window.width,
        'transform': rasterio.windows.transform(window, src.transform)})
    return arr, metadata


def load_raster_from_poly(src, geom, crop=True):
    """
    Load a raster array from an input shape

    params:
    --------

        src : rasterio.DatasetReader object

        geom : GEOJson-like dict
               input shape, e.g. { 'type': 'Polygon', 'coordinates': [[(),(),(),()]] }

        crop : bool (dafault:True)
               Whether to crop the raster to the extent of the shapes

    return:

        tuple: array, metadata
    """
    arr, out_transform = rasterio.mask.mask(src, [geom], crop=crop)
    metadata = src.profile
    metadata.update({
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "transform": out_transform})
    return arr, metadata


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

    metadata = copy.deepcopy(src.profile)
    metadata.update(transform=transform, driver='GTiff', height=height,
                    width=width)

    # resampling
    arr = src.read(out_shape=(src.count, int(height), int(width),),
                   resampling=method)
    return arr, metadata


def get_windows(src):
    """
    Return a list of all windows composing the entire raster

    """
    return [win for _, win in src.block_windows(1)]


def gen_windows(src):
    """
    Yields all windows composing the entire raster

    """
    for _, win in src.block_windows(1):
        yield win


def gen_blocks(src):
    """
    Yields all block-arrays composing the entire raster, each block has associated metadata

    """
    for _, win in src.block_windows(1):
        arr, meta = load_window(src, win)
        yield arr, meta


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


    def calc_ndvi(self, red_src, nir_src):
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


    def calc_nbr(self, nir_src, swir_src):
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


    def calc_bsi(self, blue_src, red_src, nir_src, swir_src):
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


    def calc_ndwi(self, green_src, nir_src):
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


class Rstack:
    """
    Create a stack of geo-rasters and metadata associate with it

    Rasters added to this class must have the same crs, dimension
    and spacial resolution

    Attributes
    ----------

      items: list
             collection of rasterio.DatasetReader objects
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


    def get_window(self, window, masked=False):
        """
        Load a window of the rstack instance

        params:
        ----------

        window : rasterio.windows.Window

        masked : bool (default=False)
                 if True exclude nodata values

        return:
            tuple: array, metadata
        """
        new_meta = self.metadata_collect
        new_meta.update({
        'driver': 'GTiff',
        'height': window.height,
        'width': window.width,
        'transform': rasterio.windows.transform(window, self.metadata_collect['transform'])})

        arr = np.array([src.read(1, window=window, masked=masked) for src in self.items])
        return arr, new_meta
