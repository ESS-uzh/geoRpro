import copy
import logging
from contextlib import contextmanager

import numpy as np
import rasterio
from rasterio.mask import mask
import shapely

import geoRpro.aope as ao

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# * - Series of raster operations yielding a rasterio.DatasetReader

def write_raster(src, fname):
    logger.debug(f"Writing to disk..")
    with rasterio.open(fname, 'w', **src.meta) as dst:
        dst.write(src.read())
    logger.debug(f"{fname} saved to disk")
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
        #logger.debug("Opening memfile as DataSetWriter")
        #logger.debug(f"array shape: {arr.shape}")
        with memfile.open(**metadata) as data: # Open as DatasetWriter
            if arr.ndim == 2:
                data.write_band(1, arr)
                logger.debug(f"Saved band and metadata as DataSetWriter")
            else:
                data.write(arr.astype(metadata['dtype']))
                logger.debug(f"Saved array and metadata as DataSetWriter")
            for arr in to_del_arr:
                del arr

        #logger.debug(f"Open memfile as DataSetReader")
        with memfile.open() as data:  # Reopen as DatasetReader
          yield data


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
    logger.debug(f"Loading: {src_red.name} as red array")
    red_arr = src_red.read(1)
    logger.debug(f"Loaded red array of shape: {red_arr.shape}")
    logger.debug(f"Load: {src_nir.name} as nir array")
    nir_arr = src_nir.read(1)
    logger.debug(f"Loaded nir array of shape: {nir_arr.shape}")
    ndvi_arr, ndvi_meta = ao.aope_ndvi(red_arr, nir_arr, src_red.meta)
    return mem_file(ndvi_arr, ndvi_meta, ndvi_arr)


@contextmanager
def get_aoi(src, window):
    """
    Writes an area of interest (aoi) to disk
    *********

    params:
        src -> rasterio.DatasetReader
        window -> rasterio.windows.Window

    yield:
        src -> resampled rasterio.DatasetReader
    """
    logger.debug(f"Selecting AOI for window: {window}")
    aoi = src.read(window=window)
    logger.debug(f"AOI has shape: {aoi.shape}")
    new_meta = src.meta.copy()
    new_meta.update({
        'driver': 'GTiff',
        'height': window.height,
        'width': window.width,
        'transform': rasterio.windows.transform(window, src.transform)})
    return mem_file(aoi, new_meta, aoi)


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
    logger.debug(f"Resampling raster cells of a factor: {scale}")
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
        src -> numpy array or rasterio.DatasetReader
        meta -> metadata associated with src
        vals -> a list of values
    yield:
        src -> binary (0=not masked, 1=masked) rasterio.DatasetReader

    """
    arr = src.read()
    logger.debug(f"Masking raster values: {vals}")
    mask, meta = ao.aope_mask(vals, arr, src.meta)
    return mem_file(mask.mask, meta, mask.mask)


@contextmanager
def apply_raster_mask(src, mask, fill_value=0):
    """
    Mask an array using a mask array and fill it with
    a fill value.

    *********

    params:
        arr -> numpy array to be masked
        mask -> numpy boolean mask array or a rasterio.Datareader

    return:
        numpy masked array
    """
    if not isinstance(mask, np.ndarray):
        mask = mask.read()
    arr = src.read()
    m_filled = ao.aope_apply_mask(arr, mask, fill_value)
    return mem_file(m_filled, src.meta, m_filled)


def extract_from_raster(src, gdf):
    """
    Extract shapes geometries from raster
    params:
        src -> rasterio.DatasetReader
        gdf -> geodataframe ('classname', 'id', 'geometry')

    yield:
        X,y numpy arrays
    """
    # Numpy array of shapely objects
    geoms = gdf.geometry.values

    # convert all labels_id to int
    gdf.id = gdf.id.astype(int)

    X = np.array([]).reshape(0,src.count)# pixels for training
    y = np.array([], dtype=np.int64) # labels for training
    for index, geom in enumerate(geoms):
        # Transform to GeoJSON format
        # [{'type': 'Point', 'coordinates': (746418.3300011896, 3634564.6338985614)}]
        feature = [shapely.geometry.mapping(geom)]

        # the mask function returns an array of the raster pixels within this feature
        # out_image.shape == (band_count,1,1) for one Point Object
        out_image, out_transform = mask(src, feature, crop=True)

        # reshape the array to [pixel values, band_count]
        out_image_reshaped = out_image.reshape(-1, src.count)
        y = np.append(y,[gdf['id'][index]] * out_image_reshaped.shape[0])
        X = np.vstack((X,out_image_reshaped))
    return X,y
