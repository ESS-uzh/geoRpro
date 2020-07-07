import copy
from contextlib import contextmanager
from memory_profiler import profile
import numpy as np
import rasterio
from rasterio.mask import mask
import shapely
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
        with memfile.open(**metadata) as data: # Open as DatasetWriter
            if arr.ndim == 2:
                data.write_band(1, arr)
            else:
                data.write(arr.astype(metadata['dtype']))
            for arr in to_del_arr:
                del arr

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
    red_arr = src_red.read(1)
    nir_arr = src_nir.read(1)
    np.seterr(divide='ignore', invalid='ignore')
    ndvi_arr = (nir_arr.astype(np.float32)-red_arr.astype(np.float32))/ \
               (nir_arr.astype(np.float32)+red_arr.astype(np.float32))

    # grab and copy metadata of one of the two array
    ndvi_meta = copy.deepcopy(src_red.meta)
    ndvi_meta.update(count=1, dtype="float32", driver='GTiff')
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
    aoi = src.read(window=window)
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
    return mem_file(mask_arr.mask, new_meta, mask_arr.mask)


@contextmanager
def apply_raster_mask(src, src_mask, fill_value=0):
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
    arr = src.read()
    mask = src_mask.read()

    # check arr and mask have the same dim
    assert (arr.shape == mask.shape),\
        "Array and mask must have the same dimensions!"
    masked_arr = np.ma.array(arr, mask=mask)

    # Fill masked vales with zero !! maybe to be changed
    m_filled = np.ma.filled(masked_arr, fill_value=fill_value)
    return mem_file(m_filled, src.meta, m_filled)


def extract_from_raster(src, gdf):
    """
    Extract shapes geometries from raster
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
