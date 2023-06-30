import logging
from contextlib import contextmanager
from contextlib import ExitStack

import rasterio
import os
from pathlib import Path
from glob import glob
import geoRpro.raster as rst

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@contextmanager
def to_src(arr, metadata):
    if metadata["driver"] != "GTiff":
        metadata["driver"] = "GTiff"

    with ExitStack() as stack:
        memfile = stack.enter_context(rasterio.MemoryFile())
        with memfile.open(**metadata) as data:  # Open as DatasetWriter
            if arr.ndim == 2:
                data.write_band(1, arr)
                logger.debug(f"Saved band and metadata as DataSetWriter")
            else:
                data.write(arr.astype(metadata["dtype"]))
                logger.debug(f"Saved array and metadata as DataSetWriter")
            del arr
        with memfile.open() as data:  # Reopen as DatasetReader
            yield data


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
        if meta["driver"] != "GTiff":
            meta["driver"] = "GTiff"

        with rasterio.open(fpath, "w", **meta) as dst:
            count = 1
            for src in srcs:
                if src.meta["count"] > 1:
                    stack_count = 1
                    while stack_count <= src.meta["count"]:
                        print(f"Writing to disk src with res: {src.res}")
                        arr, _ = rst.load(src, bands=[stack_count], masked=mask)
                        dst.write_band(stack_count, arr[0, :, :].astype(meta["dtype"]))
                        stack_count += 1
                    count = stack_count
                else:
                    print(f"Writing to disk src with res: {src.res}")
                    arr = src.read(masked=mask)
                    dst.write_band(count, arr[0, :, :].astype(meta["dtype"]))
                    count += 1

    return fpath


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
    assert (
        meta["driver"] == "GTiff"
    ), "Please use GTiff driver to write to disk. \
    Passed {} instead.".format(
        meta["driver"]
    )

    assert (
        arr.ndim == 3
    ), "np_array must have ndim = 3. \
    Passed np_array of dimension {} instead.".format(
        arr.ndim
    )

    # Register GDAL format drivers and configuration options with a
    # context manager.
    with rasterio.Env():
        with rasterio.open(fpath, "w", **meta) as dst:
            for layer_idx in range(arr.shape[0]):
                # follow gdal convention, start indexing from 1 -> layer_idx+1
                dst.write_band(
                    layer_idx + 1, arr[layer_idx, :, :].astype(meta["dtype"])
                )
    return fpath


def get_metadata(fpath):
    """
    Return a dict of metadata associated with the geo-raster
    """
    with rasterio.open(fpath, "r") as dst:
        return dst.profile

def delete_rasters(dpath, *keep):

    fpaths_all = [Path(f).resolve() for f in glob(os.path.join(dpath, "*.tif"))]
    if keep:
        fpaths_to_keep = [Path(dpath, name+".tif").resolve() for name in keep]
        fpaths_to_delete = set(fpaths_all).difference(fpaths_to_keep)
    else:
        fpaths_to_delete = fpaths_all

    for fpath in fpaths_to_delete:
        try:
            os.remove(fpath)
        except IOError as e:
            print(e)

