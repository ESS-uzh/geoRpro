import os
import copy
import fnmatch
from contextlib import contextmanager
from contextlib import ExitStack
from pathlib import Path
from collections import OrderedDict

import numpy as np
import rasterio
import fiona
from shapely.geometry import shape, box, Point, Polygon
import geopandas as gpd

import geoRpro.raster as rst
from geoRpro.raster import Rstack, Indexes
from geoRpro.sent2 import Sentinel2
from geoRpro.extract import DataExtractor
from geoRpro.model import train_RandomForestClf, predict, predict_parallel
from geoRpro.utils import load_json

import logging
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

########################

# Workflow used to classify sentinel2 raster data

# 1. A stack of sent2 images (20 m resolution) over an AOI is produced
# 1.1 The stack has 14 bands, including calculated indexes (ndvi, bsi, etc.)
# 1.2 The bands of the stack are cloud-masked. This is achieved using the
#     sent2 SCL file
# 2. Pixel values of the stack are extracted at point locations forming and
#    used to train a classifier
# 3. A prediction map is produced using a trained classifier over the whole dataset

########################


def make_raster_stack():

    with ExitStack() as stack_files:

        # use SCL file to create a cloud-mask array
        ras_scl = stack_files.enter_context(rasterio.open(s20.get_fpaths('SCL_20m')[0]))
        arr_scl, meta_scl = rst.load_raster_from_poly(ras_scl, poly)
        arr_mask, _ = rst.mask_vals(arr_scl, meta_scl, [3,7,8,9,10])

        # collect all relevant bands
        ras10 = [stack_files.enter_context(rasterio.open(fp))
            for fp in s10.get_fpaths( 'B08_10m')]
        ras20 = [stack_files.enter_context(rasterio.open(fp))
            for fp in s20.get_fpaths('B02_20m', 'B03_20m', 'B04_20m', 'B05_20m', 'B06_20m', 'B07_20m', 'B8A_20m', 'B11_20m', 'B12_20m')]
        ras_collect = ras10+ras20

        with ExitStack() as stack_action:
            rstack = Rstack()
            for idx_ras, src in enumerate(ras_collect):
                if int(src.res[0]) == 10: # resample to 20 m
                    logger.info(f"Resampling: {src.files[0]} to 20 m resolution")
                    arr_r, meta = rst.load_resample(src, 0.5)
                    src = stack_action.enter_context(rst.to_src(arr_r, meta))
                arr, meta = rst.load_raster_from_poly(src, poly)
                arr_masked = rst.apply_mask(arr, arr_mask.mask, fill_value=9999)
                src = stack_action.enter_context(rst.to_src(arr_masked, meta))
                logger.info(f"Add raster with resulution: {src.res} to the stack")
                rstack.add_item(src)

            # calc indexes
            indexes_to_calc = OrderedDict([("calc_ndvi", [rstack.items[3], rstack.items[0]]),
                                           ("calc_nbr", [rstack.items[0], rstack.items[9]]),
                                           ("calc_bsi", [rstack.items[1], rstack.items[3], rstack.items[0], rstack.items[9]]),
                                           ("calc_ndwi", [rstack.items[2], rstack.items[0]])])

            indexes = Indexes(metadata=rstack.items[0].profile)

            for idx,vals in indexes_to_calc.items():
                arr_idx, meta_idx = getattr(indexes, idx)(*vals)
                arr_idx_masked = rst.apply_mask(arr_idx, arr_mask.mask, fill_value=9999)
                src_idx = stack_action.enter_context(rst.to_src(arr_idx_masked, meta_idx))
                rstack.add_item(src_idx)
            # final bands order of the stack
            rstack.set_metadata_param('interleave', 'band')
            order = [1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 11, 12, 13]
            rstack.reorder_items(order)
            fname = "_".join([s20.get_tile_number("B02_20m"), s20.get_datetake("B02_20m")])+"_stack.tif"
            fpath = rst.write_raster(rstack.items, rstack.metadata_collect, os.path.join(OUTDIR, fname))
    return fpath


def extract_Xy(fpath):

    with rasterio.open(fpath) as src:
        data = DataExtractor.extract(src, gdf, mask_value=9999)
        print(data.X.shape)
        print(data.y.shape)
        print(data.labels_map)
    data.add_class("masked", 9999)
    data.remove_class("Agriculture")
    data.remove_class("Building")
    data.remove_class("Road")
    logger.info(f"Final data mapping: {data.labels_map}")
    fname = "_".join([os.path.basename(fpath).split(".")[0], "training_data.json"])
    data.save(os.path.join(OUTDIR, fname))
    return data.X, data.y


if __name__ == "__main__":

    # Globals
    OUTDIR = "" # change me
    BASEDIR = "" # change me (Full path to IMG_DATA)
    DATA10 = "R10m/"
    DATA20 = "R20m/"
    POINTS_FP = "" # change me
    s10 = Sentinel2(os.path.join(BASEDIR, DATA10))
    s20 = Sentinel2(os.path.join(BASEDIR, DATA20))
    gdf = gpd.read_file(POINTS_FP)
    poly = box(*list(gdf.total_bounds))

    with rasterio.open(s20.get_fpaths('TCI_20m')[0]) as ras_tci:
        # get the AOI over TCI and save to disk
        arr_tci, meta_tci = rst.load_raster_from_poly(ras_tci, poly)
        fname_tci = "_".join([s20.get_tile_number("TCI_20m"), s20.get_datetake("TCI_20m")])+"_AOI.tif"
        rst.write_array_as_raster(arr_tci, meta_tci, os.path.join(OUTDIR, fname_tci))

    fp = make_raster_stack()
    X, y = extract_Xy(fp)

    for run in range(1, 31):

        logger.info("")
        logger.info(f"Start run {run}")

        clf = train_RandomForestClf(X, y, estimators=200)

        with rasterio.open(fp) as src:
            fname_cls = "_".join([os.path.basename(fp).split(".")[0], "classification_map_" , str(run) +".json"])
            class_f = predict_parallel(src, clf, write=True, fpath=os.path.join(OUTDIR, fname_cls))
            # save class_f as raster
            new_meta = src.meta.copy()
            new_meta.update({
            'driver': 'GTiff',
            'dtype': 'uint8',
            'count': 1})
            class_f = np.expand_dims(class_f, axis=0)
            rst.write_array_as_raster(class_f, new_meta, os.path.join(OUTDIR, fname_cls.split(".")[0]+".tif"))

        logger.info("Done")
        logger.info("")
