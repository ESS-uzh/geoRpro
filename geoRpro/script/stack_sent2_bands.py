import os
import copy
from contextlib import contextmanager
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
import shapely

from geoRpro.sent2 import Sentinel2
from geoRpro.raster import Rstack
import geoRpro.raster as rst

import pdb

INDIR = "/home/diego/work/dev/data"
s10 = Sentinel2(os.path.join(INDIR, "amazon/S2B_MSIL2A_20200803T142739_N0214_R053_T20MPA_20200803T165642.SAFE/GRANULE/L2A_T20MPA_A017811_20200803T142734/IMG_DATA/R10m/"))
s20 = Sentinel2(os.path.join(INDIR, "amazon/S2B_MSIL2A_20200803T142739_N0214_R053_T20MPA_20200803T165642.SAFE/GRANULE/L2A_T20MPA_A017811_20200803T142734/IMG_DATA/R20m/"))
win = Window(0, 0, 3000, 3000)

with ExitStack() as stack_files:

    ras10 = [stack_files.enter_context(rasterio.open(fp))
        for fp in s10.get_fpaths('B02_10m', 'B03_10m', 'B04_10m', 'B08_10m')]


    ras20 = [stack_files.enter_context(rasterio.open(fp))
        for fp in s20.get_fpaths('B05_20m', 'B06_20m', 'B07_20m', 'B8A_20m', 'B11_20m', 'B12_20m')]

    ras_final = ras10+ras20
    order = [0, 1, 2, 4, 5, 6, 3, 7, 8, 9]

    with ExitStack() as stack_action:
        rstack = Rstack()
        for idx, src in enumerate(ras_final):
            if idx > 3: # resample to 10 m
                print(f"scr to resample, res: {src.res}")
                arr_r, meta = rst.load_resample(src)
                src = stack_action.enter_context(rst.to_src(arr_r, meta))
                print(f"scr resampled with res: {src.res}")
            arr, meta = rst.load_window(src, win)
            src = stack_action.enter_context(rst.to_src(arr, meta))
            print(f"scr to add to the stack with res: {src.res}")
            rstack.add_item(src)
        rstack.set_metadata_param('interleave', 'band')
        rstack.reorder_items(order)
        fpath = rst.write_raster(rstack.items, rstack.metadata_collect, os.path.join(s10.dirpath, "S2B_T20MPA_20200803_Subset_med.tif"))
    print(fpath)

