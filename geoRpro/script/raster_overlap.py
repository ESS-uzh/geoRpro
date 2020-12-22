import os
import glob
import copy
import fnmatch
from contextlib import contextmanager
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
import fiona
from shapely.geometry import box, Point
import geopandas as gpd

import geoRpro.raster as rst
import geoRpro.utils as ut
import pdb

########################

# Select a ROI defined as the overlap between two rasters and save it to disk

########################


if __name__ == "__main__":

    BASEDIR = "/home/diego/work/dev/data/amazon/bioDiv_out/"

    ras = [fp for fp in glob.glob(os.path.join(BASEDIR, "*_no_rad_filtering/BetaDiversity_BCdiss_PCO_10"))]

    with ExitStack() as stack_files:
        srcs = [stack_files.enter_context(rasterio.open(rpath)) for rpath in ras]
        polys = [box(*src.bounds) for src in srcs]

        intersects = []

        for poly, poly_front in ut.gen_current_front_pairs(polys):
            if poly.intersects(poly_front) and poly.intersection(poly_front) not in intersects:
                intersects.append(poly.intersection(poly_front))
        print(intersects)

        for src in srcs:
            for idx, intersect in enumerate(intersects, start=1):
                try:
                    arr, meta = rst.load_raster_from_poly(src, intersect)
                except ValueError:
                    print(f"No overlap found for {src.files[0]} and {intersect} Try the next one ..")
                    continue
                else:
                    fname = "_".join([os.path.dirname(src.files[0]).split("/")[-1].split("_")[0], "BetaDiversity_overlap", str(idx)]) + ".tif"
                    print(fname)
                    print(arr.shape)
                    meta.update({
                    "interleave": "band"})
                    rst.write_array_as_raster(arr, meta, os.path.join(os.path.dirname(src.files[0]), fname))
