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
    OUTFOLDER_COMMUNE_NAME = "overlap7tiles"

    #ras = [fp for fp in glob.glob(os.path.join(BASEDIR, "*03ndvi*/ALPHA/Shannon_10_Fullres"))]
    ras = [fp for fp in glob.glob(os.path.join(BASEDIR, "*03ndvi*/BETA/BetaDiversity_BCdiss_PCO_10"))]

    with ExitStack() as stack_files:
        srcs = [stack_files.enter_context(rasterio.open(rpath)) for rpath in ras]
        polys = [box(*src.bounds) for src in srcs]

        intersects = []

        for poly, poly_front in ut.gen_current_front_pairs(polys):
            if poly.intersects(poly_front) and poly.intersection(poly_front) not in intersects:
                intersects.append(poly.intersection(poly_front))
        print(intersects)

        # save a polygon
        #gdf1 = gpd.GeoDataFrame({"geometry": intersects}, crs=f"EPSG:{srcs[0].crs.to_epsg()}")
        #gdf1.to_file(os.path.join(BASEDIR, "overlap_polygon"))

        for src in srcs:
            print("")
            print(f"Run for raster: {src.files[0]} ..")
            for idx, intersect in enumerate(intersects, start=1):
                try:
                    arr, meta = rst.load_raster_from_poly(src, intersect)
                    print(f"Overlap found, getting arr from poly: {idx}")
                except ValueError:
                    print(f"No overlap found for poly: {idx}")
                    continue
                else:
                    fname = "_".join([os.path.dirname(src.files[0]).split("/")[-1].split("_")[0], f"poly_{idx}", str(arr.shape[1]), str(arr.shape[2])]) + ".tif"
                    print(fname)
                    print(arr.shape)
                    meta.update({
                    "interleave": "band"})
                    OUTDIR = os.path.join(os.path.dirname(src.files[0]), OUTFOLDER_COMMUNE_NAME)
                    if not os.path.exists(OUTDIR):
                        os.makedirs(OUTDIR)
                    rst.write_array_as_raster(arr, meta, os.path.join(OUTDIR, fname))
