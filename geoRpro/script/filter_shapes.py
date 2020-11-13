import os
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


########################

# Script to select data points contained in a polygon and matching certain criteria

# Polygons are generated from a large input raster (ndvi)
# The condition is to select points that are outside the vegetation canopy
# Selected Points and relative Polygons are saved to disk as shapefiles

########################


if __name__ == "__main__":


    SHAPES = "/home/diego/work/dev/data/planet_data/CharcoalSitesReprojected/CharcoalSitesReprojected.shp"
    fp_ndvi = "/home/diego/work/dev/data/planet_data/imgs_lcc_charcoal_overlap/20200716_104044_ssc11_u0001_analytic_ndvi.tif"

    # get shapes from the shapefile
    with fiona.open(SHAPES, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    count_taken = 0
    count_excluded = 0
    changed = 0
    final_data_points = []
    final_data_polys = []
    with rasterio.open(fp_ndvi) as src:
        for win in rst.gen_windows(src):
            bbox = rasterio.windows.bounds(win, src.transform)
            poly = box(*bbox)
            count_total = 0
            for sh in shapes:
                if Point(sh["coordinates"]).within(poly):
                    arr, meta = rst.load_window(src, win)
                    # mask out dense vegetation canopy
                    arr, meta_bin = rst.mask_cond(arr, meta, arr > 0.3)
                    arr = rst.apply_mask(arr, arr.mask)
                    with rst.to_src(arr, meta) as arr_src:
                        out_image, out_transform = rasterio.mask.mask(arr_src, [sh], crop=True)
                        print(f"Extracted value: {out_image}")
                        if out_image:
                            print(f"!! found data point {sh} outside vegetation, taken")
                            final_data_points.append(Point(sh["coordinates"]))
                            count_taken += 1
                        else:
                            print(f"found data point {sh} inside vegetation, excluded")
                            count_excluded += 1
                count_total += 1
            if count_taken > changed:
                final_data_polys.append(poly)
                changed = count_taken

    print(f"Total data points: {count_total}")
    print(f"Total data points within raster found: {count_taken+count_excluded}")
    print(f"Total data points within raster taken: {count_taken}")
    print(f"Total polygons taken: {len(final_data_polys)}")
    print(f"Total data points within raster excluded: {count_excluded}")
    gdf1 = gpd.GeoDataFrame({"geometry": final_data_points}, crs=f"EPSG:{src.crs.to_epsg()}")
    gdf2 = gpd.GeoDataFrame({"geometry": final_data_polys}, crs=f"EPSG:{src.crs.to_epsg()}")
    gdf1_fname = os.path.basename(fp_ndvi).split(".")[0]+"_points"
    gdf2_fname = os.path.basename(fp_ndvi).split(".")[0]+"_polys"
    gdf1.to_file(os.path.join(os.path.dirname(fp_ndvi), gdf1_fname))
    gdf2.to_file(os.path.join(os.path.dirname(fp_ndvi), gdf2_fname))
    print("done")
