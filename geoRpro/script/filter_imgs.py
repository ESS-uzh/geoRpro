import os
import copy
import fnmatch
from contextlib import contextmanager
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
import fiona
import geopandas as gpd
import shapely
from shapely.geometry import box

from geoRpro.sent2 import Sentinel2
from geoRpro.raster import Rstack
import geoRpro.raster as rst


########################

# Script to select planet images with low cloud coverage
# that overlap with given data points

########################


# - Helpers

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def gen_fpath(dirpath, pattern):
    """
    Yield files in a dirpath
    with a file name matching a pattern
    """
    # consider first level directory only
    for f in os.listdir(dirpath):
        if fnmatch.fnmatch(f, pattern):
            yield os.path.join(dirpath, f)


def get_file_info(fname):
    """
    Parse the asset file name

    e.g. inp: 20200714_103348_ssc6_u0001_analytic.tif
         out: analytic, 20200714_103348_ssc6_u0001

    """
    info = fname.split("_")
    asset_type, other_info = info[4:], info[:4]

    # for asset_type of 2 words
    if len(asset_type) == 2:
        asset_type = ['_'.join(asset_type)]

    return asset_type[0].split(".")[0], '_'.join(other_info)

def get_udm2(fp):
    asset, info = get_file_info(os.path.basename(fp))
    dirpath = os.path.dirname(fp)
    if asset == 'analytic':
        match = ['*ortho_udm2.tif', '*[0-9]_udm2.tif']
    elif asset == 'ortho_panchromatic':
        match = ['*panchromatic_udm2.tif']
    for fp_udm2 in gen_fpath(dirpath, info+'*udm2*'):
        if [fp_udm2 for pt in match
                if fnmatch.fnmatch(os.path.basename(fp_udm2), pt)]:
            return fp_udm2
        continue

def get_low_cc(fp, threshold):
    with rasterio.open(fp) as src:
        arr, meta = rst.load_band(src, 6)
        elements, counts = np.unique(arr, return_counts=True)
        if counts.size == 1 and elements[0] == 0:
            print(f"CloudCoverage for {src.files[0]} is ~ 0 %")
            return 0.1
        if counts.size == 1 and elements[0] == 1:
            print(f"CloudCoverage for {src.files[0]} is 100 %")
            return
        elif counts.size == 2:
            cc = (counts[1] / (counts[0]+counts[1]))*100
            print(f"CloudCoverage for {src.files[0]} is {cc:.2f} %")
            if cc < threshold:
                return round(cc, 2)
            return

if __name__ == "__main__":


    SHAPES = "/home/diego/work/dev/data/planet_data/CharcoalSitesReprojected/CharcoalSitesReprojected.shp"
    INDIR = "/run/user/1000/gvfs/smb-share:server=files.geo.uzh.ch,share=shared/group/ess_charcoal/data/raw_data/planet/july2020"
    files = os.listdir(INDIR)

    # get shapes from the shapefile
    with fiona.open(SHAPES, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    count = 0
    overlap = []
    for fp in gen_fpath(INDIR, '*analytic.tif'):
        with rasterio.open(fp) as src:
            for sh in shapes:
                try:
                    out_image, out_transform = rasterio.mask.mask(src, [sh], crop=True)
                except ValueError:
                    # No overlap found
                    continue
                else:
                    print(f"!! found overlap for {fp}")
                    udm2_fp = get_udm2(fp)
                    lcc = get_low_cc(udm2_fp, 5)
                    if lcc:
                        print(f"Found {src.files[0]} with cloud coverage lower than 5 %")
                        overlap.append((lcc, fp))
                        count += 1
                    break

    # sort by cloud coverage value
    overlap.sort()
    with open('overlap_july2020.txt', 'w') as f:
        for cc, item in overlap:
            fsize = sizeof_fmt(Path(item).stat().st_size)
            f.write("File: %s: Size: %s CloudCoverage: %s\n" % (os.path.basename(item), fsize, cc))

    print("")
    print(f"Total images with point data overlap found: {count}")
