import os
import rasterio
from contextlib import ExitStack
import geoRpro.rope as ro
import geoRpro.aope as ao
from sent2 import Sentinel2
import geopandas as gpd
import numpy as np
import pdb
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def gen_raster_mask(fpath, outdir, fname=None):
    with rasterio.open(fpath) as src:
        with ro.resample_raster(src) as resampled:
            print(resampled.meta)
            with ro.create_raster_mask(resampled, [3,7,8,9,10]) as mask:
                ro.write_raster(mask, os.path.join(outdir, fname))


def cal_raster_ndvi(fpathr, fpathn, outdir, fname=None):
    with rasterio.open(fpathr) as src_r:
        with rasterio.open(fpathn) as src_n:
            with ro.calc_ndvi(src_r, src_n) as ndvi:
                ro.write_raster(ndvi, os.path.join(outdir, fname))


def extract_from_raster(fpath, outdir, fname=None):
    with rasterio.open(fpath) as src:
        with rasterio.open(os.path.join(outdir,"scl_mask.tiff")) as src_mask:
            with ro.apply_raster_mask(src, src_mask) as masked:
                gdf  = gpd.read_file(os.path.join(INDIR, "all_points/all_points.shp"))
                ro.write_raster(masked, os.path.join(outdir,fname))
                return ro.extract_from_raster(masked, gdf)


if __name__ == "__main__":

    INDIR = "/home/diego/work/dev/ess_diego/github/goeRpro_inp"
    OUTDIR = "/home/diego/work/dev/ess_diego/github/goeRpro_out"

    def wf0():
        logger.info("Start workflow")
        p = Sentinel2(os.path.join(INDIR, "S2A_MSIL2A_20190628T073621_N9999_R092_T37MBN_20191121T145522.SAFE/GRANULE/L2A_T37MBN_A020967_20190628T075427/IMG_DATA/R10m"))
        print("generate ndvi")
        cal_raster_ndvi(p.sfpaths[2][0], p.sfpaths[3][0], OUTDIR, 'ndvi.tiff')

    def wf1():
        inpScl = os.path.join(INDIR, "T37MBN_20190628T073621_SCL_20m.jp2")
        p = Sentinel2(os.path.join(INDIR, "S2A_MSIL2A_20190628T073621_N9999_R092_T37MBN_20191121T145522.SAFE/GRANULE/L2A_T37MBN_A020967_20190628T075427/IMG_DATA/R10m"))

        print("generate a mask")
        gen_raster_mask(inpScl, OUTDIR, 'scl_mask.tiff')

        print("generate ndvi")
        cal_raster_ndvi(p.sfpaths[2][0], p.sfpaths[3][0], OUTDIR, 'ndvi.tiff')

        p.sfpaths.append((os.path.join(OUTDIR, 'ndvi.tiff'), 'ndvi'))

        features = []
        for i,r in enumerate(p.sfpaths):
            fname = r[1] + '_m.tiff'
            print(f"extract pixel values from: {r[1]}")
            X,y = extract_from_raster(r[0], OUTDIR, fname)
            print(X.shape, y.shape)
            features.append(X)
        features = np.concatenate(features, axis=1)
        print(features)


    def wf2():
        inpScl = os.path.join(INDIR, "T37MBN_20190628T073621_SCL_20m.jp2")
        p = Sentinel2(os.path.join(INDIR, "S2A_MSIL2A_20190628T073621_N9999_R092_T37MBN_20191121T145522.SAFE/GRANULE/L2A_T37MBN_A020967_20190628T075427/IMG_DATA/R10m"))

        print("generate a mask")
        gen_raster_mask(inpScl, OUTDIR, 'scl_mask.tiff')

        print("generate ndvi")
        cal_raster_ndvi(p.sfpaths[2][0], p.sfpaths[3][0], OUTDIR, 'ndvi.tiff')

        p.sfpaths.append((os.path.join(OUTDIR, 'ndvi.tiff'), 'ndvi'))

        gdf  = gpd.read_file(os.path.join(INDIR, "all_points/all_points.shp"))

        with rasterio.open(os.path.join(OUTDIR,"scl_mask.tiff")) as src_mask:
            mask = src_mask.read()
            features = []
            for i,r in enumerate(p.sfpaths):
                with rasterio.open(r[0]) as src:
                    fname = r[1] + '_m.tiff'
                    with ro.apply_raster_mask(src, mask) as masked:
                        ro.write_raster(masked, os.path.join(OUTDIR,fname))
                        print(f"extract pixel values from: {r[1]}")
                        X,y = ro.extract_from_raster(masked, gdf)
                        print(X.shape, y.shape)
                        features.append(X)
        features = np.concatenate(features, axis=1)
        print(features)

    wf0()
