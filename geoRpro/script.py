import os
import logging

import geopandas as gpd
import rasterio
from rasterio.windows import Window
import numpy as np

import geoRpro.rope as ro
import geoRpro.aope as ao
from geoRpro.utils import write_rasters_as_stack
from geoRpro.sent2 import Sentinel2

import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


# * some raster routine examples


def _gen_fname(inp_fpath, append):
    inp_fname = os.path.basename(inp_fpath)
    inp_name = '_'.join(inp_fname.split('.')[0].split('_')[:3])
    return inp_name + append

def gen_raster_mask(fpath, outdir, window):
    """resample a raster and generate a mask over an AOI"""
    with rasterio.open(fpath) as src:
        with ro.resample_raster(src) as resampled:
            with ro.get_aoi(resampled, window) as aoi:
                fname_r = _gen_fname(fpath, '_resampled_aoi.tiff')
                ro.write_raster(aoi, os.path.join(outdir, fname_r))
                with ro.create_raster_mask(aoi, [3,7,8,9,10]) as mask:
                    fname_m = _gen_fname(fpath, '_mask_aoi.tiff')
                    ro.write_raster(mask, os.path.join(outdir, fname_m))


def cal_raster_ndvi(fpathr, fpathn, outdir, fname, window):
    "calc ndvi over an AOI"
    with rasterio.open(fpathr) as src_r:
        with rasterio.open(fpathn) as src_n:
            with ro.get_aoi(src_r, window) as aoi_r:
                with ro.get_aoi(src_n, window) as aoi_n:
                    with ro.calc_ndvi(aoi_r, aoi_n) as ndvi:
                        ro.write_raster(ndvi, os.path.join(outdir, fname))


if __name__ == "__main__":

    INDIR = "/home/diego/work/dev/ess_diego/github/goeRpro_inp"
    OUTDIR = "/home/diego/work/dev/ess_diego/github/goeRpro_out"


    def wf1():
        s10 = Sentinel2(os.path.join(INDIR, "S2A_MSIL2A_20190628T073621_N9999_R092_T37MBN_20191121T145522.SAFE/GRANULE/L2A_T37MBN_A020967_20190628T075427/IMG_DATA/R10m"))
        s20 = Sentinel2(os.path.join(INDIR, "S2A_MSIL2A_20190628T073621_N9999_R092_T37MBN_20191121T145522.SAFE/GRANULE/L2A_T37MBN_A020967_20190628T075427/IMG_DATA/R20m"))

        fpath_scl = s20.get_fpaths('SCL_20m')[0]
        # resample and generate mask over AOI
        gen_raster_mask(fpath_scl, OUTDIR, Window(0,0,4000,2000))

        # calc ndvi over AOI
        cal_raster_ndvi(s10.get_fpaths('B04_10m')[0], s10.get_fpaths('B08_10m')[0], OUTDIR, 'ndvi_aoi.tiff', Window(0,0,4000,2000))

        # mask sent2 bands over AOI
        with rasterio.open(Sentinel2(OUTDIR).get_fpaths('SCL_mask_aoi')[0]) as src_mask:
            mask = src_mask.read()
            rfiles = s10.get_fpaths('B02_10m','B03_10m', 'B04_10m', 'B08_10m')
            rfiles.append(os.path.join(OUTDIR, 'ndvi_aoi.tiff'))
            for i,r in enumerate(rfiles):
                with rasterio.open(r) as src:
                    fname_f_masked = _gen_fname(r, '_masked_aoi.tiff')
                    with ro.get_aoi(src, Window(0,0,4000,2000)) as aoi:
                        with ro.apply_raster_mask(aoi, mask, -9999) as masked:
                            ro.write_raster(masked, os.path.join(OUTDIR,fname_f_masked))

        # stack sent2 bands and ndvi
        s10_m = Sentinel2(OUTDIR)
        rfiles_m = s10_m.get_fpaths('B02_masked_aoi','B03_masked_aoi', 'B04_masked_aoi', 'B08_masked_aoi')
        rfiles_m.append(os.path.join(OUTDIR, 'ndvi_aoi.tiff'))
        
        with rasterio.open(rfiles_m[0]) as first:
            meta = first.meta
            meta.update(count = len(rfiles_m))
            meta.update(driver = 'GTiff')
        #pdb.set_trace()
        write_rasters_as_stack(rfiles_m, meta, 'stack', OUTDIR)


    wf1()
