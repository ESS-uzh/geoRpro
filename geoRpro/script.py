import os
import logging
from contextlib import contextmanager
from contextlib import ExitStack

import geopandas as gpd
import rasterio
from rasterio.windows import Window
import numpy as np

#from geoRpro.utils import write_rasters_as_stack
from geoRpro.sent2 import Sentinel2
from geoRpro.raster import RArr, to_src
from geoRpro.rcollect import GRcollect
from geoRpro.extract import DataExtractor


import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


# * some raster routine examples


def _gen_fname(inp_fpath, append):
    inp_fname = os.path.basename(inp_fpath)
    inp_name = '_'.join(inp_fname.split('.')[0].split('_')[:3])
    return inp_name + append


def create_mask(fpath):
    with rasterio.open(fpath) as src:
        r = RArr.load(src)
        r.mask_vals([3,7,8,9,10])
        return r.arr

def create_mask_resample(fpath):
    with rasterio.open(fpath) as src:
        r = RArr.load_resample(src)
        r.mask_vals([3,7,8,9,10])
        return r

def create_ndvi(fpath_red, fpath_nir):
    with ExitStack() as stack_files:
        red_src = stack_files.enter_context(rasterio.open(fpath_red))
        nir_src = stack_files.enter_context(rasterio.open(fpath_nir))
        r = RArr.load_ndvi(red_src, nir_src)
        return r

def get_datasets_low_cc(basedir, threshold):

    dirpaths_20m = [x[0] for x in os.walk(basedir) if 'GRANULE' and 'IMG_DATA' and 'R20m' in x[0]]
    sc_20m = [Sentinel2(dp) for dp in dirpaths_20m]

    low_cc_datasets = []
    for s in sc_20m:
        mask = create_mask(s.get_fpaths('SCL_20m')[0])
        not_masked = mask.count()
        masked = np.ma.count_masked(mask)
        perc = masked / (masked+not_masked)

        if perc <= threshold:
            #logger.info('Added: {}, Masked portion: {}'.format(_dir, perc))
            dirpath = s.dirpath.replace("/R20m", "")
            low_cc_datasets.append((dirpath, perc))

    return low_cc_datasets


def create_raster_stack(imgdir, gdf):
    dirpath_10m = os.path.join(imgdir, 'R10m')
    dirpath_20m = os.path.join(imgdir, 'R20m')

    s10 = Sentinel2(dirpath_10m)
    s20 = Sentinel2(dirpath_20m)

    red_fp = s10.get_fpaths('B04_10m')[0]
    nir_fp = s10.get_fpaths('B08_10m')[0]

    ndvi = create_ndvi(red_fp, nir_fp)
    mask = create_mask_resample(s20.get_fpaths('SCL_20m')[0])
    with ExitStack() as stack_files:
        gr_collect = GRcollect([stack_files.enter_context(rasterio.open(fp)) 
            for fp in s10.get_fpaths('B02_10m', 'B03_10m', 'B04_10m', 'B08_10m')])

        with ExitStack() as stack_action:
            src_ndvi = stack_action.enter_context(to_src(ndvi.arr, ndvi.meta)) 
            src_mask = stack_action.enter_context(to_src(mask.arr, mask.meta)) 
            gr_collect.add_item(src_ndvi)

            for p, m in gr_collect.get_patches(1000, src_mask, 0):
                src_p = stack_action.enter_context(to_src(p, m))
                data = DataExtractor.extract(src_p, gdf, mask_value=0)
                pdb.set_trace()
                print(data)









if __name__ == "__main__":

    INDIR = "/mnt/task1/output/ESSCharcoal_from201908_to202002_out/sent2_level2A"
    OUTDIR = "/home/diego/work/dev/ess_diego/github/goeRpro_out"
    basedir = '/home/diego/work/dev/data/Hanneke'
    shape_dir = os.path.join(basedir,'shapes_22092020/final_datapoints_22092020')
    gdf = gpd.read_file(os.path.join(shape_dir, 'final_datapoints_22092020.shp'))


    def wf1():
        #dataset = get_datasets_low_cc(INDIR, 0.40)
        dataset = '/mnt/task1/output/ESSCharcoal_from201908_to202002_out/sent2_level2A/S2B_MSIL2A_20190812T073619_N9999_R092_T37MBN_20200218T143156.SAFE/GRANULE/L2A_T37MBN_A012702_20190812T075555/IMG_DATA'
        create_raster_stack(dataset, gdf)


    wf1()
