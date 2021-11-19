import os
import sys
import copy
import math
import json
import logging
import collections
from contextlib import contextmanager
from contextlib import ExitStack

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from rasterio.windows import Window
from rasterio.warp import Resampling
from geoRpro.sent2 import Sentinel2
import geoRpro.raster as rst
import geoRpro.io as io

import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

##### Test to restructure processing.py ###
##### Work in progress...

class Action:
    PREPROCESS = {'Res', 'Window', 'Polygon', 'Layer'}

    def __init__(self, inputs, indir, outdir, prep=None, satellite=None):
        self.inputs = inputs
        self.outputs = None
        self.indir = indir
        self.outdir = outdir
        self.prep = prep
        self.satellite = satellite
        self._get_fpaths()
        self.res = None
        self._polygon = None
        self.window = None
        self.layer = None

    @property
    def polygon(self):
        return self._polygon

    @polygon.setter
    def polygon(self, poly_param):

        shape_file = poly_param[0]
        crs = poly_param[1]
        index = poly_param[2]

        gdf = gpd.read_file(shape_file)
        gdf = gdf.to_crs(f"EPSG:{crs}")
        self._polygon = gdf['geometry'][index]

    def _parse_pre_process(self):
        for k, _ in self.pre_process.items():

            if k not in self.PREPROCESS:
                raise ValueError(f'{k} is not a valid argument')

            ## NOte: Values inserted in data should be checked, property ?!
            self.res = self.pre_process.get('Res')
            self.polygon = self.pre_process.get('Polygon')
            self.window = self.pre_process.get('Window')
            if self.window:
                self.window = tuple(self.window)
            self.layer = self.pre_process.get('Layer')

    def pre_process_run(self):
        self._parse_pre_process()
        pdb.set_trace()
        with ExitStack() as stack_files:
            self.outputs = {k:stack_files.enter_context(rasterio.open(v))
                            for (k, v) in self.inputs.items()}

            with ExitStack() as stack_action:

                for name, src in self.outputs.items():

                    print(f'Raster: {name} has {src.res[0]} m resolution..')

                    if self.res and src.res[0] != self.res: # resample to match res param
                        print(f'Raster: {name} will be resampled to {self.res} m resolution..')
                        scale_factor = src.res[0] / self.res
                        arr, meta = rst.load_resample(src, scale_factor)
                        src = stack_action.enter_context(io.to_src(arr, meta))

                    if self.layer:
                        print(f'Selected layer: {self.layer}')
                        arr, meta = rst.load(src, bands=self.layer)
                        src = stack_action.enter_context(io.to_src(arr, meta))

                    if self.polygon:
                        print(f"Selected a polygon as AOI")
                        arr, meta = rst.load_polygon(src, self.polygon)
                        src = stack_action.enter_context(io.to_src(arr, meta))

                    elif self.window:
                        print(f'Selected a window: {self.window} as AOI')
                        arr, meta = rst.load(src, window=self.window)
                        src = stack_action.enter_context(io.to_src(arr, meta))

                    if not os.path.exists(self.indir):
                        print(f'Creating {self.indir}')
                        os.makedirs(self.indir)

                    fpath = os.path.join(self.indir, name + '.tif')
                    io.write_raster([src], src.profile, fpath)
                    self.outputs[name] = fpath

    def _get_fpaths(self):

        # mapping band with fpath using Sentinel2 parser
        if self.satellite == 'Sentinel2':

            s10 = Sentinel2(os.path.join(self.indir, 'R10m'))
            s20 = Sentinel2(os.path.join(self.indir, 'R20m'))
            s60 = Sentinel2(os.path.join(self.indir, 'R60m'))

            for name, band_name in self.inputs.items():
                try:
                    self.inputs[name] = s10.get_fpath(band_name)
                except KeyError:
                    try:
                        self.inputs[name] = s20.get_fpath(band_name)
                    except KeyError:
                        try:
                            self.inputs[name] = s60.get_fpath(band_name)
                        except KeyError:
                            raise ValueError(f"Cannot find band: '{band_name}'. Please \
                                    provide valid Sentinel2 band name.")

        # mapping band with fpath using fname
        else:
            for name, file_name in self.inputs.items():
                self.inputs[name] = os.path.join(self.indir, file_name)



if __name__ == '__main__':

    ORIGIN = '/home/diego/work/dev/data/test_data/S2A_MSIL2A_20190906T073611_N0213_R092_T37MBN_20190906T110000.SAFE/GRANULE/L2A_T37MBN_A021968_20190906T075543/IMG_DATA'
    SHAPES = '/home/diego/work/dev/data/test_data/StudyVillages_Big/StudyVillages_Big.shp'
    DATADIR = '/home/diego/work/dev/github/test_data'
    WINDOW = ((10970, 10980), (5025, 5040)) # (row, col)
    RES = 10
    s20 = Sentinel2(os.path.join(ORIGIN, 'R20m'))
    import geopandas as gpd
    with rasterio.open(s20.get_fpaths('TCI_20m')[0]) as ras_tci:
        gdf_polys = gpd.read_file(SHAPES)
        gdf_polys = gdf_polys.to_crs(f"EPSG:{ras_tci.crs.to_epsg()}")

    data = {'bo2': 'B02_20m'}
    pre_pro = {'Res': 10}
    act = Action(data, ORIGIN, '/home/diego/work/dev/github/test_data', prep=pre_pro,
            satellite='Sentinel2')
    act.pre_process_run()
