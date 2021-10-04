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
from rasterio.mask import mask
from rasterio.windows import Window
from rasterio.warp import Resampling
from geoRpro.sent2 import Sentinel2
import geoRpro.raster as rst
import geoRpro.io as io

import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ProcessBase:
    SATELLITES = 'Sentinel2'
    INSTRUCTIONS = {'Inputs', 'Pre_Process', 'Data_Dir', 'Satellite', 'Sat_Dir'}
    PREPROCESS = {'Res', 'Window', 'Polygon', 'Layer'}

    def __init__(self, instructions):

        self.instructions = copy.deepcopy(instructions)
        self.inputs = {}
        self.outputs = {}
        self._parse_instructions()
        self._get_fpaths()
        self.res = None
        self.polygon = None
        self.window = None
        self.layer = None

    def _parse_instructions(self):
        for k, _ in self.instructions.items():

            if k not in self.INSTRUCTIONS:
                raise ValueError(f'{k} is not a valid argument')

            ## NOte: Values inserted in data should be checked, property ?!
            self.inputs = self.instructions.get('Inputs')
            self.pre_process = self.instructions.get('Pre_Process')
            self.data_dir = self.instructions.get('Data_Dir')
            self.satellite = self.instructions.get('Satellite')
            self.sat_dir = self.instructions.get('Sat_Dir')

    def _parse_pre_process(self):
        for k, _ in self.pre_process.items():

            if k not in self.PREPROCESS:
                raise ValueError(f'{k} is not a valid argument')

            ## NOte: Values inserted in data should be checked, property ?!
            self.res = self.pre_process.get('Res')
            self.polygon = self.pre_process.get('Polygon')
            self.window = tuple(self.pre_process.get('Window'))
            self.layer = self.pre_process.get('Layer')

    def pre_process_run(self):
        self._parse_pre_process()
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

                    fpath = os.path.join(self.data_dir, name + '.tif')
                    io.write_raster([src], src.profile, fpath)
                    self.outputs[name] = fpath

    def _get_fpaths(self):

        # mapping band with fpath using Sentinel2
        if self.satellite == 'Sentinel2' and self.sat_dir:

            s10 = Sentinel2(os.path.join(self.sat_dir, 'R10m'))
            s20 = Sentinel2(os.path.join(self.sat_dir, 'R20m'))
            s60 = Sentinel2(os.path.join(self.sat_dir, 'R60m'))

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
        elif self.data_dir:
            for name, file_name in self.inputs.items():
                self.inputs[name] = os.path.join(self.data_dir, file_name)

    def run(self):
        pass


class ComposeBase:
    pass


class RProcess(ProcessBase):

    def __init__(self, instructions):
        super().__init__(instructions)

    def run(self):
        if self.pre_process:
            print(f'Start pre-processing for: {list(self.inputs.keys())}')
            self.pre_process_run()
        print('Done pre-processing...')


class RMask(ProcessBase):
    INSTRUCTIONS = ProcessBase.INSTRUCTIONS
    INSTRUCTIONS.add('Mask')

    def __init__(self, instructions):
        super().__init__(instructions)
        self.mask_val = self.instructions.get('Mask')

    def run(self):
        if self.pre_process:
            print('Pre-processing ...')
            self.pre_process_run()
            print('Done')
            self.inputs = self.outputs
            self.outputs = {}
        with ExitStack() as stack_files:
            self.outputs = {k:stack_files.enter_context(rasterio.open(v))
                            for (k, v) in self.inputs.items()}

            for name, src in self.outputs.items():
                print(f'Masking: {name}')
                arr, meta = rst.load(src)
                arr_mask, meta_mask = rst.mask_vals(arr, meta, self.mask_val)
                fpath = os.path.join(self.data_dir, name + '.tif')
                io.write_array_as_raster(arr_mask.mask, meta_mask, fpath)
                self.outputs[name] = fpath


class RReplace(ProcessBase):
    INSTRUCTIONS = ProcessBase.INSTRUCTIONS
    INSTRUCTIONS.add('Replace')

    def __init__(self, instructions):
        super().__init__(instructions)
        self.replace = self.instructions.get('Replace')

    def run(self):
        if self.pre_process:
            print('Pre-processing ...')
            self.pre_process_run()
            print('Done')
            self.inputs = self.outputs
            self.outputs = {}
        with ExitStack() as stack_files:
            self.outputs = {k:stack_files.enter_context(rasterio.open(v))
                            for (k, v) in self.inputs.items()}

            for name, src in self.outputs.items():
                print(f"Replace values at mask array position")
                # check that mask and array are the same dimension
                arr, meta = rst.load(src)
                with rasterio.open(os.path.join(self.data_dir, self.replace[0])) as src_mask:
                    mask, _ = rst.load(src_mask)
                    assert mask.shape == arr.shape, 'Array and mask must the have same shape'
                    arr_replaced = rst.apply_mask(arr, mask, fill_value=self.replace[1])
                fpath = os.path.join(self.data_dir, name + '.tif')
                io.write_array_as_raster(arr_replaced, meta, fpath)
                self.outputs[name] = fpath

if __name__ == '__main__':

    ORIGIN = '/home/diego/work/dev/data/test_data/S2A_MSIL2A_20190906T073611_N0213_R092_T37MBN_20190906T110000.SAFE/GRANULE/L2A_T37MBN_A021968_20190906T075543/IMG_DATA'
    DATADIR = '/home/diego/work/dev/github/test_data'
    WINDOW = ((10970, 10980), (5025, 5040)) # (row, col)
    RES = 10

    pr1_instructions = {'Inputs': {'scl': 'SCL_20m', 'b11': 'B11_20m'},
                        'Pre_Process': {'Res': RES},
                        'Data_Dir': DATADIR,
                        'Satellite': 'Sentinel2',
                        'Sat_Dir': ORIGIN}

    mask1_instructions = {'Inputs': {'scl_mask': 'scl.tif'},
                          'Data_Dir': DATADIR,
                          'Mask': (3, 7, 8, 9, 10)}

    replace1_instructions = {'Inputs': {'b11_replaced': 'b11.tif'},
                             'Data_Dir': DATADIR,
                             'Replace': ('scl_mask.tif', 9999)}


    pr1 = RProcess(pr1_instructions)
    mask1 = RMask(mask1_instructions)
    replace1 = RReplace(replace1_instructions)
    pr1.run()
    mask1.run()
    replace1.run()

    scl = rasterio.open(pr1.outputs['scl'])
    scl_mask = rasterio.open(mask1.outputs['scl_mask'])
    b11_replaced = rasterio.open(replace1.outputs['b11_replaced'])

    scl_arr, meta_scl = rst.load(scl)
    scl_mask_arr, meta_scl_mask = rst.load(scl_mask)
    b11, meta_b11 = rst.load(b11_replaced)

#class RStack(ComposeBase):
#    ALLOWED = ('Inputs', 'Replace')
#
#    def __init__(self, instructions, data_dir, satellite=None, sat_dir=None,
#            metadata=None):
#        super().__init__(instructions, data_dir, satellite, sat_dir)
#        self.metadata = metadata
#
#    def _parse_instructions(self):
#
#        for k, _ in self.instructions.items():
#
#            if k not in self.ALLOWED:
#                raise ValueError(f'{k} is not a valid argument')
#
#            ## NOte: Values inserted in data should be checked, property ?!
#            self.inputs = self.instructions.get('Inputs')
#            self.replace_val = self.instructions.get('Replace')
#
#    def _get_fpaths(self):
#        # mapping band with fpath using Sentinel2
#        if self.satellite == 'Sentinel2' and self.sat_dir:
#            s10 = Sentinel2(os.path.join(self.sat_dir, 'R10m'))
#            s20 = Sentinel2(os.path.join(self.sat_dir, 'R20m'))
#            s60 = Sentinel2(os.path.join(self.sat_dir, 'R60m'))
#
#            for name, bands in self.inputs.items():
#                for idx, band_name in enumerate(bands):
#                    try:
#                        self.inputs[name][idx] = s10.get_fpath(band_name)
#                    except KeyError:
#                        try:
#                            self.inputs[name][idx] = s20.get_fpath(band_name)
#                        except KeyError:
#                            try:
#                                self.inputs[name][idx] = s60.get_fpath(band_name)
#                            except KeyError:
#                                raise ValueError(f"Cannot find band: '{band_name}'. Please \
#                                        provide valid Sentinel2 band name.")
#
#        # mapping band with fpath using fname
#        elif self.data_dir:
#            for name, file_names in self.inputs.items():
#                for idx, file_name in enumerate(bands):
#                    self.inputs[name][idx] = os.path.join(self.data_dir, file_name)
#
#
#    def _check_metadata(self):
#        for name, srcs in self.outputs.items():
#            test_scr = srcs[0]
#            for idx, src in srcs[1:]:
#                if src.crs.to_epsg() != test_src.crs.to_epsg():
#                    raise ValueError('Raster data must have the same CRS')
#                if src.width != test_src.width or src.heigth != test_src.heigth:
#                    raise ValueError('Raster data must have the same size')
#                if src.res != test_src.res:
#                    raise ValueError('Raster data must have the same spacial resolution')
#        self.metadata = self.outputs[test_name].profile
#        self.metadata.update(count=len(self.outputs.keys()))
#        self.metadata.update(driver='GTiff')
#
#    def run(self):
#
#        srcs = []
#        with ExitStack() as stack_files:
#            for name, fpaths in self.inputs.items():
#                for idx, fpath in enumerate(fpaths):
#                    self.outputs[name][idx] = rasterio.open(fpath)
#
#            self._check_metadata()
#
#            with ExitStack() as stack_action:
#
#                for name, src in self.outputs.items():
#
#                    if self.replace_val:
#                        print(f"Replace values at mask array position")
#                        # check that mask and array are the same dimension
#                        arr, meta = rst.load(src)
#                        with rasterio.open(self.replace_val[0]) as src_mask:
#                            mask, _ = rst.load(src_mask)
#                        assert mask.shape == arr.shape, 'Array and mask must the have same shape'
#                        arr = rst.apply_mask(arr, mask, fill_value=self.replace_val[1])
#                        src = stack_action.enter_context(io.to_src(arr, meta))
#                    srcs.append(src)
#                fpath = os.path.join(self.data_dir, name + '.tif')
#                io.write_raster(srcs, metadata, fpath)
#                self.outputs[name] = fpath
#
#class RIndex(Task):
#    pass
