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
            if self.window:
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
    INSTRUCTIONS = {'Input', 'Data_Dir', 'Satellite', 'Sat_Dir'}

    def __init__(self, instructions):
        self.instructions = copy.deepcopy(instructions)
        self.input = {}
        self.output = {}
        self._parse_instructions()
        self._get_fpaths()
        self.metadata = None

    def _parse_instructions(self):

        for k, _ in self.instructions.items():

            if k not in self.INSTRUCTIONS:
                raise ValueError(f'{k} is not a valid argument')

            ## NOte: Values inserted in data should be checked, property ?!
            self.input = self.instructions.get('Input')
            self.name = list(self.input.keys())[0]
            self.data_dir = self.instructions.get('Data_Dir')
            self.satellite = self.instructions.get('Satellite')
            self.sat_dir = self.instructions.get('Sat_Dir')


    def _get_fpaths(self):
        # mapping band with fpath using Sentinel2
        if self.satellite == 'Sentinel2' and self.sat_dir:
            s10 = Sentinel2(os.path.join(self.sat_dir, 'R10m'))
            s20 = Sentinel2(os.path.join(self.sat_dir, 'R20m'))
            s60 = Sentinel2(os.path.join(self.sat_dir, 'R60m'))

            for idx, band_name in enumerate(self.input[self.name]):
                try:
                    self.input[self.name][idx] = s10.get_fpath(band_name)
                except KeyError:
                    try:
                        self.input[self.name][idx] = s20.get_fpath(band_name)
                    except KeyError:
                        try:
                            self.input[self.name][idx] = s60.get_fpath(band_name)
                        except KeyError:
                            raise ValueError(f"Cannot find band: '{band_name}'. Please \
                                    provide valid Sentinel2 band name.")
        # mapping band with fpath using fname
        elif self.data_dir:
            for idx, file_name in enumerate(self.input[self.name]):
                self.input[self.name][idx] = os.path.join(self.data_dir, file_name)


    def _check_metadata(self):
        if self.output:
            test_src = self.output[self.name][0]
            end_srcs = self.output[self.name][1:]
        for src in end_srcs:
            if src.crs.to_epsg() != test_src.crs.to_epsg():
                raise ValueError('Raster data must have the same CRS')
            if src.width != test_src.width or src.height != test_src.height:
                raise ValueError('Raster data must have the same size')
            if src.res != test_src.res:
                raise ValueError('Raster data must have the same spacial resolution')
            self.metadata = test_src.profile
            self.metadata.update(driver='GTiff')

    def run(self):
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

class RStack(ComposeBase):

    def __init__(self, instructions):
        super().__init__(instructions)

    def run(self):
        print(f'Start stack procedure for {self.input[self.name]}')
        with ExitStack() as stack_files:

            srcs = [stack_files.enter_context(rasterio.open(v))
                    for v in self.input[self.name]]
            self.output = {self.name: srcs}
            self._check_metadata()
            self.metadata.update(count=len(self.output[self.name]))
            fpath = os.path.join(self.data_dir, self.name + '.tif')
            io.write_raster(srcs, self.metadata, fpath)
            self.output[self.name] = fpath
        print('Done stacking...')

class RNdvi(ComposeBase):

    def __init__(self, instructions):
        super().__init__(instructions)

    def run(self):
        print(f'Calculate ndvi index for {self.input[self.name]}')
        with ExitStack() as stack_files:

            srcs = [stack_files.enter_context(rasterio.open(v))
                    for v in self.input[self.name]]
            self.output = {self.name: srcs}
            self._check_metadata()
            index = rst.Indexes(self.metadata)
            arr, meta = index.ndvi(srcs[0], srcs[1])
            fpath = os.path.join(self.data_dir, self.name + '.tif')
            io.write_array_as_raster(arr, meta, fpath)
            self.output[self.name] = fpath
        print('Done calculating ndvi...')

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

#
#class RIndex(Task):
#    pass
