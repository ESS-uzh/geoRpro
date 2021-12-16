import os
import copy
import logging
from contextlib import ExitStack

import rasterio
import geopandas as gpd
from geoRpro.sent2 import Sentinel2
import geoRpro.raster as rst
import geoRpro.io as io

import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ProcessBase:
    """
    Base class to process geo-rasters. It sets basic attributes used by
    child classes

    Attributes
    ----------

    instructions: dict

    inputs: dict
            {"fname": "fpath", .. }

    indir: str
           Full path to input directory

    outdir: str (default: None)
            Full path to output directory. When not specify indir will be working
            as outdir

    satellite: str (default: None)
            Supports only Sentinel2

    """

    def __init__(self, instructions):
        self.instructions = copy.deepcopy(instructions)
        self.parse_instructions()
        self.get_fpaths()
        self.outputs = None

    def parse_instructions(self):
        for k, _ in self.instructions.items():

            if k not in self.INSTRUCTIONS:
                raise ValueError(f'{k} is not a valid argument')

        self.inputs = self.instructions.get('Inputs')
        self.indir = self.instructions.get('Indir')
        self.outdir = self.instructions.get('Outdir')
        self.satellite = self.instructions.get('Satellite')

    def get_fpaths(self):
        """
        Re-map inputs attribute to "name": fullpath
        """

        handler_sat = self.__sent2_raster
        handler_fname = self.__fname_raster
        if isinstance(list(self.inputs.values())[0], list):
            handler_sat = self.__sent2_rasters
            handler_fname = self.__fname_rasters

        # mapping band with fpath using Sentinel2 file Parser
        if self.satellite == 'Sentinel2':

            s10 = Sentinel2(os.path.join(self.indir, 'R10m'))
            s20 = Sentinel2(os.path.join(self.indir, 'R20m'))
            s60 = Sentinel2(os.path.join(self.indir, 'R60m'))

            handler_sat(s10, s20, s60)

        # mapping band with fpath using fname
        else:
            handler_fname()

    def run(self):
        """To be implemented by child classes"""

    def _create_outdir(self):
        outdir = self.indir

        if self.outdir:
            outdir = self.outdir

        if not os.path.exists(outdir):
            print(f'Creating {outdir}')
            os.makedirs(outdir)

        return outdir

    def __sent2_rasters(self, s10, s20, s60):
        for name, values in self.inputs.items():
            for idx, band_name in enumerate(values):
                try:
                    values[idx] = s10.get_fpath(band_name)
                except KeyError:
                    try:
                        values[idx] = s20.get_fpath(band_name)
                    except KeyError:
                        try:
                            values[idx] = s60.get_fpath(band_name)
                        except KeyError:
                            raise ValueError(f"Cannot find band: '{band_name}'. Please \
                                    provide valid Sentinel2 band name.")

    def __sent2_raster(self, s10, s20, s60):
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
    def __fname_raster(self):
        for name, file_name in self.inputs.items():
            self.inputs[name] = os.path.join(self.indir, file_name)

    def __fname_rasters(self):
        for name, values in self.inputs.items():
            for idx, file_name in enumerate(values):
                values[idx] = os.path.join(self.indir, file_name)