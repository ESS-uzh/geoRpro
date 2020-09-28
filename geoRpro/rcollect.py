import os
import collections
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.windows import Window

from geoRpro.sent2 import Sentinel2
import pdb

class GRcollect():
    """
    Collection of geo rasters

    Attributes
    ----------

    items: list of rasterio.DatasetReader objects
    """
    def __init__(self, items=None):

        if items is None:
            items = []
        self.items = items
        self.metadata_collect = None
        self.processing = collections.OrderedDict()
        self._gen_metadata()

    def __check_for_crs(self):
        """
        Check for CRS consistency for the collection
        """
        if not all(r.crs.to_epsg() == self.items[0].crs.to_epsg()
                   for r in self.items[1:]):
            raise ValueError("CRS of all rasters should be the same")

    def __check_for_dimensions(self):
        """
        Check for dimension consistency for the collection
        """
        if not all(r.width == self.items[0].width and r.height == self.items[0].height
                   for r in self.items[1:]):
            raise ValueError("height and width of all rasters should be the same")

    def _gen_metadata(self):
        """
        Generate metadata for the collection
        """
        if self.items:
            # copy metadata of the first item
            self.metadata_collect = self.items[0].meta
            self.__check_for_crs()
            self.metadata_collect.update(count=len(self.items))
            self.metadata_collect.update(driver='GTiff')

    def add_item(self, item):
        """
        Add a new item to the item collection and update metadata_collect

        params:
        ----------

             item :  rasterio.DatasetReader
        """
        self.items.append(item)
        self.__check_for_crs()
        self.metadata_collect.update(count=len(self.items))

    def get_patches(self, patch_size, mask_src=None, fill_value=None):
        """
        Yields patches of the entire stack as numpy array

        params:
        ----------

             patch_size :  int
                           size of the patch, e.g 500 pixels
        """
        v_split = self.items[0].height // patch_size
        h_split = self.items[0].width // patch_size

        # get patch arrays
        v_arrays = np.array_split(np.arange(self.items[0].height), v_split)
        h_arrays = np.array_split(np.arange(self.items[0].width), h_split)
        new_meta = self.metadata_collect
        for v_arr in v_arrays:
            v_start = v_arr[0]
            for h_arr in h_arrays:
                h_start = h_arr[0]
                window = Window(h_start, v_start, len(h_arr), len(v_arr))
                new_meta.update({
                'driver': 'GTiff',
                'height': window.height,
                'width': window.width,
                'transform': rasterio.windows.transform(window, self.items[0].transform)})

                arr = np.array([src.read(1, window=window) for src in self.items])
                
                if mask_src:
                    mask_win = mask_src.read(1, window=window)
                    masked_arr = np.ma.masked_array(*np.broadcast_arrays(arr, mask_win))
                    arr = np.ma.filled(masked_arr, fill_value=fill_value)
                yield arr, new_meta

    def add_processing(self, name, func, *args, **kwargs):
        """
        params:
        ---------
        name (str): name of the callback
        func (function): function to be called
        args (list): arguments to be passed to the callback
        kwargs (dict): keyword arguments to be passed to callback
        """
        self.processing[name] = (func, args, kwargs)

    def run_flow(self, patch_size , mask_src=None, fill_value=None):
        for p, m in self.get_patches(patch_size, mask_src, fill_value):
            if self.processing:
                for func, args, kwargs in self.processing.values():
                    yield func(p, m, *args, **kwargs)
            else:
                yield p, m


    def write_stack(self, fpath):
        self.__check_for_crs()
        self.__check_for_dimensions()
        with rasterio.open(fpath, 'w', **self.metadata_collect) as dst:
            for _id, src in enumerate(self.items, start=1):
                arr = src.read(1)
                dst.write_band(_id, arr.astype(self.metadata_collect['dtype']))
        return fpath



if __name__ == "__main__":

    INDIR = "/home/diego/work/dev/data/geoRpro_inp"
    OUTDIR = "/home/diego/work/dev/data/geoRpro_out"


    s10 = Sentinel2(os.path.join(INDIR, "S2A_MSIL2A_20190628T073621_N9999_R092_T37MBN_20191121T145522.SAFE/GRANULE/L2A_T37MBN_A020967_20190628T075427/IMG_DATA/R10m"))
    fpaths = s10.get_fpaths('B03_10m','B04_10m')
    srcs = [rasterio.open(f) for f in fpaths]
    #s = RStack(srcs)
    #print(s.items, s.meta)
    #s.append_src(rasterio.open(s10.get_fpaths('B08_10m')[0]))
    #print(s.items, s.meta)
    #for p in s.get_patches(500):
    #    print(p)
    #s.write_stack(os.path.join(OUTDIR,'test_stack.tif'))
