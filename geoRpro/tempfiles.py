import os
import json
from tempfile import mkdtemp

import rasterio
import numpy as np

#from geoRpro.raster import load_ndvi
import pdb


## * Utils to deal with large raster files


#def calc_ndvi_large_raster(finp, fout_name, fout_meta, outdir):
#    memmap_file = os.path.join(mkdtemp(), 'test.mymemmap')
#    dest_array = np.memmap(memmap_file, dtype=fout_meta["dtype"], mode='w+',
#            shape=(fout_meta["height"], fout_meta["width"]))
#
#    fout = os.path.join(outdir, fout_name)
#    with rasterio.open(fout, 'w+', **fout_meta) as src_out:
#        with rasterio.open(finp) as src_inp:
#            for ji, win in src_inp.block_windows(1):
#                #pdb.set_trace()
#                red_arr = src_inp.read(3, window=win)
#                nir_arr = src_inp.read(4, window=win)
#                ndvi = load_ndvi(red_arr, nir_arr)
#                # convert relative input window location to relative output # windowlocation
#                # using real world coordinates (bounds)
#                src_bounds = rasterio.windows.bounds(win, transform=src_inp.profile["transform"])
#                dst_window = rasterio.windows.from_bounds(*src_bounds, transform=src_out.profile["transform"])
#
#                dst_window = rasterio.windows.Window(dst_window.col_off,
#                    dst_window.row_off, dst_window.width, dst_window.height)
#
#                src_out.write(ndvi, window=dst_window)
#                print(f"Done writing window: {win}")
#
#    os.remove(memmap_file)
#    return fout
