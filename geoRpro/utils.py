import os
import json
from tempfile import mkdtemp

import rasterio
import numpy as np

#from geoRpro.raster import load_ndvi
import pdb
## * memfile

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



## * JSON related

# Serializing Python Objects not supported by JSON
class NumpyEncoder(json.JSONEncoder):

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.str_):
                return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

def load_json(fpath):
    """
    Load a json object from disk
    *********
    params:
        json_arr -> full path to json file
    return:
        a list of dictionaries
    """

    with open(fpath, 'r', encoding='utf-8') as f:
        json_str = json.loads(f.read())

    data = json.loads(json_str)
    return data


def to_json(datadict, encoder=json.JSONEncoder):
    """
    Serialize python objects using json. Supports
    numpy.ndarray serialization when encoder=NumpyEncoder
    *********
    params:
        datadict -> a dict with the python object to serialize
                   e.g. {'obj_name': obj, ..}
        encoder -> json Encoder. To be replaced by a different
                   encoder, e.g. NumpyEncoder for numpy.ndarray, to
                   serialize datatypes not supported by the default
    return:
        a json object
    """
    return json.dumps(datadict, cls=encoder)


def json_to_disk(json_arr, fpath):
    """
    Write a json object to disk
    *********
    params:
        json_arr -> json object
        name -> filename to write
    """
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(json_arr, f)
    return fpath

## * helpers

def gen_sublist(ls, inc):
    """
    Yield list content in blocks of size inc
    """

    start = 0
    block = inc

    while start < len(ls):
        if block > len(ls):
            block = len(ls)
        yield ls[start:block]
        start = block
        block = block + inc
