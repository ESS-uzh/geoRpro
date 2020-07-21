import os
import json

import rasterio
import numpy as np

import geoRpro.aope as ao


## * I/O related

def write_arr_as_raster(arr, meta, fname, outdir):
    """
    Save a numpy array as geo-raster to disk
    *********
    params:
        arr ->  3D numpy array to save as raster
        meta -> metadata for the new raster
        fname -> name of the file (without ext)
        outdir -> output directory
    return:
        fpath -> full path of the new raster
    """
    assert (meta['driver'] == 'GTiff'),\
        "Please use GTiff driver to write to disk. \
    Passed {} instead.".format(meta['driver'])

    assert (arr.ndim == 3),\
        "np_array must have ndim = 3. \
Passed np_array of dimension {} instead.".format(arr.ndim)

    fpath = os.path.join(outdir, fname + '_.tiff')

    with rasterio.open(fpath, 'w', **meta) as dst:
        dst.write(arr)
    return fpath


def write_rasters_as_stack(rfiles, meta, fname, outdir, window=None, mask=None):
    """
    Stack several rasters layer on a single raster and
    save it to disk
    *********
    params:
        rfiles -> list of raster path to be stacked
        meta -> metadata of the final raster stack
        fname -> name of the final raster stack
        outdir -> output directory
        mask -> 3D mask array
    """
    assert (meta['driver'] == 'GTiff'),\
        "Please use GTiff driver to write to disk. \
    Passed {} instead.".format(meta['driver'])

    fpath = os.path.join(outdir, fname + '.tiff')

    with rasterio.open(fpath, 'w', **meta) as dst:
        for _id, fl in enumerate(rfiles, start=1):
            # open raster files one by one (fl)
            with rasterio.open(fl) as src1:
                if mask:
                    # read all layers of fl and mask them
                    arr, _ = ao.aope_apply_mask(src1.read(), mask)
                else:
                    arr = src1.read(1)
                dst.write_band(_id, arr.astype(meta['dtype']))
    return fpath



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


def json_to_disk(json_arr, name, outdir):
    """
    Write a json object to disk
    *********
    params:
        json_arr -> json object
        name -> filename to write
    """
    fname = name + '.json'
    fpath = os.path.join(outdir, fname)
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(json_arr, f)
    return fpath
