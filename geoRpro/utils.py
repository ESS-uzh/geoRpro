import numpy as np
import json


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
