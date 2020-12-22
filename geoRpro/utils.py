import os
import json

import numpy as np

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

## * Helpers

def gen_sublist(a_list, inc):
    """
    Yield a list in blocks of length inc.

    If the length of the list divided by inc does not give a whole
    number a block of length equal to the reminder is also yielded

    >>> tuple(gen_sublist([1,2,3,4,5,6,7], 3))
    ([1, 2, 3], [4, 5, 6], [7])
    """

    start = 0
    block = inc

    while start < len(a_list):
        if block > len(a_list):
            block = len(a_list)
        yield a_list[start:block]
        start = block
        block = block + inc

def gen_current_front_pairs(a_list):
    """
    Yield a current element (elements composing the list except for
    the last one) and all its front elements in pair of two

    >>> tuple(gen_current_front_pairs([1,2,3]))
    ((1, 2), (1, 3), (2, 3))
    """
    idx_current = 0
    idx_front = 1
    while idx_current < len(a_list)-1:
        for el_front in a_list[idx_front:]:
            yield a_list[idx_current], el_front
        idx_current += 1
        idx_front += 1


if __name__ == "__main__":
    import doctest
    doctest.testmod()
