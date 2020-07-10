import numpy as np


def aope_ndvi(red_arr, nir_arr, meta):
    """
    Calc ndvi array
    *********

    params:
        red_arr, nir_arr -> 2D numpy arrays (width, heigh)
        meta -> metadata associated with one of the two arrays

    return:
        tuple of new numpy arr and relative metadata
    """
    np.seterr(divide='ignore', invalid='ignore')
    ndvi_arr = (nir_arr.astype(np.float32)-red_arr.astype(np.float32))/ \
               (nir_arr.astype(np.float32)+red_arr.astype(np.float32))

    # updata metadata
    meta.update(count=1, dtype="float32", driver='GTiff')
    return ndvi_arr, meta


def aope_mask(vals, t_arr, meta):
    """
    Binarize a numpy array based on vals
    (0=not masked, 1=masked)

    *********
    params:
        vals -> a list of values
        t_arr -> numpy arrays (bands, width, heigh)
        meta -> metadata associated with the target arrays

    return:
        tuple of new numpy arr and relative metadata
    """
    # create a mask array
    mask_arr = np.ma.MaskedArray(t_arr, np.in1d(t_arr, vals))
    # update metadata
    meta.update(driver='GTiff', nbits=1)
    return mask_arr, meta


def aope_apply_mask(t_arr, mask_arr, fill_value=0):
    """
    Apply a mask array on a target array

    *********

    params:
        vals -> a list of values
        t_arr -> numpy arrays (bands, width, heigh)
        meta -> metadata associated with the target arrays

    return:
        tuple of new numpy arr and relative metadata
    """
    # check arr and mask have the same dim
    assert (t_arr.shape == mask_arr.shape),\
        "Array and mask must have the same dimensions!"
    masked_arr = np.ma.array(t_arr, mask=mask_arr)

    # Fill masked vales with zero !! maybe to be changed
    m_filled = np.ma.filled(masked_arr, fill_value=fill_value)
    return m_filled
