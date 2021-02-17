import os
import ntpath
import fnmatch
from contextlib import ExitStack

import rasterio
import numpy as np
from sklearn.metrics import r2_score

import geoRpro.raster as rst

# * Functions to perform adjacent tiles calibration 

import logging
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calc_diff_arr(arr1, arr2):
    """
    Calculate absolute difference between two arrays. The array are min-max
    normalized first.

    Return normalizead arrays and thier absolute difference
    """
    arr1 = np.nan_to_num(arr1, nan=0)
    # use min-max normalization to make it comparable
    arr1_norm = (arr1 - arr1.min()) / (arr1.max() - arr1.min())

    arr2 = np.nan_to_num(arr2, nan=0)
    arr2_norm = (arr2 - arr2.min()) / (arr2.max() - arr2.min())

    diff = abs(arr1_norm - arr2_norm)

    return (arr1_norm, arr2_norm, diff)


def polyfit(x, y, degree=1):
    """
    Fit a polynomial p(x) = p[0] * x**deg + ... + p[deg] of degree
    to points (x, y). Returns a vector of coefficients p that minimises
    the squared error.

    x : 1D numpy array
        x values

    y : 1D numpy array
        y values

    degree : int
             Degree of the fitting polynomial
    """
    return np.polyfit(x, y, degree)


def calc_r_squared(model, x, y):
    """
    Return a r squared error between real and predicted values

    model: 1D numpy array
           vector of coefficients returned by the polyfit func

    x : 1D numpy array
        x values, used to make the prediciton based on the model

    y : 1D numpy array
        real y values
    """

    predict = np.poly1d(model)
    return r2_score(y, predict(x))


def build_a_line(model, x):
    """
    Return a line equation using a linear model

    model: 1D numpy array
           (m, b)
    x : 1D numpy array


    """
    return model[0]*x - model[1]

def gen_bands(src_1, src_2):
    """
    Yield all the bands of src_1 and src_2 as arrays
    """
    for i in range(1, src_1.count+1):
        arr1, _= rst.load_bands(src_1, [i])
        arr2, _ = rst.load_bands(src_2, [i])
        yield arr1, arr2


def array_correction(model, arr_y, arr_x):
    """
    Correct the arr_y based on a linear model
    """
    # correct arr_y using the model params and arr_x as x
    arr_y_corr = model[0]*arr_x + model[1]
    # check whether correction has produced negative values
    neg_row_idx, neg_col_idx  = np.where(arr_y_corr[0,:,:] < 0)
    if neg_row_idx.size != 0:
        # set negative values to the orifinal arr_y values
        for r, c in zip(neg_row_idx, neg_col_idx):
            arr_y_corr[0,r,c] = arr_y[0,r,c]

    return arr_y_corr.astype('uint16')



def bands_correlation(src_1, src_2, threshold=0.005):
    """
    Find the correlation between the bands of two overlapping
    polygons using a linear model, i.e. src_1 and src_2.

    Compute the diff array between two bands and find its mean value.
    If the mean values of the diff array is higher than the threshold the
    scr_2 band is corrected using array_correction(), then a new diff array,
    i.e. diff_corr is computed.

    Return a dictionary mapping the bands that need correction and their
    model params

    """
    corrections = {}
    for idx, arrays in enumerate(gen_bands(src_1, src_2), start=1):
        model = polyfit(x=arrays[0].flatten(), y=arrays[1].flatten())
        print(f'Band: {idx}')
        print(f'Model params: {model}')
        r2_err = calc_r_squared(model=model, x=arrays[0].flatten(), y=arrays[1].flatten())
        print(f'R^2: {r2_err}')
        # Calc difference between arrays
        _, _, diff = calc_diff_arr(arrays[0], arrays[1])
        print(f'diff mean: {round(np.mean(diff), 5)}')
        if round(np.mean(diff), 5) > threshold:
            array_1_corr = array_correction(model, arrays[1], arrays[0])
            _, _, diff_corr = calc_diff_arr(arrays[0], array_1_corr)
            print(f'diff_corr mean: {round(np.mean(diff_corr), 5)}')
            corrections[idx] = model
        print('')
        print('')

    return corrections


def apply_band_correction(src_ref, src_target, model, band=1):
    """
    Apply a correction to an entire band
    """
    array_corrected = np.empty((1, src_target.height, src_target.width), dtype=np.int64)

    for idx, w in enumerate(rst.gen_windows(src_ref), start=1):
        patch_ref, _ = rst.load_window(src_ref, window=w, indexes=[band])
        patch_target, _ = rst.load_window(src_target, window=w, indexes=[band])
        patch_target_corrected = array_correction(model, patch_target, patch_ref)
        # fill in the array_corrected with corrected values
        array_corrected[0, w.row_off:w.row_off+w.height, w.col_off:w.col_off+w.width] = patch_target_corrected[0,:,:]
    return array_corrected


def apply_correlation_correction(src_ref, src_target, corrections, outdir):
    """
    Create a corrected src_target stack. All the bands of src_target stack
    that appear in the correction param are corrected using the respective model
    params
    """
    print(f'Apply correction to {src_target.files[0]}')
    with ExitStack() as stack_action:
        rstack = rst.Rstack()
        for index in range(1, src_ref.count+1):
            print(f'Process band: {index}')
            array_target, _ = rst.load_bands(src_target, [index])
            for band, model in corrections.items():
                if index == band:
                    print(f'Band: {index} will be corrected..')
                    array_target = apply_band_correction(src_ref, src_target, model, band)
                    break
            band_metadata = src_target.profile
            band_metadata.update({'count': 1})
            src_target_corrected = stack_action.enter_context(rst.to_src(array_target,
                band_metadata))
            rstack.add_item(src_target_corrected)
        rstack.set_metadata_param('interleave', 'band')
        fname = '_'.join([ntpath.basename(src_target.files[0]).split('.')[0], 'corr']) + '.tif'
        fpath = rst.write_raster(rstack.items, rstack.metadata_collect, os.path.join(outdir, fname))
    return fpath



if __name__ == "__main__":

    # get file paths
    stacks_dir = "/home/diego/work/dev/data/amazon/20200729_stacks"
    overlaps_dir = "/home/diego/work/dev/data/amazon/20200729_stacks/overlaps"

    ref = rasterio.open(os.path.join(stacks_dir, 'T20MPA_20200729.tif'))

    polys_targets_fname = [f for f in os.listdir(overlaps_dir) if
                 os.path.isfile(os.path.join(overlaps_dir, f)) and not fnmatch.fnmatch(f, '*T20MPA*')]
    polys_targets_fname.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    polys_refs_fname = [f for f in os.listdir(overlaps_dir) if
                 os.path.isfile(os.path.join(overlaps_dir, f)) and fnmatch.fnmatch(f, '*T20MPA*')]
    polys_refs_fname.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    targets_fname = [f for f in os.listdir(stacks_dir) if
                   os.path.isfile(os.path.join(stacks_dir, f)) and f != 'T20MPA_20200729.tif']


    for poly_ref_fname, poly_target_fname in zip(polys_refs_fname, polys_targets_fname):

        print(f'Start polygon {poly_target_fname}')

        poly_ref = rasterio.open(os.path.join(overlaps_dir, poly_ref_fname))
        poly_target = rasterio.open(os.path.join(overlaps_dir, poly_target_fname))
        target_tile_name = '*'+poly_target_fname.split('.')[0].split('_')[-1]+'*'
        target_fname = [f for f in targets_fname if fnmatch.fnmatch(f, target_tile_name)][0]
        target = rasterio.open(os.path.join(stacks_dir, target_fname))

        band_corrections = bands_correlation(poly_ref, poly_target)
        print(band_corrections)
        if target_fname == 'T20MPB_20200729.tif':
            apply_correlation_correction(ref, target, band_corrections, stacks_dir)

