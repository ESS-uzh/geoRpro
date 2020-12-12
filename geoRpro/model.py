import time
import os
import json
import sys

import joblib
from joblib import Parallel, delayed

import rasterio
from rasterio.plot import reshape_as_image
from rasterio.mask import mask
from rasterio.merge import merge
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score

from shapely.geometry import mapping
from geoRpro.utils import NumpyEncoder, to_json, json_to_disk, load_json, gen_sublist
import geoRpro.raster as rst



import logging
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_RandomForestClf(X, y, estimators):
    """
    Use train_test split to get accuacy of a Random forerst classifier
    """
    logger.info('Start to train the model...')

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = RandomForestClassifier(n_estimators=estimators)
    clf.fit(X_train, y_train)
    # Acc on X_train
    y_pred_train = clf.predict(X_train)
    logger.info(f"Train overall accuracy: {accuracy_score(y_train, y_pred_train)}")
    # Acc on X_test
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test,y_pred)
    diag = np.diagonal(cm, offset=0)
    label_ids = np.unique(y)

    logger.info(f"User accuracy:")
    for idx, row in enumerate(cm):
        user_accuracy  =  round( diag[idx] / sum(row), 2)
        logger.info(f"label_id: {label_ids[idx]}, user accuracy {user_accuracy}")

    logger.info(f"Producer accuracy:")
    for idx, col in enumerate(cm.T):
        producer_accuracy  = round( diag[idx] / sum(col), 2)
        logger.info(f"label_id: {label_ids[idx]}, producer accuracy {producer_accuracy}")

    logger.info(f"Test overall accuracy: {accuracy_score(y_test, y_pred)}")
    logger.info(f"Test cohen_kappa accuracy: {cohen_kappa_score(y_test, y_pred)}")
    ##print(classification_report(y_test,y_pred))
    return clf


def predict_arr(patch, classifier):
    """
    Use a classifier to predict on an array


    params:
    ----------

    patch : nd numpy array

    classifier: optimized classifier


    return:
        2D numpy array of int with
        the same size as the window
    """
    patch = reshape_as_image(patch)
    np.nan_to_num(patch, copy=False,nan=0.0, posinf=0,neginf=0)
    # Generate a prediction array
    # new rows: rows * cols
    # new cols: bands
    class_prediction = classifier.predict(patch.reshape(-1, patch.shape[2]))
    # Reshape our classification map back into a 2D matrix so we can visualize it
    class_prediction = class_prediction.reshape(patch[:,:,0].shape)

    return class_prediction


def predict_parallel(src, classifier, number_of_cpu=4, write=False, fpath=None):
    """
    Use a classifier to predict on an entire raster. A number of patches, usually
    equal to the number_of_cpu are processed in parallel

    params:
    ----------

    src : nd numpy array

    classifier: optimized classifier

    number_of_cpu: int

    write: bool

    fpath: str


    return:
        2D numpy array of int with
        the same size as the window

    """
    # Allocate a numpy array for strings
    logger.info('Start predictions on real data...')
    class_final = np.empty((src.height, src.width), dtype = np.int64)
    logger.debug('Numpy arr allocated for classname of type {}'.format(class_final.dtype))

    windows = rst.get_windows(src)

    for blocks in gen_sublist(windows, number_of_cpu):
        patches =  [rst.load_window(src, w) for w in blocks]

        with joblib.parallel_backend(backend="threading"):
            parallel = Parallel(n_jobs=number_of_cpu, verbose=5)
            results = parallel([delayed(predict_arr)(patch, classifier)
                for patch, _ in patches])

        for r, w in zip(results, blocks): # fill in the class_final with class id
            class_final[ w.row_off:w.row_off+w.height,
                    w.col_off:w.col_off+w.width] = r

    logger.info(f'All patches processed')
    if write:
        logger.info('Writing class prediction to disk')
        json_arr = to_json({'class_prediction':class_final}, encoder=NumpyEncoder)
        json_to_disk(json_arr, fpath)
    return class_final


def predict(src, classifier, write=False, fpath=None):
    """
    Use an already optimized skitlearn classifier to classify pixels of
    a raster object.
    ************
    params:
        r_obj -> An istance of the Raster class
        classifier -> optimised classifier
        outdir -> full path to output directory
    return:
        class_final: A 2D numpy array of classnames with
                     the same height and width of src
    """
    # Allocate a numpy array for strings
    logger.info('Start predictions on real data...')
    class_final = np.empty((src.height, src.width), dtype = np.int64)
    logger.debug('Numpy arr allocated for classname of type {}'.format(class_final.dtype))


    for idx, w in enumerate(rst.gen_windows(src), start=1):
        logger.info(f'Processing patch: {idx}')
        logger.info(f'Window: {w}')
        patch, meta = rst.load_window(src, w)
        patch = reshape_as_image(patch)
        np.nan_to_num(patch, copy=False,nan=0.0, posinf=0,neginf=0)
        # Generate a prediction array
        # new rows: rows * cols
        # new cols: bands
        class_prediction = classifier.predict(patch.reshape(-1, patch.shape[2]))
        # Reshape our classification map back into a 2D matrix so we can visualize it
        class_prediction = class_prediction.reshape(patch[:,:,0].shape)
        # fill in the class_final with class names
        class_final[ w.row_off:w.row_off+w.height, w.col_off:w.col_off+w.width] = class_prediction
    logger.info(f'All patches processed')
    if write:
        logger.info('Writing class prediction to disk')
        json_arr = to_json({'class_prediction':class_final}, encoder=NumpyEncoder)
        json_to_disk(json_arr, fpath)
    return class_final

