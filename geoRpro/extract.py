import os
import copy
import logging
from contextlib import contextmanager
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd
import pandas as pd

from geoRpro.sent2 import Sentinel2
from geoRpro.utils import NumpyEncoder, to_json, json_to_disk, load_json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


import pdb

class DataExtractor:
    """
    Container for X features (georaster pixel values) and y labels (class id) matrix

    *********
    
    params:
    _________


    X : 2D numpy array
        features matrix, contains raster values for each band (points, bands)
        

    y : 1D numpy array
        labels matrix, contains ids of all classes

    labels_map : dict
                 'classname': 'id' mapping


       Example:

       X         band1  band2 ... band_n          y
       Point1    val1   val2  ... val_n           id
       Point2                                     id
       .                                          .
       .                                          .
       Point_n                                    id_n  


    """
    def __init__(self, X, y, labels_map):

        self.X = X
        self.y = y
        self.labels_map = labels_map
        self.label_ids, self.label_ids_count = np.unique(self.y, return_counts=True)


    def save(self, fpath):
        json_arr = to_json({'X':self.X, 'y':self.y, 'labels_map':self.labels_map},
                           encoder=NumpyEncoder)
        logger.debug(f"Saving training data to disk")
        json_to_disk(json_arr, fpath)


    def add_class(self, label_name, value):
        """
        Extend X,y in the 0 dimension by adding a number of
        values to X and a numbers of label_id to y.
        map_cl_id is updated with a new mapping: label_name,
        label_id
        """

        logger.debug(f"Current labels ids: {self.label_ids} Sample numbers: {self.label_ids_count}")
        logger.debug(f"Adding a new class: {label_name} with value {value} to the training data ...")

        # set number of samples equal to the highest number of existing samples
        samples_len = np.max(self.label_ids_count)

        # make a label_id for the new label_name
        label_id = max(self.label_ids) + 1

        # create arrays of values and label_ids
        values = np.ones((samples_len, self.X.shape[1]))*value
        ids = np.array([label_id for _ in range(samples_len)])

        # update instance attributes
        self.X = np.vstack((self.X, values))
        self.y = np.append(self.y, ids)
        self.label_ids = np.append(self.label_ids, label_id)
        self.label_ids_count = np.append(self.label_ids_count, samples_len)
        self.labels_map[label_name] = str(label_id)
        logger.debug(f"New labels ids: {self.label_ids} Sample numbers: {self.label_ids_count}")

    def remove_class(self, label_name):
        """
        Extend X,y in the 0 dimension by adding a number of
        values to X and a numbers of label_id to y.
        map_cl_id is updated with a new mapping: label_name,
        label_id
        """

        logger.debug(f"Current Labels ids: {self.label_ids} Sample numbers: {self.label_ids_count}")
        logger.debug(f"Removing the class: {label_name} from the training data ...")
        # get label_id corresponding to label_name
        label_id = self.labels_map[label_name]

        # get row indexes for the class to be removed
        idxs = np.where(self.y == int(label_id))

        # remove class (from X and y)
        self.X = np.delete(self.X, idxs, axis=0)
        self.y = np.delete(self.y, idxs, axis=0)

        # update instance attributes
        idx = np.where(self.label_ids == int(label_id))
        self.label_ids = np.delete(self.label_ids, idx)
        self.label_ids_count = np.delete(self.label_ids_count, idx)
        del self.labels_map[label_name]
        logger.debug(f"New labels ids: {self.label_ids} Sample numbers: {self.label_ids_count}")


    def to_df(self, X, y):
        df = pd.DataFrame(data=self.X,
                          index=[i for i in range(0, self.X.shape[0])],
                          columns=["Band"+str(i) for i in range(1, self.X.shape[1]+1)])
        df['labels'] = self.y.tolist()

        # !! should return df[id] too !!
        return df

    @classmethod
    def load_xy(cls, fpath):
        data = load_json(fpath)
        X = np.asarray(data['X'])
        y = np.asarray(data['y'])
        labels_map = data['labels_map']
        return DataExtractor(X, y, labels_map)

    @classmethod
    def clip_gdf(cls, src, gdf):
        pass

    @classmethod
    def extract(cls, src, gdf, mask_value=9999):
        """
        Extract the pixel values at point location from a georaster
        
        **********

        params:
        _________

        src : rasterio.DatasetReader object

        gdf : geopandas.geodataframe.GeoDataFrame object
              Must be (geometry, classname, id(int))


        mask_value : int
                     pixels values that will be excluded


        Return:
           An instance of the DataExtractor class
        """
        # Numpy array of shapely objects
        geoms = gdf.geometry.values

        # convert all labels_id to int
        gdf.id = gdf.id.astype(int)

        # allocate empty ndarray
        X = np.array([]).reshape(0,src.count) # values
        y = np.array([], dtype=np.int64) # ids

        # build classname: id mapping
        label_names = np.unique(gdf.classname)
        #labels_map = {str(gdf.loc[gdf['classname'] == label_name, 'id'].iloc[0]): label_name for label_name in label_names}
        labels_map = {label_name: str(gdf.loc[gdf['classname'] == label_name, 'id'].iloc[0])
                for label_name in label_names}

        for index, geom in enumerate(geoms):
            #print(f"Start {index}: {geom}")

            # Transform to GeoJSON format
            feature = [mapping(geom)]

            try:
                # out_image.shape == (band_count,1,1) for one Point Object
                out_image, out_transform = mask(src, feature, crop=True)
            except ValueError:
                print(f"Cannot extract point: {geom}, classname: {gdf['classname'].iloc[index]} . Try the next one ..")
                continue

            # reshape the array to [pixel values, band_count]
            out_image_reshaped = out_image.reshape(-1, src.count)
            #print(f"Extracted values {out_image_reshaped}")

            # Do not include value if masked
            if np.all(out_image_reshaped == mask_value):
                #print(f"To mask {index}: {geom}")
                continue


            y = np.append(y,[gdf['id'].iloc[index]] * out_image_reshaped.shape[0])
            #print(f"Append {index}: label: {gdf['id'].iloc[index]}")
            X = np.vstack((X,out_image_reshaped))

        return DataExtractor(X, y, labels_map)
