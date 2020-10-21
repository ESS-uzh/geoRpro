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
from geoRpro.raster import RArr
from geoRpro.utils import NumpyEncoder, to_json, json_to_disk, load_json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


import pdb

class DataExtractor:
    def __init__(self, X, y, labels_map):

        self.X = X
        self.y = y
        self.labels_map = labels_map
        self.label_ids, self.label_ids_count = np.unique(self.y, return_counts=True)


    def save(self, fpath):
        json_arr = to_json({'X':self.X, 'y':self.y, 'labels_map':self.labels_map},
                           encoder=NumpyEncoder)
        json_to_disk(json_arr, fpath)


    def add_class(self, label_name, value):
        """
        Extend X,y in the 0 dimension by adding a number of
        values to X and a numbers of label_id to y.
        map_cl_id is updated with a new mapping: label_name,
        label_id
        """

        logger.debug('''Current Labels ids: {}
                     Sample numbers: {} '''.format(self.label_ids, self.label_ids_count))
        logger.debug('Adding a new class to the training data ...')

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
        self.labels_map[str(label_id)] = label_name

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
    def extract(cls, src, gdf, mask_value=999999):
        """
        Extract the pixel values at point location from a raster src

        Return:

        X matrix (Points, raster bands)
        y vector (point id)
        classname_id mapping

        X         col1 col2 ... band_n          y
        Point1    val1 val2                     id
        Point2                                  id

        ************

        params:
            src -> rasterio.DatasetReader
            gdf -> A geodataframe (geometry, classname, id)

        return:
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
        labels_map = {str(gdf.loc[gdf['classname'] == label_name, 'id'].iloc[0]): label_name for label_name in label_names}

        for index, geom in enumerate(geoms):

            # Transform to GeoJSON format
            feature = [mapping(geom)]

            # out_image.shape == (band_count,1,1) for one Point Object
            out_image, out_transform = mask(src, feature, crop=True)

            # reshape the array to [pixel values, band_count]
            out_image_reshaped = out_image.reshape(-1, src.count)

            # Do not include value if masked
            if np.all(out_image_reshaped == mask_value):
                continue
            pdb.set_trace()
            y = np.append(y,[gdf['id'].iloc[index]] * out_image_reshaped.shape[0])
            X = np.vstack((X,out_image_reshaped))

        return DataExtractor(X, y, labels_map)




if __name__ == "__main__":


    from rasterio.windows import Window
    from contextlib import ExitStack

    basedir = '/home/diego/work/dev/data/Hanneke'

    shape_dir = os.path.join(basedir,'shapes_22092020/final_datapoints_22092020')
    gdf = gpd.read_file(os.path.join(shape_dir, 'final_datapoints_22092020.shp'))
    img_dir = os.path.join(basedir, 'ESSChacoal_from291908_to202002/S2A_MSIL2A_20190906T073611_N0213_R092_T37MBN_20190906T110000.SAFE/GRANULE/L2A_T37MBN_A021968_20190906T075543/IMG_DATA/R10m')

    # pick a band
    s10 = Sentinel2(img_dir)

    with rasterio.open(s10.get_fpaths('B02_10m')[0]) as src:
        r = RArr.load(src)
        data = DataExtractor.extract(src, gdf)
        print(data.X.shape)
        print(data.y.shape)
        print(data.labels_map)
