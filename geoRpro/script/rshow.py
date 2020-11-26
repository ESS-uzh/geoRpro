import os
import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.plot import show, reshape_as_image
from matplotlib import pyplot

import pdb

INDIR = "/home/diego/work/dev/data/planet_data/imgs_lcc_charcoal_overlap"

win = Window(0, 0, 6000, 6000)

def show_ndvi(fname, n):
    with rasterio.open(fname) as src:
        arr = src.read(1, window=win)
        print(type(arr))
        print(arr.shape)
        pyplot.figure(n)
        return pyplot.imshow(arr, cmap='RdYlGn', vmin=0, vmax=0.6)

def show_img(fpath, n):
    with rasterio.open(fpath) as src:
        arr = src.read([3,2,1], masked=True)
        print(type(arr))
        print(arr.shape)
        if arr.dtype == 'uint16':
            arr = (255 * (arr / np.max(arr)) ).astype(np.uint8)
        pyplot.figure(n)
        return pyplot.imshow(reshape_as_image(arr))

def show_hist(arr):
    fig, axs = pyplot.subplots()

    # We can set the number of bins with the `bins` kwarg
    return axs.hist(arr.flatten(), bins=50)


ndvi = os.path.join(INDIR, "overlap01_ndvi.tif")

show_ndvi(ndvi, 1)

fpath = "/home/diego/work/dev/data/planet_data/imgs_lcc_charcoal_overlap/overlap01.tif"

show_img(fpath, 2)

pyplot.show()
