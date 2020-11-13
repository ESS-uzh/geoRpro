import os
import rasterio
from rasterio.windows import Window
from matplotlib import pyplot

INDIR = "/home/diego/work/dev/data/planet_data/imgs_lcc_charcoal_overlap"

win = Window(0, 0, 6000, 6000)

def show_ndvi(fname, n):
    with rasterio.open(fname) as src:
        arr = src.read(1, window=win)
        print(type(arr))
        print(arr.shape)
        pyplot.figure(n)
        return pyplot.imshow(arr, cmap='RdYlGn', vmin=0, vmax=0.6)

def show_img(fname, n, vmin, vmax):
    with rasterio.open(fname) as src:
        arr = src.read(1)
        print(type(arr))
        print(arr.shape)
        pyplot.figure(n)
        return pyplot.imshow(arr, cmap='pink', vmin=vmin, vmax=vmax)

def show_hist(arr):
    fig, axs = pyplot.subplots()

    # We can set the number of bins with the `bins` kwarg
    return axs.hist(arr.flatten(), bins=50)


ndvi = os.path.join(INDIR, "20200716_104044_ssc11_u0001_analytic_ndvi.tif")

show_ndvi(ndvi, 1)

pyplot.show()
