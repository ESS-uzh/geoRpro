import os
import rasterio
from matplotlib import pyplot

INDIR = "/home/diego/work/dev/ess_diego/github/goeRpro_inp"
OUTDIR = "/home/diego/work/dev/ess_diego/github/goeRpro_out"

def show_ndvi(fname, n):
    with rasterio.open(fname) as src:
        arr = src.read(1)
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


masked_b02 = os.path.join(OUTDIR, "T37MBN_20190628T073621_B02_masked_aoi.tiff") 
scl_mask = os.path.join(OUTDIR, "T37MBN_20190628T073621_SCL_mask_aoi.tiff") 

show_img(masked_b02, 1, 100, 2500)
show_img(scl_mask, 2, 0, 1)

with rasterio.open(masked_b02) as src:
        b02 = src.read(1)
#with rasterio.open(masked_ndvi) as src1:
#        ndvi_arr_m = src1.read(1)
show_hist(b02)
#show_hist(ndvi_arr_m)

pyplot.show()
