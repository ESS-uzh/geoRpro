import os
import rasterio
from matplotlib import pyplot

INDIR = "/home/diego/work/dev/ess_diego/github/goeRpro_inp"
OUTDIR = "/home/diego/work/dev/ess_diego/github/goeRpro_out"

def show(fname, n):
    with rasterio.open(fname) as src:
        arr = src.read(1)
        print(type(arr))
        print(arr.shape)
        pyplot.figure(n)
        return pyplot.imshow(arr, cmap='pink')

orig = os.path.join(INDIR, "T37MBN_20190628T073621_B04_10m.jp2")
mask = os.path.join(OUTDIR, "scl_mask.tiff") 
masked = os.path.join(OUTDIR, "masked_B04.tiff") 

show(orig, 1)
show(mask, 2)
show(masked, 3)


pyplot.show()
