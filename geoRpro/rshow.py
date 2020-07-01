import rasterio
from matplotlib import pyplot

def show(fname, n):
    with rasterio.open(fname) as src:
        arr = src.read(1)
        print(type(arr))
        print(arr.shape)
        pyplot.figure(n)
        return pyplot.imshow(arr, cmap='pink')


show('mask.tif', 1)
show("T37MBN_20190628T073621_SCL_res_10m.tif", 2)


pyplot.show()
