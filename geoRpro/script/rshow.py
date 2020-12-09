import os
import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.plot import show, reshape_as_image
from matplotlib import pyplot
import matplotlib.pyplot as plt

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


def plot_select_class_prediction(class_prediction, mapping ,color_map, classes,
                                 figsize=(10,10), fontsize=8, src=None,
                                 save=False, path=None):



    # find the highest pixel value in the prediction image
    #n = int(np.max(class_prediction[0,:,:]))
    n = len(color_map)

    # create a default white color map using colors as float 0-1
    index_colors = [(1.0, 1.0, 1.0) for key in range(0, n )]


    for cl, val in color_map.items():
        # convert the class_prediction values to match the plotting values
        class_prediction = np.where(class_prediction == int(mapping[cl]), [*val][0], class_prediction)

    # Replace index_color with the one you want to visualize
    for cl in classes:
        idx = list(color_map[cl].keys())[0]
        vals = list(color_map[cl].values())[0]
        # Transform 0 - 255 color values from colors as float 0 - 1
        _v = [_v / 255.0 for _v in vals]
        index_colors[idx] = tuple(_v)

    #pdb.set_trace()

    cmap = plt.matplotlib.colors.ListedColormap(index_colors)


    from matplotlib.patches import Patch
    # Create a list of labels sorted by the int of color_map
    class_labels = [el[0] for el in sorted(color_map.items(), key=lambda label: label[1].keys())]
    # A path is an object drawn by matplotlib. In this case a patch is a box draw on your legend
    # Below you create a unique path or box with a unique color - one for each of the labels above
    legend_patches = [Patch(color=icolor, label=label)
                      for icolor, label in zip(index_colors, class_labels)]

    # Plot Classification
    fig, axs = plt.subplots(1,1,figsize=figsize)
    axs.imshow(class_prediction, cmap=cmap, interpolation='none')
    if src:
        from rasterio.plot import plotting_extent
        axs.imshow(class_prediction, extent=plotting_extent(src), cmap=cmap, interpolation='none')

    axs.legend(handles=legend_patches,
              facecolor="white",
              edgecolor="white",
              bbox_to_anchor=(1.20, 1),
              fontsize=fontsize)  # Place legend to the RIGHT of the map
    axs.set_axis_off()
    plt.show()
    if save and path:
        fig.savefig(path, bbox_inches='tight')

def points_on_layer_plot(src, arr, gdf, n, **kwargs):

        cmap, marker, markersize, color, label = kwargs.get('cmap',"pink"), \
                                  kwargs.get('marker',"s"), \
                                  kwargs.get('markersize',30), \
                                  kwargs.get('color',"purple"), \
                                  kwargs.get('label',"classname")
        # Plotting
        fig = plt.figure(n, figsize=(8, 8))
        ax1 = plt.subplot(2,1,1)
        gdf.plot(ax=ax1,
                     marker=marker,
                     markersize=markersize,
                     color=color,
                     label=label)

        ax1.imshow(arr,
                  # Set the spatial extent or else the data will not line up with your geopandas layer
                  extent=plotting_extent(src),
                  cmap=cmap)
        ax2 = plt.subplot(2,1,2, sharex=ax1, sharey=ax1)
        ax2.imshow(arr,
                  extent=plotting_extent(src),
                  cmap=cmap)
