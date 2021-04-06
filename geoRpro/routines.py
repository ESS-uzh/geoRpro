import os
from contextlib import ExitStack

import rasterio
from shapely.geometry import box, Point

from geoRpro.sent2 import Sentinel2
from geoRpro.raster import Rstack, Indexes
import geoRpro.raster as rst
import geoRpro.utils as ut

import pdb

# * Commune routines

def stack_sent2_bands(indir, bands, outdir, resolution=10, mask=None,
        window=None, polygon=None, indexes=None, fname=None):
    """
    Create a stack of sentinel 2 bands with defined spacial resolution

    **********

    params
    ---------

    indir : strg
            full path to the IMG_DATA folder. This directory is expected to have
            the standard sent2 structure, i.e. three subdirerctories: R10m, R20m
            and R60m

    bands : list
            Bands that should be included into the stack, e.g. 'B02_10m' or 'B05_20m'

    outdir : strg
             full path to the output directory

    resolution : int
                 final resolution of all the bands of the stack

    mask : numpy boolean mask arr

    window : rasterio.windows.Window
             Define a final extent of the stack

    polygon : GEOJson-like dict
              e.g. { 'type': 'Polygon', 'coordinates': [[(),(),(),()]] }
             Define a final extent of the stack
    
    fnam: strg
          Name of the output raster stack; e.g. my_stack.tif

    indexes: OrderDict
    """
    s10 = Sentinel2(os.path.join(indir, 'R10m'))
    s20 = Sentinel2(os.path.join(indir, 'R20m'))

    if window is not None and polygon is not None:
        raise ValueError("Cannot choose both window and polygon params!")

    with ExitStack() as stack_files:

        fpaths = []
        for band in bands:
            try:
                fpaths.append(s10.get_fpath(band))
            except KeyError:
                try:
                    fpaths.append(s20.get_fpath(band))
                except KeyError:
                    raise ValueError(f"Cannot find band: '{band}'. Please provide valid band name.")

        srcs = [stack_files.enter_context(rasterio.open(fp))
            for fp in fpaths]

        band_src_map = dict(zip(bands, srcs))

        with ExitStack() as stack_action:
            rstack = Rstack()
            for band, src in band_src_map.items():
                print(f'Band {band} to be processed..')
                if int(src.res[0]) != resolution: # resample to match res param
                    print(f'Band: {band} will be resampled to {resolution} m resolution..')
                    scale_factor = src.res[0] / resolution
                    arr, meta = rst.load_resample(src, scale_factor)
                    if mask:
                        arr = rst.apply_mask(arr, mask, fill_value=65535)
                    src = stack_action.enter_context(rst.to_src(arr, meta))
                if window:
                    print(f'Selected a window: {window} as AOI')
                    arr, meta = rst.load_window(src, window)
                    src = stack_action.enter_context(rst.to_src(arr, meta))
                if polygon:
                    print(f"Selected a polygon as AOI")
                    arr, meta = rst.load_raster_from_poly(src, polygon)
                    src = stack_action.enter_context(rst.to_src(arr, meta))
                band_src_map[band] = src # update the mapping
                rstack.add_item(src)

            if indexes:
                print(f'Compute indexes: {indexes.keys()} and add them to the stack')

                to_calc = Indexes(metadata=rstack.items[0].profile)

                for idx,vals in indexes.items():
                    try:
                        vals = [band_src_map[v] for v in vals]
                    except KeyError:
                        raise ValueError(f"One or more bands: '{vals}' were not defined as part of the stack.")
                    try:
                        arr_idx, meta_idx = getattr(to_calc, idx)(*vals)
                    except AttributeError:
                        raise ValueError(f"'{idx}' is not a valid Index method")
                    arr_idx, meta_idx = getattr(to_calc, idx)(*vals)
                    src_idx = stack_action.enter_context(rst.to_src(arr_idx, meta_idx))
                    band_src_map[idx] = src_idx # update the mapping
                    rstack.add_item(src_idx)

            rstack.set_metadata_param('interleave', 'band')
            print(band_src_map.keys())
            
            if not fname:
                fname = '_'.join([s10.get_tile_number('B02_10m'), s10.get_datetake('B02_10m')])+'.tif'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            fpath = rst.write_raster(rstack.items, rstack.metadata_collect, os.path.join(outdir, fname))
        return fpath


def find_total_overlaps(indir):

    fpaths = [os.path.join(indir, rfile) for rfile in os.listdir(indir)
            if rfile.endswith('.tif')]

    with ExitStack() as stack_files:
        srcs = [stack_files.enter_context(rasterio.open(fp)) for fp in fpaths]
        polys = [box(*src.bounds) for src in srcs]

        intersects = []

        for poly, poly_front in ut.gen_current_front_pairs(polys):
            if poly.intersects(poly_front) and poly.intersection(poly_front) not in intersects:
                intersects.append(poly.intersection(poly_front))
        print(intersects)

        # save a polygon
        #gdf1 = gpd.GeoDataFrame({"geometry": intersects}, crs=f"EPSG:{srcs[0].crs.to_epsg()}")
        #gdf1.to_file(os.path.join(BASEDIR, "overlap_polygon"))

        for src in srcs:
            print("")
            print(f"Run for raster: {src.files[0]} ..")
            for idx, intersect in enumerate(intersects, start=1):
                try:
                    arr, meta = rst.load_raster_from_poly(src, intersect)
                    print(f"Overlap found, getting arr from poly: {idx}")
                except ValueError:
                    print(f"No overlap found for poly: {idx}")
                    continue
                else:
                    fname = "_".join([f"poly_{idx}", str(arr.shape[1]), str(arr.shape[2])]) + ".tif"
                    print(fname)
                    print(arr.shape)
                    meta.update({
                    "interleave": "band"})
                    outdir = os.path.join(indir, src.files[0].split("/")[-1].split("_")[1])
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    rst.write_array_as_raster(arr, meta, os.path.join(outdir, fname))


def find_tile_overlap(indir, ref):
    """
    Find overlapping areas (polygons) between a reference (ref) and target
    rasters. Polygons are written to disk

    indir : strg
            Full path to the directory containing all raster files

    ref : strg
          Filename of the reference raster. Filenames are expected to have
          the following format: tileName_date.tif,
          e.g. 'T20MPA_20200729.tif'

    """

    rfiles = [rfile for rfile in os.listdir(indir) if rfile.endswith('.tif')
            and rfile != ref]

    outdir = os.path.join(indir, 'overlaps')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with ExitStack() as stack_files:

        src_ref = stack_files.enter_context(rasterio.open(os.path.join(indir, ref)))


        srcs = [stack_files.enter_context(rasterio.open(os.path.join(indir, f)))
                for f in rfiles]

        poly_ref = box(*src_ref.bounds)

        for idx, src in enumerate(srcs, start=1):
            print("")
            print(f"Check overlap between {src_ref.files[0]} and {src.files[0]}")
            poly = box(*src.bounds)
            intersect = poly_ref.intersection(poly)
            try:
                arr, meta = rst.load_raster_from_poly(src, intersect)
                meta.update({
                    "interleave": "band"})
                arr_ref, meta_ref = rst.load_raster_from_poly(src_ref, intersect)
                meta_ref.update({
                    "interleave": "band"})
                print(f"Overlap found, getting arrays from polygon: {idx}")
            except ValueError:
                print(f"No overlap found for poly: {idx}")
                continue
            else:
                tile_ref = src_ref.files[0].split("/")[-1].split("_")[0]
                tile_target = src.files[0].split("/")[-1].split("_")[0]
                fname_ref = "_".join([f"poly_{idx}", tile_ref]) + ".tif"
                fname_target = "_".join([f"poly_{idx}", tile_target]) + ".tif"
                rst.write_array_as_raster(arr, meta, os.path.join(outdir, fname_target))
                rst.write_array_as_raster(arr_ref, meta_ref,
                        os.path.join(outdir, fname_ref))


def raster_to_mask(fpath, resolution, rule, polygon=None, window=None):
    """
    Routine:

    Replace raster (src) values with fill_value based on a mask (src_mask)

    *********


    params:
    ________

     src : rasterio.DatasetReader object

     src_mask : rasterio.DatasetReader object
                Used to generate a boolean mask, the values equal to
                vals are masked

     vals : list
            values to be masked


     fill_value : int
                  replacing values

    """
    pass

if __name__ == "__main__":

    find_tile_overlap('/home/diego/work/dev/data/amazon/20200729_stacks',
            'T20MPA_20200729.tif')
