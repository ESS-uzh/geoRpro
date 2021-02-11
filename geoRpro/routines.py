import os
from contextlib import ExitStack

import rasterio
from shapely.geometry import box, Point

from geoRpro.sent2 import Sentinel2
from geoRpro.raster import Rstack
import geoRpro.raster as rst
import geoRpro.utils as ut

import pdb

# * Commune routines

def stack_sent2_bands(indir, bands, outdir, window=None):
    """
    Create a stack of sentinel 2 bands of 10m resolution

    **********

    params
    ---------

    indir: strg

    bands: list

    outdir: strg

    window: rasterio.windows.Window
    """
    s10 = Sentinel2(os.path.join(indir, 'R10m'))
    s20 = Sentinel2(os.path.join(indir, 'R20m'))

    with ExitStack() as stack_files:

        fpaths = []
        for band in bands:
            try:
                fpaths.append(s10.get_fpath(band))
            except KeyError:
                fpaths.append(s20.get_fpath(band))

        srcs = [stack_files.enter_context(rasterio.open(fp))
            for fp in fpaths]

        with ExitStack() as stack_action:
            rstack = Rstack()
            for src in srcs:
                if int(src.res[0]) == 20: # resample to 10 m
                    print(f"scr to resample, res: {src.res}")
                    arr_r, meta = rst.load_resample(src)
                    src = stack_action.enter_context(rst.to_src(arr_r, meta))
                if window:
                    arr, meta = rst.load_window(src, window)
                    src = stack_action.enter_context(rst.to_src(arr, meta))
                print(f"scr to add to the stack with res: {src.res}")
                rstack.add_item(src)
            rstack.set_metadata_param('interleave', 'band')

            fname = '_'.join([s10.get_tile_number('B02_10m'), s10.get_datetake('B02_10m')])+'.tif'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            fpath = rst.write_raster(rstack.items, rstack.metadata_collect, os.path.join(outdir, fname))
        print(fpath)


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
                arr_ref, meta_ref = rst.load_raster_from_poly(src_ref, intersect)
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


def replace_raster_values(src, src_mask, vals=[3,7,8,9,10], fill_value=65535):
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
    arr, _ = rst.load(src, masked=True)

    res = int(src.res[0])
    res_mask = int(src_mask.res[0])

    if res != res_mask: # resample src_sc to match src
        scale_factor = res / res_mask
        arr_mask, meta_mask = rst.load_resample(src_mask, scale_factor)
    else:
        arr_mask, meta_mask = rst.load(src_mask)

    # create a mask array from src_mask file
    mask_arr, _ = rst.mask_vals(arr_mask, meta_mask, vals)

    return rst.apply_mask(arr, mask_arr.mask, fill_value=fill_value)

if __name__ == "__main__":

    find_tile_overlap('/home/diego/work/dev/data/amazon/SA_MSIL2A_20200729_stacks',
            'T20MPA_20200729.tif')
