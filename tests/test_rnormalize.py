import pytest
import rasterio
import numpy as np
from contextlib import ExitStack
from nptyping import NDArray, UInt8, Int32, Float32, Float64, Shape, Bool
import geopandas as gpd

import geoRpro.processing as prc
import geoRpro.io as io
from geoRpro.raster import Indexes, load

from typing import Generator, Any, Final

import pdb


def test_rnormalize(cleanup_files) -> None:
    inst: Final[dict[str, Any]] = {
        "Inputs": {"rgb_normalized": "aviris_rgb.tif"},
        "Indir": "data",
        "Outdir": "data/out",
        "Smin": 1,
        "Smax": 255,
        "NoDataValue": -9999,
    }
    rnormalize: Any = prc.RNormalize(inst)
    rnormalize.run()
    arr, meta = load(rasterio.open(rnormalize.outputs["rgb_normalized"]))

    expected_arr_slice_with_ndv: NDArray[Any, UInt8] = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    expected_arr_slice_with_values: NDArray[Any, UInt8] = np.array(
        [
            [190, 205, 220, 227, 234],
            [190, 198, 212, 227, 234],
            [198, 198, 212, 220, 220],
            [198, 198, 205, 212, 220],
            [212, 205, 198, 212, 220],
        ],
        dtype=np.uint8,
    )
    assert (arr[0, :5, :5] == expected_arr_slice_with_ndv).all()
    assert (arr[0, 45:50, 45:50] == expected_arr_slice_with_values).all()
    assert meta["dtype"] == "uint8"
    assert meta["count"] == 3
    cleanup_files(rnormalize.outputs["rgb_normalized"])


# def test_rnormalize_with_replace_ndv_equal_55(cleanup_files) -> None:
#    inst: Final[dict[str, Any]] = {
#        "Inputs": {"rgb_normalized": "aviris_rgb.tif"},
#        "Indir": "data",
#        "Outdir": "data/out",
#        "Smin": 1,
#        "Smax": 255,
#        "ReplaceNdv": 55,
#    }
#    rnormalize: Any = prc.RNormalize(inst)
#    rnormalize.run()
#    arr, meta = load(rasterio.open(rnormalize.outputs["rgb_normalized"]))
#
#    expected_arr_slice_with_replaced_ndv_equal_55: NDArray[Any, UInt8] = np.array(
#        [
#            [55, 55, 55, 55, 55],
#            [55, 55, 55, 55, 55],
#            [55, 55, 55, 55, 55],
#            [55, 55, 55, 55, 55],
#            [55, 55, 55, 55, 55],
#        ],
#        dtype=np.uint8,
#    )
#    assert (arr[0, :5, :5] == expected_arr_slice_with_replaced_ndv_equal_55).all()
#    assert meta["dtype"] == "uint8"
#    assert meta["count"] == 3
#    cleanup_files(rnormalize.outputs["rgb_normalized"])
