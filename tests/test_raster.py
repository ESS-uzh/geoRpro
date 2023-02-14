import pytest
import rasterio
import numpy as np
from contextlib import ExitStack
from nptyping import NDArray, UInt8, Int32, Float32, Shape, Bool
import geopandas as gpd

import geoRpro.raster as rst
import geoRpro.io as io
from geoRpro.raster import Indexes

from typing import Generator, Any, Final

import pdb


@pytest.fixture
def raster_src() -> Generator[Any, None, None]:
    src: Any = rasterio.open("./data/RGB.byte.tif")
    yield src
    src.close()


def get_shape(poly_param: list) -> Any:

    shape_file = poly_param[0]
    crs = poly_param[1]
    index = poly_param[2]

    gdf = gpd.read_file(shape_file)
    gdf = gdf.to_crs(f"EPSG:{crs}")
    return gdf["geometry"][index]


def test_load_raise_bands_arg_not_a_list(raster_src: Any) -> None:
    """Pass int to bands raise ValueError"""
    with pytest.raises(ValueError):
        rst.load(raster_src, bands=1)  # type: ignore


def test_load_meta_bands(raster_src) -> None:
    """Pass list of bands produces count equal 2"""
    meta: dict[str, Any]

    _, meta = rst.load(raster_src, bands=[1, 2])

    assert meta["count"] == 2
    assert meta["dtype"] == "uint8"


def test_load_meta_all_bands(raster_src) -> None:
    """Pass no bands argument"""
    meta: dict[str, Any]

    _, meta = rst.load(raster_src)
    assert meta["count"] == 3
    assert meta["height"] == 718
    assert meta["width"] == 791


def test_load_meta_window(raster_src) -> None:
    """Pass bands and window"""
    meta: dict[str, Any]

    _, meta = rst.load(raster_src, bands=[1, 2], window=((6, 12), (171, 175)))
    assert meta["height"] == 6
    assert meta["width"] == 4
    assert meta["count"] == 2


def test_load_array_window(raster_src) -> None:
    arr: NDArray[Any, UInt8]

    arr, _ = rst.load(raster_src, bands=[1], window=((6, 12), (171, 175)))
    expected: NDArray[Any, UInt8] = np.array(
        [
            [6, 8, 6, 8],
            [8, 8, 8, 50],
            [8, 9, 8, 29],
            [8, 8, 8, 8],
            [8, 6, 6, 6],
            [6, 6, 6, 5],
        ],
        dtype=np.int8,
    )
    assert (arr == expected).all()


def test_load_array_polygon(raster_src) -> None:

    shape = get_shape(["./data/poly_rgb.shp", "32618", 0])

    arr: NDArray[Any, UInt8]
    meta: dict[str, Any]

    arr, meta = rst.load_polygon(raster_src, shape)
    expected: NDArray[Any, UInt8] = np.array(
        [
            [12, 9, 9, 27, 128, 40],
            [6, 11, 17, 17, 38, 178],
            [11, 12, 41, 35, 255, 31],
            [24, 24, 31, 37, 221, 74],
            [26, 27, 26, 35, 32, 11],
        ],
        dtype=np.uint8,
    )

    assert (arr[0, 50:55, 100:106] == expected).all()
    assert meta["count"] == 3
    assert meta["height"] == 141
    assert meta["width"] == 259


def test_load_array_mask(raster_src) -> None:
    arr: Any

    arr, _ = rst.load(raster_src, bands=[1], window=((6, 12), (156, 160)), masked=True)
    expected = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [True, True, False, False],
            [True, False, False, False],
            [True, False, False, False],
            [True, False, False, False],
        ]
    )
    assert (arr.mask == expected).all()


def test_load_resample_upsample(raster_src) -> None:
    meta: dict[str, Any]
    arr: NDArray[Any, UInt8]
    arr, meta = rst.load_resample(raster_src, scale=2)

    expected: NDArray[Any, UInt8] = np.array(
        [
            [26, 26, 32, 32],
            [26, 26, 32, 32],
            [36, 36, 35, 35],
            [36, 36, 35, 35],
            [40, 40, 42, 42],
            [40, 40, 42, 42],
            [47, 47, 48, 48],
            [47, 47, 48, 48],
            [42, 42, 40, 40],
            [42, 42, 40, 40],
        ],
        dtype=np.uint8,
    )

    assert (arr[2, 100:110, 472:476] == expected).all()

    assert meta["width"] == 1582
    assert meta["height"] == 1436


def test_load_resample_downsample(raster_src) -> None:
    meta: dict[str, Any]
    arr: NDArray[Any, UInt8]
    arr, meta = rst.load_resample(raster_src, scale=0.5)
    expected: NDArray[Any, UInt8] = np.array(
        [
            [26, 22, 31, 24, 30, 25, 27],
            [29, 23, 33, 28, 26, 34, 23],
            [27, 31, 28, 28, 29, 24, 31],
            [33, 32, 30, 29, 27, 24, 33],
            [34, 37, 38, 32, 28, 24, 24],
            [39, 31, 30, 30, 18, 70, 30],
            [37, 34, 31, 32, 33, 35, 29],
            [40, 34, 40, 35, 33, 47, 33],
            [40, 37, 39, 40, 35, 47, 34],
            [40, 19, 43, 37, 35, 40, 38],
        ],
        dtype=np.uint8,
    )

    assert (arr[2, 10:20, 100:107] == expected).all()
    assert meta["width"] == 395.5
    assert meta["height"] == 359.0


def test_mask_vals(raster_src) -> None:
    arr: NDArray[Any, UInt8]
    meta: dict[str, Any]
    arr, meta = rst.load(raster_src, bands=[1], window=((6, 12), (171, 175)))

    arr_mask: NDArray[Any, UInt8]
    meta_mask: dict[str, Any]
    arr_mask, meta_mask = rst.mask_vals(arr, meta, [6, 8])

    expected_data: NDArray[Any, UInt8] = np.array(
        [
            [6, 8, 6, 8],
            [8, 8, 8, 50],
            [8, 9, 8, 29],
            [8, 8, 8, 8],
            [8, 6, 6, 6],
            [6, 6, 6, 5],
        ],
        dtype=np.uint8,
    )

    expected_mask = np.array(
        [
            [
                [True, True, True, True],
                [True, True, True, False],
                [True, False, True, False],
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, False],
            ]
        ]
    )

    assert meta_mask["nbits"] == 1
    assert (arr_mask.mask == expected_mask).all()  # type: ignore
    assert (arr_mask.data == expected_data).all()  # type: ignore


def test_apply_mask(raster_src) -> None:
    arr_inp: NDArray[Any, UInt8]
    meta_inp: dict[str, Any]
    arr_inp, meta_inp = rst.load(raster_src, bands=[1], window=((6, 12), (171, 175)))
    arr_mask: NDArray[Any, UInt8]
    meta_mask: dict[str, Any]
    arr_mask, meta_mask = rst.mask_vals(arr_inp, meta_inp, [6, 8])

    arr_out = rst.apply_mask(arr_inp, arr_mask.mask)  # type: ignore

    expected: NDArray[Any, UInt8] = np.array(
        [
            [
                [0, 0, 0, 0],
                [0, 0, 0, 50],
                [0, 9, 0, 29],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 5],
            ]
        ],
        dtype=np.uint8,
    )
    assert (arr_out == expected).all()


def test_get_windows(raster_src) -> None:
    result: Final[list[Any]] = rst.get_windows(raster_src)
    assert len(result) == 240


def test_indexes_ndvi_scale_factor_1000(raster_src) -> None:
    red: NDArray[Any, UInt8]
    meta_red: dict[str, Any]
    nir: NDArray[Any, UInt8]
    meta_nir: dict[str, Any]

    red, meta_red = rst.load(raster_src, bands=[1], window=((6, 12), (171, 175)))
    nir, meta_nir = rst.load(raster_src, bands=[2], window=((6, 12), (171, 175)))
    with ExitStack() as stack:
        src_red: Any = stack.enter_context(io.to_src(red, meta_red))
        src_nir: Any = stack.enter_context(io.to_src(nir, meta_nir))
        indx: Indexes = Indexes(meta_red, scale_factor=1000)

        ndvi: NDArray[Any, Int32] | NDArray[Any, Float32]

        ndvi, _ = indx.ndvi(src_red, src_nir)  # self.instructions.get("Scale", 1000)
    expected_ndvi: NDArray[Any, Int32] = np.array(
        [
            [700, 578, 538, 384],
            [627, 644, 529, 107],
            [529, 513, 600, 256],
            [529, 600, 600, 529],
            [578, 666, 647, 600],
            [625, 647, 600, 655],
        ],
        dtype=np.int32,
    )
    assert (ndvi == expected_ndvi).all()


def test_load_ndvi_scale_factor_1(raster_src) -> None:
    red: NDArray[Any, UInt8]
    meta_red: dict[str, Any]
    nir: NDArray[Any, UInt8]
    meta_nir: dict[str, Any]

    red, meta_red = rst.load(raster_src, bands=[1], window=((6, 12), (171, 175)))
    nir, meta_nir = rst.load(raster_src, bands=[2], window=((6, 12), (171, 175)))

    with ExitStack() as stack:
        src_red: Any = stack.enter_context(io.to_src(red, meta_red))
        src_nir: Any = stack.enter_context(io.to_src(nir, meta_nir))
        indx: Indexes = Indexes(meta_red, scale_factor=1)

        ndvi: NDArray[Any, Int32] | NDArray[Any, Float32]

        ndvi, _ = indx.ndvi(src_red, src_nir)
    expected_ndvi: NDArray[Any, Float32] = np.array(
        [
            [0.7, 0.57894737, 0.53846157, 0.3846154],
            [0.627907, 0.64444447, 0.5294118, 0.10714286],
            [0.5294118, 0.5135135, 0.6, 0.25641027],
            [0.5294118, 0.6, 0.6, 0.5294118],
            [0.57894737, 0.6666667, 0.64705884, 0.6],
            [0.625, 0.64705884, 0.6, 0.6551724],
        ],
        dtype=np.float32,
    )
    assert (ndvi == expected_ndvi).all()
