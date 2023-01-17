import pytest
import rasterio
import numpy as np
from contextlib import ExitStack

import geoRpro.raster as rst
import geoRpro.io as io
from geoRpro.raster import Indexes

from typing import Generator, Any

import pdb


@pytest.fixture
def raster_src() -> Generator[Any, None, None]:
    src: Any = rasterio.open('./data/RGB.byte.tif')
    yield src
    src.close()


def test_raise_bands_arg_not_a_list(raster_src: Any) -> None:
    """Pass int to bands raise ValueError"""
    with pytest.raises(ValueError):
        rst.load(raster_src, bands=1)


def test_meta_bands(raster_src):
    """Pass list of bands produces count equal 2"""
    _, meta = rst.load(raster_src, bands=[1, 2])
    assert meta['count'] == 2


def test_meta_all_bands(raster_src):
    """Pass no bands argument"""
    _, meta = rst.load(raster_src)
    assert meta['count'] == 3
    assert meta['height'] == 718
    assert meta['width'] == 791


def test_meta_window(raster_src):
    """Pass bands and window"""
    arr, meta = rst.load(raster_src, bands=[
                         1, 2], window=((6, 12), (171, 175)))
    assert meta['height'] == 6
    assert meta['width'] == 4
    assert meta['count'] == 2


def test_array_window(raster_src):
    arr, _ = rst.load(raster_src, bands=[1], window=((6, 12), (171, 175)))
    expected = np.array([
        [6, 8, 6, 8],
        [8, 8, 8, 50],
        [8, 9, 8, 29],
        [8, 8, 8, 8],
        [8, 6, 6, 6],
        [6, 6, 6, 5]], dtype=np.int8)
    assert (arr == expected).all()


def test_array_mask(raster_src):
    arr, _ = rst.load(raster_src, bands=[1], window=((6, 12), (156, 160)),
                      masked=True)
    expected = np.array([
        [True, True, False, False],
        [True, True, False, False],
        [True, True, False, False],
        [True, False, False, False],
        [True, False, False, False],
        [True, False, False, False]])
    assert (arr.mask == expected).all()


def test_meta_upsample(raster_src):
    _, meta = rst.load_resample(raster_src, scale=2)
    assert meta['width'] == 1582
    assert meta['height'] == 1436


def test_meta_downsample(raster_src):
    _, meta = rst.load_resample(raster_src, scale=0.5)
    assert meta['width'] == 395.5
    assert meta['height'] == 359.0


def test_ndvi_scale_factor_1000(raster_src):
    red, meta_red = rst.load(
        raster_src, bands=[1], window=((6, 12), (171, 175)))
    nir, meta_nir = rst.load(
        raster_src, bands=[2], window=((6, 12), (171, 175)))
    with ExitStack() as stack:
        src_red = stack.enter_context(io.to_src(red, meta_red))
        src_nir = stack.enter_context(io.to_src(nir, meta_nir))
        indx = Indexes(meta_red, scale_factor=1000)
        ndvi, _ = indx.ndvi(src_red, src_nir)
    expected_ndvi = np.array([
        [700, 578, 538, 384],
        [627, 644, 529, 107],
        [529, 513, 600, 256],
        [529, 600, 600, 529],
        [578, 666, 647, 600],
        [625, 647, 600, 655]], dtype=np.int32)
    assert (ndvi == expected_ndvi).all()


def test_ndvi_scale_factor_1(raster_src):
    red, meta_red = rst.load(
        raster_src, bands=[1], window=((6, 12), (171, 175)))
    nir, meta_nir = rst.load(
        raster_src, bands=[2], window=((6, 12), (171, 175)))
    with ExitStack() as stack:
        src_red = stack.enter_context(io.to_src(red, meta_red))
        src_nir = stack.enter_context(io.to_src(nir, meta_nir))
        indx = Indexes(meta_red, scale_factor=1)
        ndvi, _ = indx.ndvi(src_red, src_nir)
    expected_ndvi = np.array([
        [0.7, 0.57894737, 0.53846157, 0.3846154],
        [0.627907, 0.64444447, 0.5294118, 0.10714286],
        [0.5294118, 0.5135135, 0.6, 0.25641027],
        [0.5294118, 0.6, 0.6, 0.5294118],
        [0.57894737, 0.6666667, 0.64705884, 0.6],
        [0.625, 0.64705884, 0.6, 0.6551724]], dtype=np.float32)
    assert (ndvi == expected_ndvi).all()
