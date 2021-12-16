import pytest
import rasterio
import numpy as np

import geoRpro.raster as rst

@pytest.fixture
def raster_src():
    src = rasterio.open('./data/RGB.byte.tif')
    yield src
    src.close()

def test_meta_load_band(raster_src):
    """Pass int produces count equal 1"""
    _, meta = rst.load(raster_src, bands=1)
    assert meta['count'] == 1

def test_meta_load_bands(raster_src):
    """Pass list of bands produces count equal 2"""
    _, meta = rst.load(raster_src, bands=[1, 2])
    assert meta['count'] == 2

def test_meta_load_all_bands(raster_src):
    """Pass no bands argument"""
    _, meta = rst.load(raster_src)
    assert meta['count'] == 3
    assert meta['height'] == 718
    assert meta['width'] == 791

def test_meta_window_bands(raster_src):
    """Pass bands and window"""
    arr, meta = rst.load(raster_src, bands=[1, 2], window=((6, 12), (171, 175)))
    assert meta['height'] == 6
    assert meta['width'] == 4
    assert meta['count'] == 2

def test_window_tuple(raster_src):
    arr, _ = rst.load(raster_src, bands=1, window=((6, 12), (171, 175)))
    expected = np.array([
        [6, 8, 6, 8],
        [8, 8, 8, 50],
        [8, 9, 8, 29],
        [8, 8, 8, 8],
        [8, 6, 6, 6],
        [6, 6, 6, 5]], dtype=np.int8)
    assert (arr == expected).all()


