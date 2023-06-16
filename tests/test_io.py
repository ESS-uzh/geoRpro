from nptyping import NDArray, UInt8, Int32, Float32, Float64, Shape, Bool

from rasterio import Affine, CRS
import geoRpro.io as io

from typing import Generator, Any, Final

import pdb


def test_get_metadata():
    metadata = io.get_metadata("data/RGB.byte.tif")

    expected = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0.0,
        "width": 791,
        "height": 718,
        "count": 3,
        "crs": CRS.from_epsg(32618),
        "transform": Affine(
            300.0379266750948, 0.0, 101985.0, 0.0, -300.041782729805, 2826915.0
        ),
        "blockysize": 3,
        "tiled": False,
        "interleave": "pixel",
    }
    assert metadata == expected
