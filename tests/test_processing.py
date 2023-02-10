import pytest
import rasterio
import numpy as np
from contextlib import ExitStack
from nptyping import NDArray, UInt8, Int32, Float32, Shape, Bool
import geopandas as gpd

import geoRpro.processing as prc
import geoRpro.io as io
from geoRpro.raster import Indexes

from typing import Generator, Any, Final

import pdb


@pytest.fixture
def raster_src() -> Generator[Any, None, None]:
    src: Any = rasterio.open("./data/RGB.byte.tif")
    yield src
    src.close()


def test_rextract(raster_src) -> None:

    inst: Final[dict[str, Any]] = {
        "Inputs": {"extracted_data": ["RGB.byte.tif"]},
        "Indir": "./data",
        "Points": ["./data/pointsToExtract.shp", 32618],
        "ClassName": "test_points",
        "Id": 1,
    }
    rextract: Any = prc.RExtract(inst)
    rextract.run()
