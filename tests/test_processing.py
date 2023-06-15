import pytest
import rasterio
import numpy as np
from contextlib import ExitStack
from nptyping import NDArray, UInt8, Int32, Float32, Float64, Shape, Bool
import geopandas as gpd

import geoRpro.processing as prc
import geoRpro.io as io
from geoRpro.raster import Indexes

from typing import Generator, Any, Final

import pdb


@pytest.fixture
def raster_src() -> Generator[Any, None, None]:
    src: Any = rasterio.open("data/RGB.byte.tif")
    yield src
    src.close()


def test_rextract(cleanup_files) -> None:
    inst: Final[dict[str, Any]] = {
        "Inputs": {"extracted_data": ["RGB.byte.tif"]},
        "Indir": "data",
        "Outdir": "data/out",
        "Points": ["data/pointsToExtract.shp", 32618],
        "ClassName": "test_points",
        "Id": 1,
    }
    rextract: Any = prc.RExtractPoints(inst)
    extracted = rextract.run()

    expected_X: NDArray[Any, Float64] = np.array(
        [
            [32.0, 47.0, 43.0],
            [7.0, 47.0, 72.0],
            [20.0, 23.0, 29.0],
            [19.0, 42.0, 40.0],
            [223.0, 241.0, 255.0],
            [34.0, 64.0, 53.0],
            [27.0, 29.0, 22.0],
            [13.0, 55.0, 77.0],
            [9.0, 12.0, 21.0],
            [21.0, 55.0, 51.0],
        ],
        dtype=np.float64,
    )
    expected_y: NDArray[Any, Float64] = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=np.int64,
    )
    expected_labels_map = {"test_points": "1"}
    assert (extracted.X == expected_X).all()
    assert (extracted.y == expected_y).all()
    assert extracted.labels_map == expected_labels_map
    cleanup_files(rextract.outputs["extracted_data"])
