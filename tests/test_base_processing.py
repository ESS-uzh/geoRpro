from typing import Generator, Any, Final, Literal, Callable

import pytest
import rasterio
import numpy as np
from contextlib import ExitStack
from nptyping import NDArray, UInt8, Int32, Float32, Float64, Shape, Bool
import geopandas as gpd

from geoRpro.base_processing import ProcessBase
import geoRpro.io as io
from geoRpro.raster import Indexes

from typing import Generator, Any, Final

import pdb


class MockChild(ProcessBase):
    INSTRUCTIONS: set[str] = {"Inputs", "Indir", "Outdir", "Satellite"}

    def __init__(self, instructions) -> None:
        super().__init__(instructions)


def test_init() -> None:
    inst: Final[dict[str, Any]] = {
        "Inputs": {"input": "RGB.byte.tif"},
        "Indir": "data",
        "Outdir": "data/out",
    }
    pr = MockChild(inst)
    assert pr.inputs == {"input": "data/RGB.byte.tif"}
    assert pr.indir == "data"
    assert pr.outdir == "data/out"
    assert pr.satellite == None


def test_get_fpath_with_indir() -> None:
    inst: Final[dict[str, Any]] = {
        "Inputs": {"input1": "RGB.byte.tif", "input2": "RGB.byte_copy1.tif"},
        "Indir": "data",
        "Outdir": "data/out",
    }
    pr = MockChild(inst)
    assert pr.inputs == {
        "input1": "data/RGB.byte.tif",
        "input2": "data/RGB.byte_copy1.tif",
    }
    assert pr.indir == "data"
    assert pr.outdir == "data/out"
    assert pr.satellite == None


def test_get_fpath_with_and_without_indir() -> None:
    inst: Final[dict[str, Any]] = {
        "Inputs": {
            "input1": "RGB.byte.tif",
            "input2": "RGB.byte_copy1.tif",
            "input3": "data/data_inner/RGB.byte_copy2.tif",  # no indir used here
        },
        "Indir": "data",
        "Outdir": "data/out",
    }
    pr = MockChild(inst)
    assert pr.inputs == {
        "input1": "data/RGB.byte.tif",
        "input2": "data/RGB.byte_copy1.tif",
        "input3": "data/data_inner/RGB.byte_copy2.tif",
    }
    assert pr.indir == "data"
    assert pr.outdir == "data/out"
    assert pr.satellite == None


def test_get_fpath_with_and_without_indir_input_list() -> None:
    inst: Final[dict[str, Any]] = {
        "Inputs": {
            "inputs": [
                "RGB.byte.tif",
                "RGB.byte_copy1.tif",
                "data/data_inner/RGB.byte_copy2.tif",
            ]
        },
        "Indir": "data",
        "Outdir": "data/out",
    }
    pr = MockChild(inst)
    assert pr.inputs == {
        "inputs": [
            "data/RGB.byte.tif",
            "data/RGB.byte_copy1.tif",
            "data/data_inner/RGB.byte_copy2.tif",
        ]
    }
    assert pr.indir == "data"
    assert pr.outdir == "data/out"
    assert pr.satellite == None
