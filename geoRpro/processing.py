from typing import Generator, Any, Final, Literal, Callable
from nptyping import NDArray, UInt8, Int32, Float32, Shape, Bool

import os
import copy
import logging
from contextlib import ExitStack

import rasterio
from rasterio.io import DatasetReader
import geopandas as gpd
from geoRpro.sent2 import Sentinel2
import geoRpro.raster as rst
import geoRpro.extract as ext
import geoRpro.io as io
from geoRpro.base_processing import ProcessBase

import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RPrepro(ProcessBase):
    """
    Pre-processing of georaster

    Attributes
    ----------

    instructions: dict
                  List of attributes necessary to create an
                  instance of the class and to the run main method (self.run)
                  Required and optional attributes are specified in the
                  INSTRUCTIONS class attribute

    res: int
         Spacial resolution that applies to geo-rasters

    polygon: list (opt)
             Extract a polygon from a geo-raster

             Eg. ['*.shp', 'crs', index]

             shp: str
                  Full path to the shape file

             crs: str
                  Coord ref sys of the geo-raster from which the polygon will be
                  extracted
             index: int
                    Specify a polygon if the shp file contains multiple polygons

    window: list (opt)
            Extract a window from a georaster
            Eg. [(0, 5), (0, 10)] # 5 first rows, 10 first columns


    layer: int (opt)
           Extract a layer from a geo-raster


    """

    INSTRUCTIONS: Final[set[str]] = {
        "Inputs",
        "Indir",
        "Outdir",
        "Satellite",
        "Res",
        "Polygon",
        "Window",
        "Layers",
    }

    def __init__(self, instructions: dict[Any, Any]) -> None:
        super().__init__(instructions)
        self.res = None
        self._polygon = None
        self.window = None
        self.layers = None
        self._parse_args()

    @property
    def polygon(self) -> Any | None:
        return self._polygon

    @polygon.setter
    def polygon(self, poly_param) -> None:
        shape_file = poly_param[0]
        crs = poly_param[1]
        index = poly_param[2]

        gdf = gpd.read_file(shape_file)
        gdf = gdf.to_crs(f"EPSG:{crs}")
        self._polygon = gdf["geometry"][index]

    def _parse_args(self):
        # NOte: Values inserted in data should be checked, property ?!
        self.res = self.instructions.get("Res")
        if self.instructions.get("Polygon"):
            self.polygon = self.instructions.get("Polygon")
        if self.instructions.get("Window"):
            self.window = tuple(self.instructions.get("Window"))
        self.layer = self.instructions.get("Layer")

    def run(self):
        with ExitStack() as stack_files:
            self.outputs = {
                k: stack_files.enter_context(rasterio.open(v))
                for (k, v) in self.inputs.items()
            }

            with ExitStack() as stack_action:
                for name, src in self.outputs.items():
                    print(f"Raster: {name} has {src.res[0]} m resolution..")

                    # resample to match res param
                    if self.res and src.res[0] != self.res:
                        print(
                            f"Raster: {name} will be resampled to {self.res} m resolution.."
                        )
                        scale_factor = src.res[0] / self.res
                        arr, meta = rst.load_resample(src, scale_factor)
                        src = stack_action.enter_context(io.to_src(arr, meta))

                    if self.layers:
                        print(f"Selected layer: {self.layer}")
                        arr, meta = rst.load(src, bands=self.layer)
                        src = stack_action.enter_context(io.to_src(arr, meta))

                    if self.polygon:
                        print("Selected a polygon as AOI")
                        arr, meta = rst.load_polygon(src, self.polygon)
                        src = stack_action.enter_context(io.to_src(arr, meta))

                    elif self.window:
                        print(f"Selected a window: {self.window} as AOI")
                        arr, meta = rst.load(src, window=self.window)
                        src = stack_action.enter_context(io.to_src(arr, meta))

                    outdir = self._create_outdir()

                    fpath = os.path.join(outdir, name + ".tif")
                    io.write_raster([src], src.profile, fpath)
                    self.outputs[name] = fpath


class RMask(ProcessBase):
    """
    Create a boolean (0, 1) mask geo-raster

    Attributes
    ----------

    instructions: dict
                  List of attributes necessary to create an
                  instance of the class and to the run main method (self.run)
                  Required and optional attributes are specified in the
                  INSTRUCTIONS class attribute

    values: list
            Pixel values of the input rasters equal to *values* will be converted to 1
            the rest will be converted to 0


    """

    INSTRUCTIONS = {"Inputs", "Indir", "Outdir", "Satellite", "Values"}

    def __init__(self, instructions):
        super().__init__(instructions)
        self.mask_val = self.instructions.get("Values")

    def run(self):
        with ExitStack() as stack_files:
            self.outputs = {
                k: stack_files.enter_context(rasterio.open(v))
                for (k, v) in self.inputs.items()
            }

            for name, src in self.outputs.items():
                print(f"Masking: {name}")
                arr, meta = rst.load(src)
                arr_mask, meta_mask = rst.mask_vals(arr, meta, self.mask_val)

                outdir = self._create_outdir()

                fpath = os.path.join(outdir, name + ".tif")
                io.write_array_as_raster(arr_mask.mask, meta_mask, fpath)
                self.outputs[name] = fpath


class RReplace(ProcessBase):
    """
    Create a geo-raster with new values at locations where a mask geo-raster
    is equal to 1

    Attributes
    ----------

    instructions: dict
                  List of attributes necessary to create an
                  instance of the class and to the run main method (self.run)
                  Required and optional attributes are specified in the
                  INSTRUCTIONS class attribute

    replace: list
             Extract a polygon from a geo-raster

             Eg. ['*.tif', fill_value]

             .tif: str
                   Full path to a mask geo-raster

             fill_value: int
                   Value that will be used to fill the masked positions


    """

    INSTRUCTIONS = {"Inputs", "Indir", "Outdir", "Satellite", "Replace"}

    def __init__(self, instructions):
        super().__init__(instructions)
        self.fpath_mask, self.value = self.instructions.get("Replace")

    def run(self):
        with ExitStack() as stack_files:
            self.outputs = {
                k: stack_files.enter_context(rasterio.open(v))
                for (k, v) in self.inputs.items()
            }
            src_mask = stack_files.enter_context(
                rasterio.open(os.path.join(self.indir, self.fpath_mask))
            )
            mask, _ = rst.load(src_mask)

            for name, src in self.outputs.items():
                print(f"Replace values with {self.value} at mask array position")
                arr, meta = rst.load(src)
                assert (
                    mask.shape == arr.shape
                ), "Array and mask must the have same shape"
                arr_replaced = rst.apply_mask(arr, mask, fill_value=self.value)

                outdir = self._create_outdir()

                fpath = os.path.join(outdir, name + ".tif")
                io.write_array_as_raster(arr_replaced, meta, fpath)
                self.outputs[name] = fpath


class RStack(ProcessBase):
    INSTRUCTIONS = {"Inputs", "Indir", "Outdir", "Satellite", "Dtype"}

    def __init__(self, instructions):
        super().__init__(instructions)
        self.dtype = self.instructions.get("Dtype", "int32")

    def _check_metadata(self, srcs):
        test_src = srcs[0]
        end_srcs = srcs[1:]

        for src in end_srcs:
            if src.crs.to_epsg() != test_src.crs.to_epsg():
                raise ValueError("Raster data must have the same CRS")
            if src.width != test_src.width or src.height != test_src.height:
                raise ValueError("Raster data must have the same size")
            if src.res != test_src.res:
                raise ValueError("Raster data must have the same spacial resolution")

    def run(self):
        self.outputs = {}
        with ExitStack() as stack_files:
            for name, values in self.inputs.items():
                print(f"Start stack procedure for {name}")
                srcs = [stack_files.enter_context(rasterio.open(v)) for v in values]
                self._check_metadata(srcs)
                self.outputs[name] = srcs
                metadata = srcs[0].profile
                metadata.update(driver="GTiff")
                metadata.update(count=len(self.outputs[name]))
                metadata.update(dtype=self.dtype)

                outdir = self._create_outdir()

                fpath = os.path.join(outdir, name + ".tif")
                io.write_raster(srcs, metadata, fpath)
                self.outputs[name] = fpath


class RIndex(ProcessBase):
    INSTRUCTIONS: set[str] = {"Inputs", "Indir", "Outdir", "Satellite", "Scale"}

    def __init__(self, instructions) -> None:
        super().__init__(instructions)
        self.scale_factor: Literal[1, 1000] = self.instructions.get("Scale", 1000)

    def _check_metadata(self, srcs) -> None:
        test_src: DatasetReader = srcs[0]
        end_srcs: list[DatasetReader] = srcs[1:]

        src: DatasetReader
        for src in end_srcs:
            if src.crs.to_epsg() != test_src.crs.to_epsg():
                raise ValueError("Raster data must have the same CRS")
            if src.width != test_src.width or src.height != test_src.height:
                raise ValueError("Raster data must have the same size")
            if src.res != test_src.res:
                raise ValueError("Raster data must have the same spacial resolution")

    def calc_index(
        self, name: str, metadata: dict[str, Any], scale: Literal[1, 1000]
    ) -> Callable:
        index = rst.Indexes(metadata, scale)
        if name == "ndvi":
            return index.ndvi
        if name == "nbr":
            return index.nbr
        if name == "bsi":
            return index.bsi
        if name == "ndwi":
            return index.ndwi
        if name == "cai":
            return index.cai
        if name == "evi":
            return index.evi
        if name == "mndvi705":
            return index.mndvi705
        if name == "ndmi":
            return index.ndmi
        if name == "ndvi705":
            return index.ndvi705
        if name == "rvsi":
            return index.rvsi
        if name == "vari":
            return index.vari
        if name == "vgi":
            return index.vgi
        if name == "wi":
            return index.wi
        raise ValueError(f"Index {name} does not exist")

    def run(self) -> None:
        self.outputs: dict[str, Any] = {}

        with ExitStack() as stack_files:
            for name, values in self.inputs.items():
                print(f"Calculating {name}")
                srcs: list[DatasetReader] = [
                    stack_files.enter_context(rasterio.open(v)) for v in values
                ]
                self._check_metadata(srcs)
                self.outputs[name] = srcs
                metadata_index: dict[str, Any] = srcs[0].profile
                metadata_index.update(driver="GTiff")

                index: Callable = self.calc_index(
                    name, metadata_index, scale=self.scale_factor
                )
                arr: NDArray
                meta: dict[str, Any]
                arr, meta = index(*srcs)

                outdir = self._create_outdir()

                fpath = os.path.join(outdir, name + ".tif")
                io.write_array_as_raster(arr, meta, fpath)
                self.outputs[name] = fpath


class RExtractBands(ProcessBase):
    INSTRUCTIONS: set[str] = {"Inputs", "Indir", "Outdir", "Bands"}

    def __init__(self, instructions) -> None:
        super().__init__(instructions)
        self.bands: list = self.instructions.get("Bands")

    def run(self) -> None:
        with ExitStack() as stack_files:
            self.outputs = {
                k: stack_files.enter_context(rasterio.open(v))
                for (k, v) in self.inputs.items()
            }

            for idx, (name, src) in enumerate(self.outputs.items()):
                # print(f"Band: {name}")
                arr, meta = rst.load(src, bands=[idx + 1])

                outdir = self._create_outdir()

                fpath = os.path.join(outdir, name + ".tif")
                io.write_array_as_raster(arr, meta, fpath)
                self.outputs[name] = fpath


class RExtractPoints(ProcessBase):
    INSTRUCTIONS = {"Inputs", "Indir", "Outdir", "Points", "ClassName", "Id", "Masked"}

    def __init__(self, instructions):
        super().__init__(instructions)
        self.points = self.instructions.get("Points")
        self.masked = self.instructions.get("Masked", 9999)

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points_param) -> None:
        shape_file = points_param[0]
        crs = points_param[1]

        gdf = gpd.read_file(shape_file)
        gdf = gdf.to_crs(f"EPSG:{crs}")
        self.classname = self.instructions.get("ClassName")
        self.id = self.instructions.get("Id")

        gdf["classname"] = self.classname
        gdf["id"] = self.id

        self._points = gdf

    def _check_metadata(self, srcs) -> None:
        if srcs[0].crs.to_epsg() != self.points.crs.to_epsg():
            raise ValueError("Raster and Points location must have the same CRS")

    def run(self) -> Any:
        self.outputs: dict = {}
        with ExitStack() as stack_files:
            for name, values in self.inputs.items():
                print(f"Start extract procedure for {name}")
                srcs = [stack_files.enter_context(rasterio.open(v)) for v in values]
                self._check_metadata(srcs)
                self.outputs[name] = srcs

                extracted = ext.DataExtractor.extract(srcs[0], self.points, self.masked)

                outdir = self._create_outdir()

                fpath = os.path.join(outdir, name + ".json")
                extracted.save(fpath)
                self.outputs[name] = fpath
        return extracted


if __name__ == "__main__":
    ORIGIN: Final[
        str
    ] = "/home/diego/work/dev/data/test_data/S2A_MSIL2A_20190906T073611_N0213_R092_T37MBN_20190906T110000.SAFE/GRANULE/L2A_T37MBN_A021968_20190906T075543/IMG_DATA"
    SHAPE: Final[
        str
    ] = "/home/diego/work/dev/github/geoRpro/tests/data/point_to_extract.shp"
    DATADIR: Final[str] = "/home/diego/work/dev/data/test_data_extract"
    # inst0: Final[dict[str, Any]] = {
    #    "Inputs": {
    #        "B02": "B02_10m",
    #        "SCL": "SCL_20m",
    #        "B04": "B04_10m",
    #        "B08": "B08_10m",
    #        "B12": "B12_20m",
    #    },
    #    "Indir": ORIGIN,
    #    "Satellite": "Sentinel2",
    #    "Outdir": DATADIR,
    #    # "Window": [(0, 10), (5475, 5490)],
    #    "Polygon": [SHAPE, "32737", 1],
    #    "Res": 20,
    # }
    # inst1: Final[dict[str, Any]] = {
    #    "Inputs": {"stack1": ["B02.tif", "B04.tif", "B08.tif"]},
    #    "Indir": DATADIR,
    # }
    # prepro = RPrepro(inst0)
    # prepro.run()
    # rstackpro = RStack(inst1)
    # rstackpro.run()

    # inst1 = {"Inputs": {"SCL_mask": "SCL.tif",
    #                   },
    #         "Indir" : DATADIR,
    #         #"Values" : [4]
    #         "Values": [3, 7, 8, 9, 10]}

    # inst2 = {"Inputs": {"B02_masked": "B02.tif",
    #                    "B04_masked": "B04.tif",
    #                    "B08_masked": "B08.tif",
    #                    "B12_masked": "B12.tif"
    #                   },
    #         "Indir" : DATADIR,
    #         "Replace" : ['SCL_mask.tif', 9999]}

    # inst3 = {"Inputs": {"ndvi" : ["B02_masked.tif", "B04_masked.tif"],
    #                    "nbr" : ["B08_masked.tif", "B12_masked.tif"]},
    #         "Scale": 1,
    #         "Indir" : DATADIR}

    # inst4 = {"Inputs": {"stack1" : ["B02_masked.tif", "B04_masked.tif", "ndvi.tif"],
    #                    "stack2" : ["B08_masked.tif", "B12_masked.tif", "nbr.tif"]},
    #         "Dtype" : "float32",
    #         "Indir" : DATADIR}

    # prepro = RPrepro(inst0)
    # prepro.run()
    # maskpro = RMask(inst1)
    # maskpro.run()
    # replacepro = RReplace(inst2)
    # replacepro.run()
    # rindexpro = RIndex(inst3)
    # rindexpro.run()
    # rstackpro = RStack(inst4)
    # rstackpro.run()

    # check outputs
    # arr_scl, meta_scl = rst.load(rasterio.open(prepro.outputs['SCL']))
    # arr_b03, meta_b03 = rst.load(rasterio.open(prepro.outputs['B03']))
    # arr_b08, meta_b08 = rst.load(rasterio.open(prepro.outputs['B08']))
    # arr_scl_mask, meta_scl_mask = rst.load(rasterio.open(maskpro.outputs['SCL_mask']))
    # arr_b03_masked, meta_b03_masked = rst.load(rasterio.open(replacepro.outputs['B03_masked']))
    # arr_b08_masked, meta_b08_masked = rst.load(rasterio.open(replacepro.outputs['B08_masked']))
    # arr_ndvi, meta_ndvi = rst.load(rasterio.open(rindexpro.outputs['ndvi']))
    # arr_stack1, meta_stack1 = rst.load(rasterio.open(rstackpro.outputs['stack1']))
    # arr_stack2, meta_stack2 = rst.load(rasterio.open(rstackpro.outputs['stack2']))
