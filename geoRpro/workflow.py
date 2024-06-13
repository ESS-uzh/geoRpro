from typing import Generator, Any, Final, Literal

import sys
import os

os.environ["USE_PYGEOS"] = "0"
import geopandas

import copy

import pdb

import rasterio


class Workflow:
    ACTIONS: set[str] = {
        "RPrepro",
        "RMask",
        "RReplace",
        "RNormalize",
        "RStack",
        "RIndex",
        "RExtractBands",
        "RExtractPoints",
    }

    def __init__(self, instructions: dict[str, Any]) -> None:
        self.instructions: dict[str, Any] = copy.deepcopy(instructions)
        self.actions: list = []
        self._parse_wf()

    def _parse_wf(self) -> None:
        action_name: str
        action_param: dict[str, Any]
        for action_name, action_param in self.instructions.items():
            if action_name not in self.ACTIONS:
                raise ValueError(f"{action_name} is not a valid action")

            self.actions.append(action_name)

    def run_workflow(self):
        for action in self.actions:
            # get instruction for this action
            instruction = self.instructions[action]
            geoRpro = __import__("geoRpro")
            action_cls = getattr(geoRpro.processing, action)
            action = action_cls(instruction)
            action.run()


# if __name__ == "__main__":
#### --- Use ./scripts/driver_example_window.json to drive workflow --- ###
# --- Steps:
# --- RProcess -> prepare raster data (same spatial res, same AOI etc..)
# --- RMask -> create a boolean raster mask (True to be masked , False not)
# --- RI_ndivi etc -> Calc indexes -> ndvi, etc..
# --- RReplace -> Use mask bool raster to replace the pixel values of
# other raster with 9999 at the True position
# --- RStack -> stack up all the rasters

#    main()
