import os
import json
import copy
from collections import OrderedDict
import pdb

import rasterio


class Workflow:
    ACTIONS = {'RPrepro', 'RMask', 'RReplace', 'RStack', 'RIndex'}

    def __init__(self, instructions):
        self.instructions = copy.deepcopy(instructions)
        pdb.set_trace()
        self.actions = []
        self._parse_wf()

    def _parse_wf(self):
        for action_name, action_param in self.instructions.items():

            if action_name not in self.ACTIONS:
                raise ValueError(f'{action_name} is not a valid action')

            self.actions.append(action_name)

    def run_workflow(self):

        for action in self.actions:
            # get instruction for this action
            instruction = self.instructions[action]
            action_cls = getattr(__import__('processing'), action)
            action = action_cls(instruction)
            action.run()


if __name__ == '__main__':

    #### --- Use ./scripts/driver_example_window.json to drive workflow --- ###
    # --- Steps:
    # --- RProcess -> prepare raster data (same spatial res, same AOI etc..)
    # --- RMask -> create a boolean raster mask (True to be masked , False not)
    # --- RI_ndivi etc -> Calc indexes -> ndvi, etc..
    # --- RReplace -> Use mask bool raster to replace the pixel values of
    # other raster with 9999 at the True position
    # --- RStack -> stack up all the rasters

    with open('./scripts/driver_example_window.json') as json_file:
        wf_data = json.load(json_file, object_pairs_hook=OrderedDict)

    wf = Workflow(wf_data)
    wf.run_workflow()
