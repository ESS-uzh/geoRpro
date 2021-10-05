import os
import json
import copy
from collections import OrderedDict
import pdb

import rasterio
import geoRpro.raster as rst



class Workflow:
    ACTIONS = {'RProcess', 'RMask', 'RReplace', 'RStack', 'RNdvi'}

    def __init__(self, instructions):
        self.instructions = copy.deepcopy(instructions)
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
    datadir = "/home/diego/work/dev/github/test_data"
    # Opening JSON file
    with open('./tests/driver_test_002.json') as json_file:
        wf_data = json.load(json_file, object_pairs_hook=OrderedDict)

    wf = Workflow(wf_data)
    wf.run_workflow()

    #scl_mask = rasterio.open(os.path.join(datadir, 'scl_mask.tif'))
    #b11_replaced = rasterio.open(os.path.join(datadir, 'b11_replaced.tif'))

    #scl_mask_arr, meta_scl_mask = rst.load(scl_mask)
    #b11, meta_b11 = rst.load(b11_replaced)
