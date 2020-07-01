import aope as ao
import rope as ro

import json
from contextlib import ExitStack

json_arr1 = '''
{
    "mask_raster": {
    "inp": "tests/T37MBN_20190628T073621_B04_10m.jp2",
    "out": "tests/T37MBN_20190628T073621_B04_10m_masked.jp2",
    "operations": {
       "resample_raster": {
            "inp": "tests/T37MBN_20190628T073621_SCL_20m.jp2",
            "scale": 2},
        "create_mask": {
            "values"; [3,7,8,9,10]},
        "mask": {
            "fill_value": 0}
                 }
         }
}
'''

json_arr2 = '''
{
    "mask_raster": {
    "inp": "tests/T37MBN_20190628T073621_B04_10m.jp2",
    "out": "tests/T37MBN_20190628T073621_B04_10m_masked.jp2"
         }
}
'''

json_arr3 = '''
{
    "maskkk_raster": {
    "inp": "tests/T37MBN_20190628T073621_B04_10m.jp2",
    "out": "tests/T37MBN_20190628T073621_B04_10m_masked.jp2"
         }
}
'''

rules = {'create_mask': ao.create_mask_arr,
         'mask': ao.mask_and_fill,
         'resample': ro.resample_raster
         }

class TasksParser:

    TASKS = ['mask_raster', 'ndvi', 'extract_pixel_values']
    ATTR = ['inp', 'out', 'operations']

    def __init__(self, json_dict):
        self.json_dict = json_dict
        self._check_json()

    def _check_json(self):
        for task, cont1 in self.json_dict.items():
            if task in self.TASKS:
                for attr, cont2 in cont1.items():
                    if attr in self.ATTR:
                        return 'ok'
                    raise ValueError
            raise ValueError

    def get_json_dict(self):
        return self.json_dict

    def list_tasks(self):
        return list(self.get_json_dict().keys())

    def get_task(self, task):
        return self.get_json_dict()[task]

    def get_task_inp(self, task):
        return self.get_task(task)['inp']

    def get_task_out(self, task):
        return self.get_task(task)['out']

    def list_task_operations(self, task):
        try:
            return list(self.get_task(task)['operations'].keys())
        except KeyError:
            return None

    def get_task_operation(self, task, operation):
        try:
            return self.get_task(task)['operations'][operation]
        except KeyError:
            return None


class MaskRaster:

    TASKNAME = 'mask_raster'

    def __init__(self, p):
        self.p = p
        self.task = p.get_task(self.TASKNAME)

    def build_task(self):
        for operation in self.p.list_task_operations(self.TASKNAME):
            rules[operation]()



if __name__ == "__main__":

    arr1 = json.loads(json_arr1)
    p1 = TasksParser(arr1)

    arr2 = json.loads(json_arr2)
    p2 = TasksParser(arr2)

    arr3 = json.loads(json_arr3)
    p3 = TasksParser(arr3)
