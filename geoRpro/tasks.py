import json


json_arr1 = '''
{
INDIR: {"/home/diego/work/dev/ess_diego/github/goeRpro_inp/S2A_MSIL2A_20190628T073621_N9999_R092_T37MBN_20191121T145522.SAFE/GRANULE/L2A_T37MBN_A020967_20190628T075427/IMG_DATA/R10m"}

OUTDIR: {"/home/diego/work/dev/ess_diego/github/goeRpro_out/test_workflow"}

WORKFLOW:{
    "task1": {
    "inp": "T37MBN_20190628T073621_SCL_20m.jp2",
    "operations": {
       "resample_raster": {"scale": 2},
       "get_aoi": {"window": (0, 0, 4000, 2000), "write": "_resampled_aoi.tiff"},
       "binarize": {"values": (3, 7, 8, 9, 10), "write": "_mask_aoi.tiff"}
        }
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

#rules = {'create_mask': ao.create_mask_arr,
#         'mask': ao.mask_and_fill,
#         'resample': ro.resample_raster
#         }

class WorkFlowParser:
    """

    """

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


class GeoRWf:
    """
    Collect and run chained raster operations
    Use instances of geoRpro.RasterOp class
    """
    def __init__(self, srcs = None, all_opers = None):

        if all_opers is None:
            all_opers = []

        if srcs is None:
            srcs = []
    
        self.all_opers = all_opers
        self.srcs = srcs
        self.meta = None
        self.arr = None

    def add_oper(self, oper):
        self.all_opers.append(oper)

    def run_opers(self):
        for oper in self.all_opers:
            oper.run()


if __name__ == "__main__":

    arr1 = json.loads(json_arr1)
    p1 = TasksParser(arr1)
#
#    arr2 = json.loads(json_arr2)
#    p2 = TasksParser(arr2)
#
#    arr3 = json.loads(json_arr3)
#    p3 = TasksParser(arr3)
