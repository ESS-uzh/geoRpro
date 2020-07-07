import os
import re

class Sentinel2:

    PATTERN = re.compile(
            # example pattern to match
            # T37MBN_20170718T075211_B02.jp2
            r'(?P<tile_n>^T\d{2}MBN)'
            r'_(?P<date>[0-9]{8})'
            r'.*_((?P<band>B0(2|3|4|8)))'
            r'.*(?P<fext>jp2$)'
            )

    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.lookup = {'B02': 'Sent2_B02', # BLUE
                       'B03': 'Sent2_B03', # GREEN
                       'B04': 'Sent2_B04', # RED
                       'B08': 'Sent2_B08'} # NIR
        self.fpaths = []
        for f in os.listdir(self.dirpath):
            match = Sentinel2.PATTERN.match(f)
            if match:
                self.fpaths.append((os.path.join(self.dirpath,
                                                 f),
                                    self.lookup[match.group('band')]))
                self.date = match.group('date')
                self.tile = match.group('tile_n')
        self._verify_rfiles_not_empty()
        # sort by the Band number
        self.sfpaths = sorted(self.fpaths)

    def __repr__(self):
        return 'Sentinel2({})'.format(self.dirpath)

    def _verify_rfiles_not_empty(self):
        assert len(self.fpaths) > 0, "No file matching found at {}".format(self.dirpath)


if __name__ == '__main__':

    basedir = '/home/diego/work/dev/ess_diego/Data_Diego'

    # Work with scene classification 20m resolution
    imgdir = os.path.join(basedir,'Hanneke/S2A_MSIL2A_20190628T073621_N9999_R092_T37MBN_20191121T145522.SAFE/GRANULE/L2A_T37MBN_A020967_20190628T075427/IMG_DATA/R10m')

    p = Sentinel2(imgdir)
    print(p.sfpaths)
    print(p.date)
    print(p.tile)
