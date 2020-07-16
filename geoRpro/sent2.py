import os
import re

class Sentinel2:
    """
    Collect and parse sentinel2 files.
    It uses the sentinel2 file name format specified after 6th December 2016.
    """

    PATTERN = re.compile(
            # example pattern to match
            # T37MBN_20170718T075211_B02.jp2
            r'(?P<tile_n>^T\d{2}MBN)'
            r'_(?P<date>[0-9]{8})'
            r'.*_(?P<band>B(02|03|04|05|06|07|08|8A|09|11|12)|AOT|SCL|TCI|WVP)'
            r'.*(?P<fext>jp2$)'
            )

    def __init__(self, dirpath):
        self.dirpath = dirpath
        self._lookup = {}
        for f in os.listdir(self.dirpath):
            match = Sentinel2.PATTERN.match(f)
            if match:
                # allow only unique bands to be in self.dirpath
                if match.group('band') in list(self._lookup.keys()):
                    raise ValueError(f"Duplicate band '{match.group('band')} found. "
                                     f"The directory must contain unique bands only. ")
                # build lookup table
                self._lookup[match.group('band')] = [os.path.join(self.dirpath,f),
                                                     match.group('date'),
                                                     match.group('tile_n')]
        self._verify_for_rfiles_match()

    def get_all_bands(self):
        return list(self._lookup.keys())

    
    def get_rfiles(self, *bands):
        rfiles = []
        for b in bands:
            try:
                rfiles.append((b, self._lookup[b][0]))
            except KeyError:
                print("This band: '{}' was not found".format(rf))
        return rfiles

    
    def get_datetake(self, band):
        try:
            return self._lookup[band][1]
        except KeyError:
            print("This band: '{}' was not found".format(band))
    
    
    def get_tile_number(self, band):
        try:
            return self._lookup[band][2]
        except KeyError:
            print("This band: '{}' was not found".format(band))


    def __repr__(self):
        return 'Sentinel2({})'.format(self.dirpath)

    def _verify_for_rfiles_match(self):
        assert bool(self._lookup) == True, "No file matching found at {}".format(self.dirpath)


if __name__ == '__main__':

    basedir = '/home/diego/work/dev/ess_diego/github/goeRpro_inp'

    # Work with scene classification 20m resolution
    imgdir = os.path.join(basedir,'S2A_MSIL2A_20190628T073621_N9999_R092_T37MBN_20191121T145522.SAFE/GRANULE/L2A_T37MBN_A020967_20190628T075427/IMG_DATA/R20m')
    # some testing
    p = Sentinel2(imgdir)
    print(p.get_all_bands())
    print(p.get_rfiles('B02'))
    print(p.get_datetake('B02'))
    print(p.get_tile_number('B02'))

