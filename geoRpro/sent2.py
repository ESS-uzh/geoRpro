import os
import re
import pdb

class Sentinel2:
    """
    Collect and parse sentinel2 files.
    It uses the sentinel2 file name format specified after 6th December 2016.
    """

    PATTERN = re.compile(
            # example pattern to match
            # T37MBN_20170718T075211_B02.jp2
            r'(?P<tile_n>^T\d{2}\D{3})'
            r'_(?P<date>[0-9]{8})'
            r'.*_(?P<band>B(02|03|04|05|06|07|08|8A|09|11|12)|AOT|SCL|TCI|WVP)'
            r'(?P<attr>([\w]*))'
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
                if match.group('attr'):
                    key_lookup = match.group('band')+match.group('attr')
                else:
                    key_lookup = match.group('band')
                self._lookup[key_lookup] = [os.path.join(self.dirpath,f),
                                                     match.group('date'),
                                                     match.group('tile_n')]
        self._verify_for_rfiles_match()

    def get_all_bands(self):
        return list(self._lookup.keys())


    def get_fpaths(self, *bands):
        rfiles = []
        for b in bands:
            try:
                rfiles.append(self._lookup[b][0])
            except KeyError:
                print("This band: '{}' was not found".format(b))
        return rfiles

    def get_fpath(self, band):
        fpath = self._lookup[band][0]
        return fpath

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
