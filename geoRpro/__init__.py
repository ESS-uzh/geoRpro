# import logging
# import sys
# from warnings import filterwarnings
#
# filterwarnings(
#    action="ignore",
#    category=DeprecationWarning,
#    # message="`np.bool` is a deprecated alias",
# )

# root_logger = logging.getLogger()
# root_logger.setLevel(logging.INFO)
#
# formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
#
# file_handler = logging.FileHandler('../../../geoRpro.log')
# file_handler.setFormatter(formatter)
#
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setFormatter(formatter)
#
# root_logger.addHandler(file_handler)
# root_logger.addHandler(console_handler)
import geoRpro.processing
