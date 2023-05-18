"""."""

import numpy as np
import pyaccel
import pymodels

from idanalysis import IDKickMap


ID_PERIOD = 180.0  # [mm]

FOLDER_BASE = '/home/gabriel/repos-dev/'
# FOLDER_BASE = '/home/ximenes/repos-dev/'

DATA_PATH = 'si-wls/model-08/simulation/magnetic/integrals-and-multipoles/diff_currents/X=-10_10mm_Z=-600_600mm/'


ID_CONFIGS = {
    'I10A': '2022-08-05_SWLS_Sirius_Model8.0_Currents_X=-10_10mm_Z=-600_600mm_I=10A.txt',
    'I50A': '2022-08-05_SWLS_Sirius_Model8.0_Currents_X=-10_10mm_Z=-600_600mm_I=50A.txt',
    'I100A': '2022-08-05_SWLS_Sirius_Model8.0_Currents_X=-10_10mm_Z=-600_600mm_I=100A.txt',
    'I200A': '2022-08-05_SWLS_Sirius_Model8.0_Currents_X=-10_10mm_Z=-600_600mm_I=200A.txt',
    'I228A': '2022-08-05_SWLS_Sirius_Model8.0_Currents_X=-10_10mm_Z=-600_600mm_I=228A.txt',
    'I250A': '2022-08-05_SWLS_Sirius_Model8.0_Currents_X=-10_10mm_Z=-600_600mm_I=250A.txt',
    'I300A': '2022-08-05_SWLS_Sirius_Model8.0_Currents_X=-10_10mm_Z=-600_600mm_I=300A.txt',
    }
