"""."""
import pyaccel
import pymodels
import numpy as np

from idanalysis import IDKickMap

BEAM_ENERGY = 3.0  # [GeV]
DEF_RK_S_STEP = 0.5  # [mm] seems converged for the measurement fieldmap grids
ROLL_OFF_RT = 5.0  # [mm]
SOLVE_FLAG = True

ID_PERIOD = 1200  # [mm]
NR_PERIODS = 1  #
NR_PERIODS_REAL_ID = 1  #
SIMODEL_ID_LEN = 1.200  # [m]
ID_KMAP_LEN = SIMODEL_ID_LEN  # [m]
RESCALE_KICKS = NR_PERIODS_REAL_ID/NR_PERIODS
RESCALE_LENGTH = 1
NOMINAL_GAP = 236.2
ID_FAMNAME = 'WLS'

SIMODEL_FITTED = False
MEAS_DATA_PATH = './meas-data/'
SHIFT_FLAG = False
FILTER_FLAG = False

FOLDER_DATA = './results/model/data/'

MEAS_FLAG = False
REAL_WIDTH = 50

widths = [REAL_WIDTH]
field_component = 'by'
var_param = 'gap'

FOLDER_BASE = '/home/gabriel/repos-dev/'

ID_CONFIGS = {

    'ID001': '2023-06-12_SWLS_Sirius_Model9.0_standard_I=20A_Aibox2m_' +
    'X=-10_10mm_Y=-3_3mm_Z=-1000_1000mm_ID=0001.txt',

    'ID002': '2023-06-12_SWLS_Sirius_Model9.0_standard_I=60A_Aibox2m_' +
    'X=-10_10mm_Y=-3_3mm_Z=-1000_1000mm_ID=0002.txt',

    'ID003': '2023-06-12_SWLS_Sirius_Model9.0_standard_I=100A_Aibox2m_' +
    'X=-10_10mm_Y=-3_3mm_Z=-1000_1000mm_ID=0003.txt',

    'ID004': '2023-06-12_SWLS_Sirius_Model9.0_standard_I=150A_Aibox2m_' +
    'X=-10_10mm_Y=-3_3mm_Z=-1000_1000mm_ID=0004.txt',

    'ID005': '2023-06-12_SWLS_Sirius_Model9.0_standard_I=200A_Aibox2m_' +
    'X=-10_10mm_Y=-3_3mm_Z=-1000_1000mm_ID=0005.txt',

    'ID006': '2023-06-12_SWLS_Sirius_Model9.0_standard_I=236.2A_Aibox2m_' +
    'X=-10_10mm_Y=-3_3mm_Z=-1000_1000mm_ID=0006.txt',

    'ID007': '2023-06-12_SWLS_Sirius_Model9.0_standard_I=240A_Aibox2m_' +
    'X=-10_10mm_Y=-3_3mm_Z=-1000_1000mm_ID=0007.txt',
}

# gaps = [60, 236.0]
gaps = [20.0, 60.0, 100.0, 150.0, 200.0, 236.2, 240.0]
phases = [0.0]
MEAS_GAPS = gaps
MEAS_PHASES = [00.00]

ORDERED_CONFIGS = [['ID001', 'ID002', 'ID003', 'ID004',
                    'ID005', 'ID006', 'ID007']]


def create_ids(
        fname, nr_steps=None, rescale_kicks=None, rescale_length=None):
    # create IDs
    nr_steps = nr_steps or 40
    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 1
    if MEAS_FLAG:
        rescale_length = 1
    IDModel = pymodels.si.IDModel
    wls = IDModel(
        subsec=IDModel.SUBSECTIONS.ID10SB,
        file_name=fname,
        fam_name='WLS', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length)
    ids = [wls, ]
    return ids
