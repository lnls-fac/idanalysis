"""."""
import pyaccel
import pymodels
import numpy as np

from imaids.models import DeltaSabia
from idanalysis import IDKickMap
from mathphys.functions import load_pickle

BEAM_ENERGY = 3.0  # [GeV]
DEF_RK_S_STEP = .5  # [mm] seems converged for the measurement fieldmap grids
ROLL_OFF_RT = 5.0  # [mm]
ROLL_OFF_POS = 5.0  # [mm]
ROLL_OFF_PLANE = 'x'
FIELD_COMPONENT = 'by'
SOLVE_FLAG = True

ID_PERIOD = 52.5  # [mm]
NR_PERIODS = 21  #
NR_PERIODS_REAL_ID = 21  #
SIMODEL_ID_LEN = 1.200  # [m]
ID_KMAP_LEN = SIMODEL_ID_LEN  # [m]
RESCALE_KICKS = NR_PERIODS_REAL_ID/NR_PERIODS
RESCALE_LENGTH = 1
NOMINAL_GAP = 13.6
ID_FAMNAME = 'DELTA52'

SIMODEL_FITTED = False
FIT_PATH = '/home/gabriel/Desktop/my-data-by-day/2023-05-15-SI_low_coupling/fitting_ref_config_before_low_coupling_strengths.pickle'
SHIFT_FLAG = True
FILTER_FLAG = False

FOLDER_DATA = './results/model/data/'
MEAS_DATA_PATH = './meas-data/id-sabia/model-03/measurement/magnetic/hallprobe/'
MEAS_FLAG = True

gaps = [13.125, 26.250]
phases = [0, -13.125, -26.25]
# gaps = [0]
# phases = [0]

FOLDER_BASE = '/home/gabriel/repos-dev/'

ID_CONFIGS = {

    # shimming E
    'ID4818': 'delta_sabia_shimmingE/2023-09-04_' +
    'DeltaSabia_Phase01_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4818.dat',

    'ID4819': 'delta_sabia_shimmingE/2023-09-04_' +
    'DeltaSabia_Phase02_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4819.dat',

    'ID4820': 'delta_sabia_shimmingE/2023-09-04_' +
    'DeltaSabia_Phase03_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4820.dat',

    'ID4821': 'delta_sabia_shimmingE/2023-09-04_' +
    'DeltaSabia_Phase04_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4821.dat',

    'ID4822': 'delta_sabia_shimmingE/2023-09-04_' +
    'DeltaSabia_Phase05_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4822.dat',

    'ID4823': 'delta_sabia_shimmingE/2023-09-04_' +
    'DeltaSabia_Phase06_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4823.dat',

    'ID4824': 'delta_sabia_shimmingE/2023-09-05_' +
    'DeltaSabia_Phase07_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4824.dat',

    # Final configuration
        # phase 0
    'ID4862': 'delta_sabia_final/LinH/2023-10-25_' +
    'DeltaSabia_Phase=0mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4862.dat',

    'ID4863': 'delta_sabia_final/LinH/2023-10-25_' +
    'DeltaSabia_Phase=0mm_GV=6.5625mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4863.dat',

    'ID4864': 'delta_sabia_final/LinH/2023-10-25_' +
    'DeltaSabia_Phase=0mm_GV=13.125mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4864.dat',

    'ID4865': 'delta_sabia_final/LinH/2023-10-25_' +
    'DeltaSabia_Phase=0mm_GV=19.6875mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4865.dat',

    'ID4866': 'delta_sabia_final/LinH/2023-10-25_' +
    'DeltaSabia_Phase=0mm_GV=26.25mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4866.dat',

        # phase -26.25
    'ID4872': 'delta_sabia_final/LinV/2023-10-26_' +
    'DeltaSabia_Phase=-26.25mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4872.dat',

    'ID4873': 'delta_sabia_final/LinV/2023-10-26_' +
    'DeltaSabia_Phase=-26.25mm_GV=6.5625mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4873.dat',

    'ID4874': 'delta_sabia_final/LinV/2023-10-26_' +
    'DeltaSabia_Phase=-26.25mm_GV=13.125mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4874.dat',

    'ID4875': 'delta_sabia_final/LinV/2023-10-26_' +
    'DeltaSabia_Phase=-26.25mm_GV=19.6875mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4875.dat',

    'ID4876': 'delta_sabia_final/LinV/2023-10-27_' +
    'DeltaSabia_Phase=-26.25mm_GV=26.25mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4876.dat',

        # phase -13.125
    'ID4867': 'delta_sabia_final/CircN/2023-10-25_' +
    'DeltaSabia_Phase=-13.125mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4867.dat',

    'ID4868': 'delta_sabia_final/CircN/2023-10-26_' +
    'DeltaSabia_Phase=-13.125mm_GV=6.5625mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4868.dat',

    'ID4869': 'delta_sabia_final/CircN/2023-10-26_' +
    'DeltaSabia_Phase=-13.125mm_GV=13.125mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4869.dat',

    'ID4870': 'delta_sabia_final/CircN/2023-10-26_' +
    'DeltaSabia_Phase=-13.125mm_GV=19.6875mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4870.dat',

    'ID4871': 'delta_sabia_final/CircN/2023-10-26_' +
    'DeltaSabia_Phase=-13.125mm_GV=26.25mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4871.dat',

        # phase 13.125
    'ID4877': 'delta_sabia_final/CircP/2023-10-27_' +
    'DeltaSabia_Phase=13.125mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4877.dat',

    'ID4878': 'delta_sabia_final/CircP/2023-10-27_' +
    'DeltaSabia_Phase=13.125mm_GV=-6.5625mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4878.dat',

    'ID4879': 'delta_sabia_final/CircP/2023-10-27_' +
    'DeltaSabia_Phase=13.125mm_GV=-13.125mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4879.dat',

    'ID4880': 'delta_sabia_final/CircP/2023-10-27_' +
    'DeltaSabia_Phase=13.125mm_GV=-19.6875mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4880.dat',

    'ID4881': 'delta_sabia_final/CircP/2023-10-27_' +
    'DeltaSabia_Phase=13.125mm_GV=-26.25mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4881.dat',

        # zero K
    'ID4884': 'delta_sabia_final/zeroK/2023-10-27_' +
    'DeltaSabia_Phase=6.5625mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4884.dat',

    'ID4882': 'delta_sabia_final/zeroK/2023-10-27_' +
    'DeltaSabia_Phase=-6.5625mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4882.dat',

    'ID4883': 'delta_sabia_final/zeroK/2023-10-27_' +
    'DeltaSabia_Phase=-19.6875mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4883.dat',

    }


phases = [0, -13.125, 13.125, -26.25]
dgv = [0, 6.5625, 13.125, 19.6875, 26.25]

# phases = [6.5625, -6.5625, 19.6875]
# dgv = [0]

dp_dgv_dict = {(0, 0): 'ID4862', (0, 6.5625): 'ID4863', (0, 13.125): 'ID4864',
               (0, 19.6875): 'ID4865', (0, 26.25): 'ID4866',

               (-13.125, 0): 'ID4867', (-13.125, 6.5625): 'ID4868', (-13.125, 13.125): 'ID4869',
               (-13.125, 19.6875): 'ID4870', (-13.125, 26.25): 'ID4871',

               (13.125, 0): 'ID4877', (13.125, 6.5625): 'ID4878', (13.125, 13.125): 'ID4879',
               (13.125, 19.6875): 'ID4880', (13.125, 26.25): 'ID4881',

               (-26.25, 0): 'ID4872', (-26.25, 6.5625): 'ID4873', (-26.25, 13.125): 'ID4874',
               (-26.25, 19.6875): 'ID4875', (-26.25, 26.25): 'ID4876',

               (6.5625, 0): 'ID4884', (-6.5625, 0): 'ID4882', (-19.6875, 0): 'ID4883'}

# dp_dgv_dict = {(0, 0): 'ID4836', (0, 26.25): 'ID4837',
#                (-26.25, 26.25): 'ID4838', (-13.125, 26.25): 'ID4839',
#                (0, 13.125): 'ID4840', (-26.25, 13.125): 'ID4841',
#                (-13.125, 13.125): 'ID4842'}

CONFIG_DICT = {'params': ['phase', 'dgv'],
               'configs': dp_dgv_dict}


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
    delta52 = IDModel(
        subsec=IDModel.SUBSECTIONS.ID10SB,
        file_name=fname,
        fam_name='DELTA52', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length)
    ids = [delta52, ]
    return ids


def generate_radia_model(solve=SOLVE_FLAG, **kwargs):
    """."""

    delta = DeltaSabia()
    phase = kwargs['phase']
    dgv = kwargs['dgv']
    if 'apply_mag' in kwargs.keys():
        apply_mag = kwargs['apply_mag']
        p = str(int(np.modf(np.abs(phase))[-1]))
        gv = str(int(np.modf(np.abs(dgv))[-1]))
        mag_fname = './mags/p{}_dgv{}.pickle'.format(p, gv)

        mag_dict = load_pickle(mag_fname)
        delta.create_radia_object(magnetization_dict=mag_dict)
    delta.set_cassete_positions(dp=phase, dgv=dgv)

    # if solve:
        # delta.solve()

    return delta
