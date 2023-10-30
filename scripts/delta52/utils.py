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

    # no shimming
    'ID4384': 'delta_sabia_no_shimming/2023-03-29_' +
    'DeltaSabia_Phase01_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4384.dat',

    'ID4378': 'delta_sabia_no_shimming/2023-03-28_' +
    'DeltaSabia_Phase02_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4378.dat',

    'ID4379': 'delta_sabia_no_shimming/2023-03-28_' +
    'DeltaSabia_Phase03_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4379.dat',

    'ID4380': 'delta_sabia_no_shimming/2023-03-28_' +
    'DeltaSabia_Phase04_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4380.dat',

    'ID4381': 'delta_sabia_no_shimming/2023-03-28_' +
    'DeltaSabia_Phase05_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4381.dat',

    'ID4382': 'delta_sabia_no_shimming/2023-03-28_' +
    'DeltaSabia_Phase06_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4382.dat',

    'ID4383': 'delta_sabia_no_shimming/2023-03-28_' +
    'DeltaSabia_Phase07_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4383.dat',

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

    # shimming E + magic fingers
    'ID4836': 'delta_sabia_shimmingE_magic_fingers/2023-09-14_' +
    'DeltaSabia_Phase01_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4836.dat',

    'ID4837': 'delta_sabia_shimmingE_magic_fingers/2023-09-14_' +
    'DeltaSabia_Phase02_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4837.dat',

    'ID4838': 'delta_sabia_shimmingE_magic_fingers/2023-09-14_' +
    'DeltaSabia_Phase03_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4838.dat',

    'ID4839': 'delta_sabia_shimmingE_magic_fingers/2023-09-15_' +
    'DeltaSabia_Phase04_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4839.dat',

    'ID4840': 'delta_sabia_shimmingE_magic_fingers/2023-09-15_' +
    'DeltaSabia_Phase05_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4840.dat',

    'ID4841': 'delta_sabia_shimmingE_magic_fingers/2023-09-15_' +
    'DeltaSabia_Phase06_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4841.dat',

    'ID4842': 'delta_sabia_shimmingE_magic_fingers/2023-09-15_' +
    'DeltaSabia_Phase07_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4842.dat',
    }


phases = [00.00, -13.125, -26.25]
dgv = [13.125, 26.25]

dp_dgv_dict = {(0, 0): 'ID4818', (0, 26.25): 'ID4819',
               (-26.25, 26.25): 'ID4820', (-13.125, 26.25): 'ID4821',
               (0, 13.125): 'ID4822', (-26.25, 13.125): 'ID4823',
               (-13.125, 13.125): 'ID4824'}

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
    apply_mag = kwargs['apply_mag']

    if apply_mag:
        p = str(int(np.modf(np.abs(phase))[-1]))
        gv = str(int(np.modf(np.abs(dgv))[-1]))
        mag_fname = './mags/p{}_dgv{}.pickle'.format(p, gv)

        mag_dict = load_pickle(mag_fname)
        delta.create_radia_object(magnetization_dict=mag_dict)
    delta.set_cassete_positions(dp=phase, dgv=dgv)

    # if solve:
        # delta.solve()

    return delta
