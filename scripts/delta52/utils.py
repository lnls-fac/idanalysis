"""."""
import pyaccel
import pymodels
import numpy as np

from imaids.models import DeltaSabia
from idanalysis import IDKickMap

BEAM_ENERGY = 3.0  # [GeV]
DEF_RK_S_STEP = 1  # [mm] seems converged for the measurement fieldmap grids
ROLL_OFF_RT = 5.0  # [mm]
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

SIMODEL_FITTED = True
FIT_PATH = '/home/gabriel/Desktop/my-data-by-day/2023-05-15-SI_low_coupling/fitting_ref_config_before_low_coupling_strengths.pickle'
SHIFT_FLAG = True
FILTER_FLAG = False

FOLDER_DATA = './results/model/data/'
MEAS_DATA_PATH = './meas-data/id-sabia/model-03/measurement/magnetic/hallprobe/'
MEAS_FLAG = True
REAL_WIDTH = 45

gaps = [13.125, 26.250]
phases = [0, -13.125, -26.25]
# gaps = [0]
# phases = [0]
widths = [REAL_WIDTH]
field_component = 'by'
var_param = 'gap'

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

    # shimming A
    'ID4458': 'delta_sabia_shimmingA/2023-04-18_' +
    'DeltaSabia_Phase01_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4458.dat',

    'ID4459': 'delta_sabia_shimmingA/2023-04-18_' +
    'DeltaSabia_Phase02_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4459.dat',

    'ID4460': 'delta_sabia_shimmingA/2023-04-18_' +
    'DeltaSabia_Phase03_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4460.dat',

    'ID4461': 'delta_sabia_shimmingA/2023-04-19_' +
    'DeltaSabia_Phase04_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4461.dat',

    'ID4462': 'delta_sabia_shimmingA/2023-04-19_' +
    'DeltaSabia_Phase05_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4462.dat',

    'ID4463': 'delta_sabia_shimmingA/2023-04-19_' +
    'DeltaSabia_Phase06_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4463.dat',

    'ID4464': 'delta_sabia_shimmingA/2023-04-19_' +
    'DeltaSabia_Phase07_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4464.dat',

    # shimming B
    'ID4465': 'delta_sabia_shimmingB/2023-05-02_' +
    'DeltaSabia_Phase01_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4465.dat',

    'ID4466': 'delta_sabia_shimmingB/2023-05-02_' +
    'DeltaSabia_Phase02_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4466.dat',

    'ID4467': 'delta_sabia_shimmingB/2023-05-02_' +
    'DeltaSabia_Phase03_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4467.dat',

    'ID4468': 'delta_sabia_shimmingB/2023-05-02_' +
    'DeltaSabia_Phase04_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4468.dat',

    'ID4469': 'delta_sabia_shimmingB/2023-05-02_' +
    'DeltaSabia_Phase05_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4469.dat',

    'ID4470': 'delta_sabia_shimmingB/2023-05-02_' +
    'DeltaSabia_Phase06_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4470.dat',

    'ID4471': 'delta_sabia_shimmingB/2023-05-02_' +
    'DeltaSabia_Phase07_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4471.dat',
    }


MEAS_phase00p_nsh = ['ID4381', 'ID4378', 'ID4384']
MEAS_phase13n_nsh = ['ID4383', 'ID4380']
MEAS_phase26n_nsh = ['ID4382', 'ID4379']

MEAS_phase00p_Ash = ['ID4462', 'ID4459', 'ID4458']
MEAS_phase13n_Ash = ['ID4464', 'ID4461']
MEAS_phase26n_Ash = ['ID4463', 'ID4460']

MEAS_phase00p_Bsh = ['ID4469', 'ID4466', 'ID4465']
MEAS_phase13n_Bsh = ['ID4471', 'ID4468']
MEAS_phase26n_Bsh = ['ID4470', 'ID4467']


MEAS_GAPS = [13.125, 26.25, 0]
MEAS_PHASES = [00.00, -13.125, -26.25]

ORDERED_CONFIGS = [[MEAS_phase00p_nsh, MEAS_phase13n_nsh, MEAS_phase26n_nsh],
                   [MEAS_phase00p_Ash, MEAS_phase13n_Ash, MEAS_phase26n_Ash],
                   [MEAS_phase00p_Bsh, MEAS_phase13n_Bsh, MEAS_phase26n_Bsh]]


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


def generate_radia_model(phase, gap,
                         solve=SOLVE_FLAG, **kwargs):
    """."""

    delta = DeltaSabia()

    delta.set_cassete_positions(dp=phase, dgv=gap)

    if solve:
        delta.solve()

    return delta
