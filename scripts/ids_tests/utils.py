"""."""
# imports
from imaids.models import Kyma22
from matplotlib import pyplot as plt

import pymodels
from idanalysis import IDKickMap

BEAM_ENERGY = 3.0  # [GeV]
DEF_RK_S_STEP = 1  # [mm] seems converged for the measurement fieldmap grids
ROLL_OFF_RT = 10.0  # [mm]
SOLVE_FLAG = False

ID_PERIOD = 22  # [mm]
NR_PERIODS = 50  #
NR_PERIODS_REAL_ID = 50  #
SIMODEL_ID_LEN = 1.300
ID_KMAP_LEN = SIMODEL_ID_LEN
RESCALE_KICKS = NR_PERIODS_REAL_ID/NR_PERIODS
RESCALE_LENGTH = 1

SIMODEL_FITTED = False
SHIFT_FLAG = True

FOLDER_DATA = './results/model/data/'
MEAS_FILE = './results/measurements/fieldmap_phase0.dat'
MEAS_FLAG = False

gaps = [40]
widths = [50]
phases = [0, 11]
field_component = 'by'
var_param = 'phase'


def get_termination_parameters():
    """."""
    return None


def generate_radia_model(
        width, gap, phase,
        termination_parameters=get_termination_parameters(),
        solve=False, nr_periods=5):
    """."""
    kyma = Kyma22(nr_periods=nr_periods)
    kyma.dg = phase
    if solve:
        kyma.solve()

    return kyma
