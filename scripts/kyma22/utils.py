"""."""
import pyaccel
import pymodels
import numpy as np

from imaids.models import Kyma22
from idanalysis import IDKickMap

BEAM_ENERGY = 3.0  # [GeV]
DEF_RK_S_STEP = 1  # [mm] seems converged for the measurement fieldmap grids
ROLL_OFF_RT = 10.0  # [mm]
SOLVE_FLAG = True

ID_PERIOD = 22  # [mm]
NR_PERIODS = 10  #
NR_PERIODS_REAL_ID = 50  #
SIMODEL_ID_LEN = 1.300
ID_KMAP_LEN = SIMODEL_ID_LEN
RESCALE_KICKS = NR_PERIODS_REAL_ID/NR_PERIODS
RESCALE_LENGTH = 1
NOMINAL_GAP = 8.0  # [mm] - fixed papu50 gap.
ID_FAMNAME = 'APU22'

SIMODEL_FITTED = True
SHIFT_FLAG = True
FILTER_FLAG = False

FOLDER_DATA = './results/model/data/'
MEAS_FILE = './results/measurements/fieldmap_phase0.dat'
MEAS_FLAG = False

gaps = [NOMINAL_GAP]
phases = [0, 11]
widths = [36]
field_component = 'by'
var_param = 'phase'


class CALC_TYPES:
    """."""
    nominal = 0
    nonsymmetrized = 1
    symmetrized = 2


def create_ids(
        fname, nr_steps=None, rescale_kicks=RESCALE_KICKS,
        rescale_length=RESCALE_LENGTH):
    # create IDs
    nr_steps = nr_steps or 40
    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 1

    # if MEAS_FLAG:
        # rescale_length = 1
    IDModel = pymodels.si.IDModel
    kyma22 = IDModel(
        subsec=IDModel.SUBSECTIONS.ID09SA,
        file_name=fname,
        fam_name='APU22', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length)
    ids = [kyma22, ]
    return ids


def generate_radia_model(width=widths[0], phase=0, gap=NOMINAL_GAP,
                         solve=SOLVE_FLAG, **kwargs):
    """."""
    kyma = Kyma22(nr_periods=NR_PERIODS)
    kyma.dg = phase
    if solve:
        kyma.solve()
    return kyma
