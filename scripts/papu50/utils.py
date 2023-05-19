"""."""
import pyaccel
import pymodels
import numpy as np

from imaids.models import PAPU
from idanalysis import IDKickMap

BEAM_ENERGY = 3.0  # [GeV]
DEF_RK_S_STEP = 1.0  # [mm] seems converged for the measurement fieldmap grids
ROLL_OFF_RT = 10.0  # [mm]
SOLVE_FLAG = True

ID_PERIOD = 50  # [mm]
NR_PERIODS = 18  # [mm]
NR_PERIODS_REAL_ID = 18
SIMODEL_ID_LEN = 1.200  # [m]
ID_KMAP_LEN = SIMODEL_ID_LEN
RESCALE_KICKS = NR_PERIODS_REAL_ID/NR_PERIODS
RESCALE_LENGTH = 1  # RK traj is not calculated in free field regions
NOMINAL_GAP = 24.0  # [mm] - fixed papu50 gap.
ID_FAMNAME = 'PAPU50'

SIMODEL_FITTED = False
SHIFT_FLAG = False
FILTER_FLAG = False

FOLDER_DATA = './results/model/data/'
KYMA22_KMAP_FILENAME = (
    '/opt/insertion-devices/kyma22/results/'
    'model/kickmaps/kickmap-ID-kyma22-phase_pos00p000.txt')

INSERT_KYMA = False
KYMA_RESCALE_KICKS = 1  # Radia simulations have fewer ID periods
KYMA_RESCALE_LENGTH = 1  # Radia simulations have fewer ID periods

gaps = [NOMINAL_GAP]
phases = [0, 25]
widths = [40]
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
    ids = list()
    IDModel = pymodels.si.IDModel

    nr_steps = nr_steps or 40

    if INSERT_KYMA:
        fname = KYMA22_KMAP_FILENAME
        kyma22 = IDModel(
            subsec=IDModel.SUBSECTIONS.ID09SA,
            file_name=fname,
            fam_name='APU22', nr_steps=nr_steps,
            rescale_kicks=KYMA_RESCALE_KICKS,
            rescale_length=KYMA_RESCALE_LENGTH)
        ids.append(kyma22)

    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 1

    papu50 = IDModel(
        subsec=IDModel.SUBSECTIONS.ID17SA,
        file_name=fname,
        fam_name='PAPU50', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length)
    ids.append(papu50)

    return ids


def generate_radia_model(width=widths[0], phase=0,
                         gap=NOMINAL_GAP, solve=SOLVE_FLAG, **kwargs):
    """."""
    papu = PAPU(gap=gap)
    papu.dg = phase
    if solve:
        papu.solve()
    return papu
