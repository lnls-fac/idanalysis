"""."""
import pyaccel
import pymodels
import numpy as np

from imaids.models import HybridPlanar as Hybrid
from idanalysis import IDKickMap

from mathphys.functions import load_pickle

BEAM_ENERGY = 3.0  # [GeV]
DEF_RK_S_STEP = 0.5  # [mm] seems converged for the measurement fieldmap grids
ROLL_OFF_RT = 6.0  # [mm]
SOLVE_FLAG = True

ID_PERIOD = 18.5  # [mm]
NR_PERIODS = 5
NR_PERIODS_REAL_ID = 108
SIMODEL_ID_LEN = 2.0  # [m]  It's necessary to update kickmaps and pymodels
ID_KMAP_LEN = SIMODEL_ID_LEN
RESCALE_KICKS = NR_PERIODS_REAL_ID/NR_PERIODS
RESCALE_LENGTH = 1  # Radia simulations have fewer ID periods
NOMINAL_GAP = 4.3  # [mm]
ID_FAMNAME = 'IVU18'

SIMODEL_FITTED = False
SHIFT_FLAG = True
FILTER_FLAG = False

FOLDER_DATA = './results/model/data/'

gaps = [NOMINAL_GAP]
phases = [0]
widths = [44]
field_component = 'by'
var_param = 'width'


class CALC_TYPES:
    """."""
    nominal = 0
    nonsymmetrized = 1
    symmetrized = 2


def get_termination_parameters(width=50):
    """."""
    fname = FOLDER_DATA + '/general/respm_termination_{}.pickle'.format(width)
    term = load_pickle(fname)
    b1t, b2t, b3t, dist1, dist2 = term['results']
    return list([b1t, b2t, b3t, dist1, dist2])


def get_folder_data():
    data_path = FOLDER_DATA
    return data_path


def get_gap_str(gap):
    """."""
    gap_str = '{:04.1f}'.format(gap).replace('.', 'p')
    return gap_str


def get_kmap_filename(gap, width, shift_flag=SHIFT_FLAG):
    fpath = FOLDER_DATA + 'kickmaps/'
    fpath = fpath.replace('model/data/', 'model/')
    gap_str = get_gap_str(gap)
    fname = fpath + f'kickmap-ID-{width}-gap{gap_str}mm.txt'
    if shift_flag:
        fname = fname.replace('.txt', '-shifted_on_axis.txt')
    return fname


def create_ids(
        fname, nr_steps=None, rescale_kicks=RESCALE_KICKS,
        rescale_length=RESCALE_LENGTH, **kwargs):
    # create IDs
    nr_steps = nr_steps or 40
    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 1
    IDModel = pymodels.si.IDModel
    ivu18 = IDModel(
        subsec=IDModel.SUBSECTIONS.ID08SB,
        file_name=fname,
        fam_name='IVU18', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length)
    ids = [ivu18, ]
    return ids


def generate_radia_model(width, gap=NOMINAL_GAP, phase=0,
                         termination_parameters=get_termination_parameters(),
                         solve=SOLVE_FLAG):
    """."""
    period_length = 18.5
    br = 1.24

    height = 26
    chamfer_b = 5

    p_width = 34
    p_height = 21
    pole_length = 2.8
    chamfer_p = 3
    y_pos = 0

    block_shape = [
        [-width/2, -chamfer_b],
        [-width/2, -height+chamfer_b],
        [-width/2+chamfer_b, -height],
        [width/2-chamfer_b, -height],
        [width/2, -height+chamfer_b],
        [width/2, -chamfer_b],
        [width/2-chamfer_b, 0],
        [-width/2+chamfer_b, 0],

    ]

    pole_shape = [
        [-p_width/2, -chamfer_p-y_pos],
        [-p_width/2, -p_height-y_pos],
        [p_width/2, -p_height-y_pos],
        [p_width/2, -chamfer_p-y_pos],
        [p_width/2-chamfer_p, 0-y_pos],
        [-p_width/2+chamfer_p, 0-y_pos],

    ]

    block_subdivision = [8, 4, 4]
    pole_subdivision = [34, 5, 5]

    b1t, b2t, b3t, dist1, dist2 = termination_parameters

    lengths = [b1t, b2t, b3t]
    distances = [dist1, dist2, 0]
    start_blocks_length = lengths
    start_blocks_distance = distances
    end_blocks_length = lengths[0:-1][::-1]
    end_blocks_distance = distances[0:-1][::-1]

    ivu = Hybrid(gap=gap, period_length=period_length,
                 mr=br, nr_periods=NR_PERIODS,
                 longitudinal_distance=0, block_shape=block_shape,
                 pole_shape=pole_shape, block_subdivision=block_subdivision,
                 pole_subdivision=pole_subdivision, pole_length=pole_length,
                 start_blocks_length=start_blocks_length,
                 start_blocks_distance=start_blocks_distance,
                 end_blocks_length=end_blocks_length,
                 end_blocks_distance=end_blocks_distance,
                 trf_on_blocks=True)
    if solve:
        ivu.solve()

    return ivu
