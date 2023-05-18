#!/usr/bin/env python-sirius

import numpy as np

from mathphys.functions import save_pickle

from idanalysis.analysis import IDKickMap
import utils


SOLVE_FLAG = utils.SOLVE_FLAG
RK_S_STEP = utils.DEF_RK_S_STEP
GAP = 9.7  # [mm]

# these are the nominal termination parameters for
# HybridPlanar as of 2023-01
# paremeters: [b1t, b2t, b3t, dist1, dist2 ]
NOMINAL_TERMINATION_PARAMETERS = np.array([6.35/2, 2.9/2, 6.35, 2.9, 2.9])


def calc_traj(ivu):
    """."""
    init_rx = init_px = 0
    init_ry = init_py = 0

    idkickmap = IDKickMap()
    idkickmap.radia_model = ivu
    idkickmap.beam_energy = utils.BEAM_ENERGY
    idkickmap._radia_model_config.traj_init_px = init_px
    idkickmap._radia_model_config.traj_init_py = init_py
    idkickmap.traj_init_rz = -200
    idkickmap.rk_s_step = RK_S_STEP
    idkickmap.fmap_calc_trajectory(
        traj_init_rx=init_rx, traj_init_ry=init_ry,
        traj_init_px=init_px, traj_init_py=init_py)
    traj = idkickmap.traj

    traj = idkickmap.traj
    rxf, pxf = traj.rx[-1], traj.px[-1]
    ryf, pyf = traj.ry[-1], traj.py[-1]

    return traj, rxf, ryf, pxf, pyf


def calc_residue(ivu):
    """."""
    _, rxf, ryf, pxf, pyf = calc_traj(ivu)
    residue = np.array([rxf, pxf, ryf, pyf])
    return residue


def calc_rk_respm(width):
    """."""
    delta_p = 1e-6  # [mm]
    termination_parameters = NOMINAL_TERMINATION_PARAMETERS
    parms_labels = (
        'block1 thickness', 'block2 thickness', 'block3 thickness',
        'distance1 value ', 'distance2 value ',
    )
    respm = np.zeros((4, 5))

    for idx in range(len(termination_parameters)):
        print(parms_labels[idx] + ', ', end='', flush=True)

        # positive variation
        print('+delta ... ', end='', flush=True)
        termination_parameters[idx] += delta_p/2
        ivu = utils.generate_radia_model(
            gap=GAP, width=width,
            termination_parameters=termination_parameters,
            solve=SOLVE_FLAG
            )
        residue_p = calc_residue(ivu)

        # negative variation
        print('-delta ... ', end='', flush=True)
        termination_parameters[idx] -= delta_p
        ivu = utils.generate_radia_model(
            gap=GAP, width=width,
            termination_parameters=termination_parameters,
            solve=SOLVE_FLAG)
        residue_n = calc_residue(ivu)

        # restore parameter
        print('ok')
        termination_parameters[idx] += delta_p/2

        respm[:, idx] = (residue_p - residue_n) / delta_p

    return respm


def calc_delta_pos(width, delta_parameters=None):
    """."""
    termination_parameters = NOMINAL_TERMINATION_PARAMETERS
    if delta_parameters is not None:
        termination_parameters += delta_parameters

    traj_init_rx = 0
    traj_init_ry = 0
    traj_init_px = 0
    traj_init_py = 0
    rk_s_step = 2.0

    ivu = utils.generate_radia_model(
            gap=GAP, width=width,
            termination_parameters=termination_parameters,
            solve_flag=SOLVE_FLAG
            )

    idkickmap = IDKickMap()
    idkickmap.radia_model = ivu
    idkickmap.beam_energy = 3.0  # [GeV]
    idkickmap._radia_model_config.traj_init_px = traj_init_px
    idkickmap._radia_model_config.traj_init_py = traj_init_py
    idkickmap.traj_init_rz = -200
    idkickmap.rk_s_step = rk_s_step
    idkickmap.fmap_calc_trajectory(
        traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry,
        traj_init_px=traj_init_px, traj_init_py=traj_init_py)
    traj = idkickmap.traj
    dxf = -1*traj.rx[-1]
    dyf = -1*traj.ry[-1]
    pxf = -1*traj.px[-1]
    pyf = -1*traj.py[-1]
    delta_pos = np.array([dxf, dyf, pxf, pyf])

    return delta_pos


def calc_correction(respm, residue):
    """."""
    tol_svals = 1e-4
    umat, smat, vmat = np.linalg.svd(respm, full_matrices=False)
    sel_svals = abs(smat) > tol_svals
    ismat = np.zeros(smat.shape)
    ismat[sel_svals] = 1/smat[sel_svals]
    ismat = np.diag(ismat)
    invmat = np.dot(np.dot(vmat.T, ismat), umat.T)
    dp = - np.dot(invmat, residue)
    return dp


def calc_termination_parameters(width, respm, nr_iters=7):
    """."""
    def new_model():
        print('parameters [mm] | residue [um|urad] : ', end='')
        parms_str = ''.join([f'{val:.8f} ' for val in termination_parameters])
        ivu = utils.generate_radia_model(
            gap=GAP, width=width,
            termination_parameters=termination_parameters,
            solve=SOLVE_FLAG)
        residue = calc_residue(ivu)
        resid_str = ''.join([f'{val*1e6:+.8f} ' for val in residue])
        print(parms_str + ' | ' + resid_str)
        return residue

    # iterate
    termination_parameters = NOMINAL_TERMINATION_PARAMETERS.copy()
    for i in np.arange(nr_iters):
        residue = new_model()
        dparams = calc_correction(respm, residue)
        termination_parameters += dparams
    residue = new_model()
    return termination_parameters


def run_optimize_termination(widths=None):
    """."""
    widths = widths or [50]  # [mm]
    for width in widths:
        print('optimizing termination for width {} mm ...'.format(width))
        fname = utils.FOLDER_DATA + 'respm_termination_{}.pickle'.format(width)
        print('calculating response matrix...')
        respm = calc_rk_respm(width)
        print('finding termination parameters...')
        termination_parameters = calc_termination_parameters(width, respm)
        data = dict()
        data['respm'] = respm
        data['results'] = termination_parameters
        save_pickle(data, fname, overwrite=True)
        print()


if __name__ == "__main__":
    run_optimize_termination()
