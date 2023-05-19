#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from idanalysis import IDKickMap
from mathphys.functions import save_pickle, load_pickle

import utils

SOLVE_FLAG = utils.SOLVE_FLAG
RK_S_STEP = utils.DEF_RK_S_STEP
BEAM_ENERGY = utils.BEAM_ENERGY
ROLL_OFF_RX = utils.ROLL_OFF_RX


def create_path(phase):
    fpath = utils.FOLDER_DATA
    phase_str = utils.get_phase_str(phase)
    fpath = fpath.replace('model/data/', 'measurements/data/phase_{}/'.format(phase_str))
    return fpath


def config_traj(phase, max_rz):
    MEAS_FILE = utils.MEAS_FILE
    idkickmap = IDKickMap()
    fmap_fname = MEAS_FILE
    fmap_fname = fmap_fname.replace('phase0', 'phase{}'.format(phase))
    idkickmap.fmap_fname = fmap_fname
    idkickmap.beam_energy = BEAM_ENERGY
    idkickmap.rk_s_step = RK_S_STEP
    idkickmap.traj_init_rz = -max_rz
    idkickmap.traj_rk_min_rz = max_rz
    idkickmap.kmap_idlen = 1.3
    return idkickmap


def run_generate_kickmap(phase, posx, posy):

    idkickmap = config_traj(phase, 739)
    idkickmap.fmap_calc_kickmap(posx=posx, posy=posy)
    phase_str = utils.get_phase_str(phase)
    fname = './results/measurements/kickmaps/kickmap-ID-kyma22-phase{}.txt'.format(phase_str)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


def get_field_on_trajectory(phase, data, max_rz=739, rx_init=0):
    """Calculate RK for set of EPU configurations."""
    s = dict()
    bx, by, bz = dict(), dict(), dict()
    rz, rx, ry = dict(), dict(), dict()
    px, py, pz = dict(), dict(), dict()

    # create IDKickMap and calc trajectory
    idkickmap = config_traj(phase=phase, max_rz=max_rz)
    for rx0 in rx_init:
        idkickmap.fmap_calc_trajectory(
            traj_init_rx=rx0, traj_init_ry=0,
            traj_init_px=0, traj_init_py=0)
        traj = idkickmap.traj

        s = traj.s
        bx, by, bz = traj.bx, traj.by, traj.bz
        rx, ry, rz = traj.rx, traj.ry, traj.rz
        px, py, pz = traj.px, traj.py, traj.pz

        data[('ontraj_s', rx0)] = s

        data[('ontraj_rx', rx0)] = rx
        data[('ontraj_ry', rx0)] = ry
        data[('ontraj_rz', rx0)] = rz

        data[('ontraj_bx', rx0)] = bx
        data[('ontraj_by', rx0)] = by
        data[('ontraj_bz', rx0)] = bz

        data[('ontraj_px', rx0)] = px
        data[('ontraj_py', rx0)] = py
        data[('ontraj_pz', rx0)] = pz

    return data


def save_data(phase, data):
    """."""
    fpath = create_path(phase)
    fpath = fpath.replace('model', 'measurements')
    fname = fpath + 'field_data_kyma22'
    save_pickle(data, fname, overwrite=True)


def plot_rk_traj(data, rx_init=0):

    phase = data['phase']
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    fpath = create_path(phase)
    for i, rx0 in enumerate(rx_init):
        rz = data[('ontraj_rz', rx0)]
        rx = data[('ontraj_rx', rx0)]
        ry = data[('ontraj_ry', rx0)]
        px = 1e6*data[('ontraj_px', rx0)]
        py = 1e6*data[('ontraj_py', rx0)]

        rx -= 1e3*rx0
        plt.figure(1)
        plt.plot(
            rz, 1e3*rx, color=colors[i],
            label='rx0 = {} mm, delta rx: {:.3f} um'.format(
                1e3*rx0, 1e3*rx[-1]))

        plt.figure(2)
        plt.plot(
            rz, 1e3*ry, color=colors[i],
            label='rx0 = {} mm, delta ry: {:.3f} um'.format(
                1e3*rx0, 1e3*ry[-1]))

        plt.figure(3)
        plt.plot(
            rz, px, color=colors[i],
            label='rx0 = {} mm, delta px: {:.3f} urad'.format(
                1e3*rx0, px[-1]))

        plt.figure(4)
        plt.plot(
            rz, py, color=colors[i],
            label='rx0 = {} mm, delta py: {:.3f} urad'.format(
                1e3*rx0, py[-1]))

    xlabel = 'rz [mm]'
    ylabel = ['rx [um]', 'ry [um]', 'px [urad]', 'py [urad]']
    tlt = 'kyma50 On-axis Runge-Kutta Traj. at phase {:+.3f} mm'.format(phase)
    fig_sulfix = ['traj-rx', 'traj-ry', 'traj-px', 'traj-py']
    for j, i in enumerate([1, 2, 3, 4]):
        plt.figure(i)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel[j])
        plt.title(tlt)
        plt.grid()
        plt.legend()
        plt.savefig(fpath + fig_sulfix[j], dpi=300)
    plt.show()


def run_plot_data(phase, rx_init):

    fpath = create_path(phase)
    fpath = fpath.replace('model', 'measurements')
    fname = fpath + 'field_data_kyma22.pickle'
    data = load_pickle(fname)
    plot_rk_traj(data=data, rx_init=rx_init)


if __name__ == "__main__":
    """."""
    phase = 0
    data = dict(phase=phase)
    rx_init = [-10e-3, 0, 10e-3]  # High beta's worst initial conditions [m]
    data = get_field_on_trajectory(phase=phase, data=data, rx_init=rx_init)
    save_data(phase, data)
    run_plot_data(phase, rx_init=rx_init)
