#!/usr/bin/env python-sirius

import matplotlib.pyplot as plt

from fieldmaptrack import FieldMap, Beam, Trajectory

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import WLS_CONFIGS   


def run(idconfig, plot=True):
    
    MEAS_FILE = WLS_CONFIGS[idconfig]
    
    fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE
    print(fmap_fname)
    fmap = FieldMap(fmap_fname)
    beam = Beam(energy=3.0)
    traj = Trajectory(
        beam=beam, fieldmap=fmap, not_raise_range_exceptions=True)
    traj.calc_trajectory(init_rz=-600, s_step=0.2, min_rz=600, init_rx=0.0)
    
    # NOTE: unify pos and ang plots in a single figure

    deltarx = traj.rx[-1] - traj.rx[0]
    deltary = traj.ry[-1] - traj.ry[0]
    by = fmap.by[fmap.ry_zero][fmap.rx_zero][:]

    print('2nd Integral: {:+.3e} Tm2'.format(deltarx/1000*beam.brho))
    if plot:
        plt.plot(fmap.rz, by, color='g')
        plt.xlabel('z [mm]')
        plt.ylabel('B [T]')
        plt.grid()
        plt.title('Vertical field given by {}'.format(idconfig))
        plt.savefig('results/' + idconfig + '/field-by-' + idconfig + '.png',dpi=300)
        # plt.show()
        plt.clf()

        labelx = 'rx, delta: {:+.1f} um'.format(deltarx*1e3)
        labely = 'ry, delta: {:+.1f} um'.format(deltary*1e3)
        plt.plot(traj.rz, 1e3*traj.rx, label=labelx, color='C1')
        plt.plot(traj.rz, 1e3*traj.ry, label=labely, color='C0')
        plt.xlabel('rz [um]')
        plt.ylabel('pos [um]')
        plt.legend(loc='upper left')
        plt.grid()
        plt.title('Runge-Kutta Trajectory Pos for fmap {}'.format(idconfig))
        plt.savefig('results/' + idconfig + '/rk-trajectory-pos-' + idconfig + '.png',dpi=300)
        # plt.show()
        plt.clf()

    deltapx = traj.px[-1] - traj.px[0]
    deltapy = traj.py[-1] - traj.py[0]
    print('1st Integral: {:+.3e} Tm'.format(deltapx*beam.brho))
    if plot:
        labelx = 'px, delta: {:+.3f} urad '.format(1e6*deltapx)
        labely = 'py, delta: {:+.3f} urad'.format(1e6*deltapy)
        plt.plot(traj.rz, 1e6*traj.px, label=labelx, color='C1')
        plt.plot(traj.rz, 1e6*traj.py, label=labely, color='C0')
        plt.xlabel('rz [um]')
        plt.ylabel('ang [urad]')
        plt.legend(loc='upper left')
        plt.grid()
        plt.title('Runge-Kutta Trajectory Ang for fmap {}'.format(idconfig))
        plt.savefig('results/' + idconfig + '/rk-trajectory-ang-' + idconfig + '.png',dpi=300)
        # plt.show()
        plt.clf()

    return deltarx, deltary, deltapx, deltapy


if __name__ == "__main__":
    """."""
    config_list = ['I10A','I50A','I100A','I200A','I228A','I250A','I300A']
    for config in config_list:
        run(config, plot=True)
    # run('I228A', plot=True)
