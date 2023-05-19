#!/usr/bin/env python-sirius

import matplotlib.pyplot as plt
import numpy as np

from fieldmaptrack import FieldMap, Beam, Trajectory, Multipoles


# ID_PERIOD = 180.0  # [mm]

# FOLDER_BASE = '/home/gabriel/repos-dev/'
FOLDER_BASE = './results/'

DATA_PATH = 'correctors/fieldmaps/'

ID_CONFIGS = {

    # wiggler with correctors - gap 045.00mm
    'CV': '2023-03-20_PAPU_Corrector_Model01_Sim_completo_X=-12_12mm_Z=-200_200mm_Ich=0A_Icv=10A.txt',
    'CH': '2023-03-20_PAPU_Corrector_Model01_Sim_completo_X=-12_12mm_Z=-200_200mm_Ich=10A_Icv=0A.txt',
    }


def calc_multipoles(traj, harmonics, idconfig):

    # calc multipoles
    n_list = harmonics
    s_list = harmonics
    mp = Multipoles()
    mp.trajectory = traj
    mp.perpendicular_grid = np.linspace(-3, 3, 21)
    mp.normal_field_fitting_monomials = n_list
    mp.skew_field_fitting_monomials = s_list

    mp.calc_multipoles(is_ref_trajectory_flag=False)
    mp.calc_multipoles_integrals()

    if idconfig == 'CH':
        mp.calc_multipoles_integrals_relative(
                mp.normal_multipoles_integral,
                main_monomial=0,
                r0=12,
                is_skew=False)
    else:
        mp.calc_multipoles_integrals_relative(
                mp.skew_multipoles_integral,
                main_monomial=0,
                r0=12,
                is_skew=True)

    normal_dip = mp.normal_multipoles[0, :]
    skew_dip = mp.skew_multipoles[0, :]
    normal_sext = mp.normal_multipoles[1, :]
    skew_sext = mp.skew_multipoles[1, :]
    normal_dode = mp.normal_multipoles[2, :]
    skew_dode = mp.skew_multipoles[2, :]
    i_ndip = mp.normal_multipoles_integral[0]
    i_sdip = mp.skew_multipoles_integral[0]
    i_nsext = mp.normal_multipoles_integral[1]
    i_ssext = mp.skew_multipoles_integral[1]
    i_ndode = mp.normal_multipoles_integral[2]
    i_sdode = mp.skew_multipoles_integral[2]

    plt.figure(1)
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    ndip = normal_dip
    plt.plot(
        traj.rz, ndip, color=colors[1],
        label='Integrated = {:.4f} Tm'.format(i_ndip))
    plt.xlabel('rz [mm]')
    plt.ylabel('Dipolar component [T]')
    plt.grid()
    plt.legend()
    plt.title(idconfig + ' By dipole')

    plt.figure(2)
    sdip = skew_dip
    plt.plot(
        traj.rz, sdip, color=colors[1],
        label='Integrated = {:.4f} T/m'.format(i_sdip))
    plt.xlabel('rz [mm]')
    plt.ylabel('Dipolar component [T]')
    plt.grid()
    plt.legend()
    plt.title(idconfig + ' Bx dipole')

    plt.figure(3)
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    nsext = normal_sext
    plt.plot(
        traj.rz, nsext, color=colors[3],
        label='Integrated = {:.4f} T/m'.format(i_nsext))
    plt.xlabel('rz [mm]')
    plt.ylabel('Sextupolar component [T/m²]')
    plt.grid()
    plt.legend()
    plt.title(idconfig + ' By sextupole')

    plt.figure(4)
    ssext = skew_sext
    plt.plot(
        traj.rz, ssext, color=colors[3],
        label='Integrated = {:.4f} T/m'.format(i_ssext))
    plt.xlabel('rz [mm]')
    plt.ylabel('Sextupolar component [T/m²]')
    plt.grid()
    plt.legend()
    plt.title(idconfig + ' Bx sextupole')

    plt.figure(5)
    ndode = normal_dode
    plt.plot(
        traj.rz, ndode, color=colors[2],
        label='Integrated = {:.4f} T/m³'.format(i_ndode))
    plt.xlabel('rz [mm]')
    plt.ylabel('Dodecapolar component [T/m⁴]')
    plt.grid()
    plt.legend()
    plt.title(idconfig + ' By dodecapole')

    plt.figure(6)
    sdode = skew_dode
    plt.plot(
        traj.rz, sdode, color=colors[2],
        label='Integrated = {:.4f} T/m³'.format(i_sdode))
    plt.xlabel('rz [mm]')
    plt.ylabel('Dodecapolar component [T/m⁴]')
    plt.grid()
    plt.legend()
    plt.title(idconfig + ' Bx dodecapole')
    plt.show()

    print(mp)
    return


def run(idconfig, plot=True):

    MEAS_FILE = ID_CONFIGS[idconfig]

    print(MEAS_FILE)

    # _, meas_id =  MEAS_FILE.split('ID=')
    # meas_id = meas_id.replace('.dat', '')

    fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE
    print(fmap_fname)
    fmap = FieldMap(fmap_fname)
    beam = Beam(energy=3.0)
    traj = Trajectory(
        beam=beam, fieldmap=fmap, not_raise_range_exceptions=True)
    traj.calc_trajectory(init_rz=-199, s_step=0.2, min_rz=199)

    # NOTE: unify pos and ang plots in a single figure

    deltarx = traj.rx[-1] - traj.rx[0]
    deltary = traj.ry[-1] - traj.ry[0]
    by = fmap.by[fmap.ry_zero][fmap.rx_zero][:]
    bx = fmap.bx[fmap.ry_zero][fmap.rx_zero][:]

    b = bx if idconfig == 'CV' else by
    integrated = np.trapz(b, fmap.rz/1000)
    if plot:
        plt.plot(
            fmap.rz, b, color='g',
            label='B integrated = {:.4f} Tm'.format(integrated))
        plt.xlabel('z [mm]')
        plt.ylabel('B [T]')
        plt.grid()
        plt.title('Field given by {}'.format(idconfig))
        plt.legend()
        plt.savefig(
            'results/correctors/' + idconfig + '/field-by-' + idconfig +
            '.png', dpi=300)
        plt.show()

        labelx = 'rx, delta: {:+.2f} mm'.format(deltarx)
        labely = 'ry, delta: {:+.2f} mm'.format(deltary)
        plt.plot(traj.rz, 1e3*traj.rx, label=labelx, color='C1')
        plt.plot(traj.rz, 1e3*traj.ry, label=labely, color='C0')
        plt.xlabel('rz [um]')
        plt.ylabel('pos [um]')
        plt.legend()
        plt.grid()
        plt.title('Runge-Kutta Trajectory Pos for fmap {}'.format(idconfig))
        plt.savefig(
            'results/correctors/' + idconfig + '/rk-trajectory-pos-' +
            idconfig + '.png', dpi=300)
        plt.show()

    deltapx = traj.px[-1] - traj.px[0]
    deltapy = traj.py[-1] - traj.py[0]
    if plot:
        labelx = 'px, delta: {:+.1f} urad'.format(1e6*deltapx)
        labely = 'py, delta: {:+.1f} urad'.format(1e6*deltapy)
        plt.plot(traj.rz, 1e6*traj.px, label=labelx, color='C1')
        plt.plot(traj.rz, 1e6*traj.py, label=labely, color='C0')
        plt.xlabel('rz [um]')
        plt.ylabel('ang [urad]')
        plt.legend()
        plt.grid()
        plt.title('Runge-Kutta Trajectory Ang for fmap {}'.format(idconfig))
        plt.savefig('results/correctors/' + idconfig + '/rk-trajectory-ang-' +
                    idconfig + '.png', dpi=300)
        plt.show()

    return traj, deltarx, deltary, deltapx, deltapy


if __name__ == "__main__":
    """."""
    idconfig = 'CH'
    traj, deltarx, deltary, deltapx, deltapy = run(idconfig, plot=True)
    harmonics = [0, 2, 4]
    calc_multipoles(traj, harmonics, idconfig)
