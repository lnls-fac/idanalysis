#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize

from idanalysis.kickmaps import IDKickMap

from utils import GAPS, PHASES
from utils import get_idconfig
from run_rk_traj import create_idkickmap
from run_rk_traj import load_rk_traj


def plot_multipoles(
        phase, traj_init_rx, traj_init_ry, rk_s_step=0.2):
    """."""
    rz_dict = dict()
    skew_quad_dict = dict()
    normal_quad_dict = dict()
    skew_sext_dict = dict()
    normal_sext_dict = dict()
    i_nquad_dict = dict()
    i_squad_dict = dict()
    i_nsext_dict = dict()
    i_ssext_dict = dict()

    for i, gap in enumerate(GAPS):
        # create IDKickMap and calc trajectory
        idkickmap = create_idkickmap(phase=phase, gap=gap)
        idkickmap.rk_s_step = rk_s_step
        idkickmap.fmap_calc_trajectory(
            traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry)
        traj = idkickmap.traj
        rz = traj.rz
        rz_dict[gap] = rz

        # multipolar analysis
        nlist = np.arange(0, 3, 1).tolist()
        slist = np.arange(0, 3, 1).tolist()
        idkickmap.fmap_config.multipoles_perpendicular_grid = np.linspace(
            -3, 3, 7)
        idkickmap.fmap_config.multipoles_normal_field_fitting_monomials = nlist
        idkickmap.fmap_config.multipoles_skew_field_fitting_monomials = slist
        idkickmap.fmap_config.multipoles_r0 = 12  # [mm]
        idkickmap.fmap_config.normalization_monomial = 0
        IDKickMap.multipoles_analysis(idkickmap.fmap_config)
        multipoles = idkickmap.fmap_config.multipoles
        normal_quad = multipoles.normal_multipoles[1, :]
        skew_quad = multipoles.skew_multipoles[1, :]
        normal_sext = multipoles.normal_multipoles[2, :]
        skew_sext = multipoles.skew_multipoles[2, :]
        i_nquad = multipoles.normal_multipoles_integral[1]
        i_squad = multipoles.skew_multipoles_integral[1]
        i_nsext = multipoles.normal_multipoles_integral[2]
        i_ssext = multipoles.skew_multipoles_integral[2]
        normal_quad_dict[gap] = normal_quad
        skew_quad_dict[gap] = skew_quad
        normal_sext_dict[gap] = normal_sext
        skew_sext_dict[gap] = skew_sext
        i_nquad_dict[gap] = i_nquad
        i_squad_dict[gap] = i_squad
        i_nsext_dict[gap] = i_nsext
        i_ssext_dict[gap] = i_ssext

    plt.figure(1)
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    for i, gap in enumerate(GAPS):
        nquad = normal_quad_dict[gap]
        plt.plot(rz, nquad, colors[i], label=GAPS[i] + ' mm')
    plt.xlabel('rz [mm]')
    plt.ylabel('Quadrupolar component [T/m]')
    plt.grid()
    plt.legend()
    plt.title(
        'Normal quadrupole for phase {} mm'.format(phase))
    fig_path = 'results/phase-organized/' + phase + '/Normal quadrupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(2)
    for i, gap in enumerate(GAPS):
        squad = skew_quad_dict[gap]
        plt.plot(rz, squad, colors[i], label=GAPS[i] + ' mm')
    plt.xlabel('rz [mm]')
    plt.ylabel('Quadrupolar component [T/m]')
    plt.grid()
    plt.legend()
    plt.title(
        'Skew quadrupole for phase {} mm'.format(phase))
    fig_path = 'results/phase-organized/' + phase + '/Skew quadrupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(3)
    for i, gap in enumerate(GAPS):
        nsext = normal_sext_dict[gap]
        plt.plot(rz, nsext, colors[i], label=GAPS[i] + ' mm')
    plt.xlabel('rz [mm]')
    plt.ylabel('Sextupolar component [T/m²]')
    plt.grid()
    plt.legend()
    plt.title(
        'Normal sextupole for phase {} mm'.format(phase))
    fig_path = 'results/phase-organized/' + phase + '/Normal sextupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(4)
    for i, gap in enumerate(GAPS):
        ssext = skew_sext_dict[gap]
        plt.plot(rz, ssext, colors[i], label=GAPS[i] + ' mm')
    plt.xlabel('rz [mm]')
    plt.ylabel('Sextupolar component [T/m²]')
    plt.grid()
    plt.legend()
    plt.title(
        'Skew sextupole for phase {} mm'.format(phase))
    fig_path = 'results/phase-organized/' + phase + '/Skew sextupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # generate table
    row1 = [
        'Gap [mm]',
        'Normal quadrupole [T]',
        'Skew quadrupole [T]',
        'Normal sextupole [T/m]',
        'Skew sextupole [T/m]']

    row_list = []
    row_list.append(row1)
    for gap in GAPS:
        i_nquadf = format(i_nquad_dict[gap], '+5.4f')
        i_squadf = format(i_squad_dict[gap], '+5.4f')
        i_nsextf = format(i_nsext_dict[gap], '+5.4f')
        i_ssextf = format(i_ssext_dict[gap], '+5.4f')
        row = [
            gap,
            i_nquadf,
            i_squadf,
            i_nsextf,
            i_ssextf]
        row_list.append(row)

    print(row_list[0][0] + " & " + row_list[0][1]
          + " & " + row_list[0][2] + " & " +
          row_list[0][3] + " & " + row_list[0][4] + " \\")

    for i, gap in enumerate(GAPS):
        print(
            row_list[i+1][0] + " & " + row_list[i+1][1]
            + " & " + row_list[i+1][2] + " & " +
            row_list[i+1][3] + " & " + row_list[i+1][4] + " \\")

    return


def calc_multipoles(phase, gap, data, harmonics):
    """."""
    data_ = data[phase]  # data for specific phase

    # load fieldmap file and rebuild traj attributes
    idkickmap = create_idkickmap(phase=phase, gap=gap)

    # calc multipoles
    n_list = harmonics
    s_list = harmonics
    idkickmap.fmap_config.multipoles_perpendicular_grid = np.linspace(-3, 3, 7)
    idkickmap.fmap_config.multipoles_normal_field_fitting_monomials = n_list
    idkickmap.fmap_config.multipoles_skew_field_fitting_monomials = s_list
    idkickmap.fmap_config.multipoles_r0 = 12  # [mm]
    idkickmap.fmap_config.normalization_monomial = 0

    mult_quad_norm, mult_quad_skew = dict(), dict()
    mult_sext_norm, mult_sext_skew = dict(), dict()
    mult_quad_norm_integ, mult_quad_skew_integ = dict(), dict()
    mult_sext_norm_integ, mult_sext_skew_integ = dict(), dict()

    print(gap)
    idkickmap.traj.s = data_['s'][gap]
    idkickmap.traj.rx = data_['rx'][gap]
    idkickmap.traj.ry = data_['ry'][gap]
    idkickmap.traj.rz = data_['rz'][gap]
    idkickmap.traj.px = data_['px'][gap]
    idkickmap.traj.py = data_['py'][gap]
    idkickmap.traj.pz = data_['pz'][gap]
    IDKickMap.multipoles_analysis(idkickmap.fmap_config)
    multipoles = idkickmap.fmap_config.multipoles
    mult_quad_norm[gap] = multipoles.normal_multipoles[1, :]
    mult_quad_skew[gap] = multipoles.skew_multipoles[1, :]
    mult_sext_norm[gap] = multipoles.normal_multipoles[2, :]
    mult_sext_skew[gap] = multipoles.skew_multipoles[2, :]
    mult_quad_norm_integ[gap] = multipoles.normal_multipoles_integral[1]
    mult_quad_skew_integ[gap] = multipoles.skew_multipoles_integral[1]
    mult_sext_norm_integ[gap] = multipoles.normal_multipoles_integral[2]
    mult_sext_skew_integ[gap] = multipoles.skew_multipoles_integral[2]

    data = dict()
    data['mult_quad_norm'] = mult_quad_norm
    data['mult_quad_skew'] = mult_quad_skew
    data['mult_sext_norm'] = mult_sext_norm
    data['mult_sext_skew'] = mult_sext_skew
    data['mult_quad_norm_integ'] = mult_quad_norm_integ
    data['mult_quad_skew_integ'] = mult_quad_skew_integ
    data['mult_sext_norm_integ'] = mult_sext_norm_integ
    data['mult_sext_skew_integ'] = mult_sext_skew_integ
    return data


if __name__ == "__main__":
    """."""
    rk_traj_res = load_rk_traj()
    traj_data, rk_s_step = rk_traj_res[0], rk_traj_res[1]
    traj_init_rx, traj_init_ry = rk_traj_res[2], rk_traj_res[3]
    traj_init_px, traj_init_py = rk_traj_res[4], rk_traj_res[5]

    phase, gap = PHASES[4], GAPS[0]
    harmonics = [0, 1, 2]
    data = calc_multipoles(phase, gap, traj_data, harmonics)
    plot_multipoles(phase, traj_init_rx, traj_init_ry, rk_s_step=0.2)
