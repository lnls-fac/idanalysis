#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize

from utils import GAPS, PHASES
from run_rk_traj import load_rk_traj


def plot_rk_traj_normalized_fields(
        fig_path, colors, dpi, show_flag, rz, bx, by, bz):
    """."""
    title_fld_sufix = ' for phase {} mm'.format(phase)
    title_fld_sufix = ''

    # # plot bx
    # plt.figure(1)
    # for i in range(len(GAPS)):
    #     gap = GAPS[i]
    #     rz_, bx_ = rz[gap], bx[gap]
    #     label = GAPS[i] + ' mm'
    #     plt.plot(rz_, bx_/np.max(np.abs(bx_)), colors[i], label=label)
    # plt.xlabel('rz [mm]')
    # plt.ylabel('Field [a.u.]')
    # plt.grid()
    # plt.legend()
    # plt.title('Normalized horizontal field' + title_fld_sufix)
    # plt.savefig(fig_path + 'field-bx.png', dpi=dpi)
    # if show_flag:
    #     plt.show()
    # plt.close()

    # plot by
    plt.figure(2)
    for i in range(len(GAPS)):
        gap = GAPS[i]
        rz_, by_ = rz[gap], by[gap]
        label = GAPS[i] + ' mm'
        # plt.plot(rz_, by_/np.max(np.abs(by_)), colors[i], label=label)
        plt.plot(rz_[:int(len(rz_)/4)], by_[:int(len(rz_)/4)], colors[i], label=label)
    plt.xlabel('z [mm]')
    plt.ylabel('B [T]')
    plt.grid()
    plt.legend()
    plt.title('Campo Vertical do EPU50 (Polarização Horizontal)' + title_fld_sufix)
    plt.savefig(fig_path + 'field-by.png', dpi=dpi)
    if show_flag:
        plt.show()
    plt.close()

    # # plot bz
    # plt.figure(3)
    # for i in range(len(GAPS)):
    #     gap = GAPS[i]
    #     rz_, bz_ = rz[gap], bz[gap]
    #     label = GAPS[i] + ' mm'
    #     plt.plot(rz_, bz_/np.max(np.abs(bz_)), colors[i], label=label)
    # plt.xlabel('rz [mm]')
    # plt.ylabel('Field [a.u.]')
    # plt.grid()
    # plt.legend()
    # plt.title('Normalized longitudinal field' + title_fld_sufix)
    # plt.savefig(fig_path + 'field-bz.png', dpi=dpi)
    # if show_flag:
    #     plt.show()
    # plt.close()


def plot_rk_traj_pos(fig_path, colors, dpi, show_flag, rz, rx, ry):
    """."""
    # plot rx
    plt.figure(4)
    for i in range(len(GAPS)):
        gap = GAPS[i]
        rz_, rx_ = rz[gap], 1e3*rx[gap]
        label = GAPS[i] + ' mm' + ' rx @ end: {:+.2f} um'.format(rx_[-1])
        plt.plot(rz_, rx_, colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('pos [um]')
    plt.grid()
    plt.legend()
    plt.title(
        'RK Trajectory PosX for phase {} mm'.format(phase))
    plt.savefig(fig_path + 'traj-posx.png', dpi=dpi)
    if show_flag:
        plt.show()
    plt.close()

    # plot ry
    plt.figure(6)
    for i in range(len(GAPS)):
        gap = GAPS[i]
        rz_, ry_ = rz[gap], 1e3*ry[gap]
        label = GAPS[i] + ' mm' + ' ry @ end: {:+.2f} um'.format(ry_[-1])
        plt.plot(rz_, ry_, colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('pos [um]')
    plt.grid()
    plt.legend()
    plt.title(
        'RK Trajectory PosY for phase {} mm'.format(phase))
    plt.savefig(fig_path + 'traj-posy.png', dpi=dpi)
    if show_flag:
        plt.show()
    plt.close()


def plot_rk_traj_ang(fig_path, colors, dpi, show_flag, rz, px, py):
    """."""
    # plot px
    plt.figure(5)
    for i in range(len(GAPS)):
        gap = GAPS[i]
        rz_, px_ = rz[gap], 1e6*px[gap]
        label = GAPS[i] + ' mm' + ' px @ end: {:+.2f} urad'.format(px_[-1])
        plt.plot(rz_, px_, colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('ang [urad]')
    plt.grid()
    plt.legend()
    plt.title(
        'RK Trajectory AngX for phase {} mm'.format(phase))
    plt.savefig(fig_path + 'traj-angx.png', dpi=dpi)
    if show_flag:
        plt.show()
    plt.close()

    # plot py
    plt.figure(7)
    for i in range(len(GAPS)):
        gap = GAPS[i]
        rz_, py_ = rz[gap], 1e6*py[gap]
        label = GAPS[i] + ' mm' + ' py @ end: {:+.2f} urad'.format(py_[-1])
        plt.plot(rz_, py_, colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('ang [urad]')
    plt.grid()
    plt.legend()
    plt.title(
        'RK Trajectory AngY for phase {} mm'.format(phase))
    plt.savefig(fig_path + 'traj-angy.png', dpi=dpi)
    if show_flag:
        plt.show()
    plt.close()


def plot_rk_traj(traj_data, phase, show_flag=False):
    """."""
    data = traj_data
    bx, by, bz = data['bx'], data['by'], data['bz']
    rx, ry, rz = data['rx'], data['ry'], data['rz']
    fmapbx, fmapby = data['fmapbx'], data['fmapby']
    fmaprz = data['fmaprz']
    px, py = data['px'], data['py']
    i1bx, i1by = data['i1bx'], data['i1by']
    i2bx, i2by = data['i2bx'], data['i2by']

    fig_path = 'results/phase-organized/' + phase + '/'
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    dpi = 300

    plot_rk_traj_normalized_fields(
        fig_path, colors, dpi, show_flag, rz, bx, by, bz)
    
    return

    plot_rk_traj_pos(fig_path, colors, dpi, show_flag, rz, rx, ry)
    plot_rk_traj_ang(fig_path, colors, dpi, show_flag, rz, px, py)

    # # generate table
    row1 = [
        'Gap [mm]',
        'Bx 1st integral [G cm] / \Deltapy [urad]',
        'By 1st integral [G cm] / \Deltapx [urad]',
        'Bx 2nd integral [G cm²] / \Deltay [um]',
        'By 2nd integral [G cm²] / \Deltax [um]']
    row_list = []
    row_list.append(row1)
    for gap in GAPS:
        px_ = 1e6*px[gap][-1]
        py_ = 1e6*py[gap][-1]
        rx_ = 1e3*rx[gap][-1]
        ry_ = 1e3*ry[gap][-1]
        i1bx_ = 1e6*i1bx[gap][-1]
        i1by_ = 1e6*i1by[gap][-1]
        i2bx_ = 1e8*i2bx[gap][-1]
        i2by_ = 1e8*i2by[gap][-1]
        px_ = format(px_, '+4.2f')
        py_ = format(py_, '+4.2f')
        rx_ = format(rx_, '+4.2f')
        ry_ = format(ry_, '+4.2f')
        i1bx_ = format(i1bx_, '+5.1f')
        i1by_ = format(i1by_, '+5.1f')
        i2bx_ = format(i2bx_, '+3.2e')
        i2by_ = format(i2by_, '+3.2e')
        row = [
            gap,
            '{} / {}'.format(i1bx_, py_),
            '{} / {}'.format(i1by_, px_),
            '{} / {}'.format(i2bx_, ry_),
            '{} / {}'.format(i2by_, rx_)]
        row_list.append(row)

    print("\bbegin{table }[H]")
    print("\centering")
    print("\caption{Primeiras integrais de campo}")
    print("\bbegin{tabular }{|c|c|c|}")
    print("\hline")
    print(row_list[0][0] + " & " + row_list[0][1]
          + " & " + row_list[0][2] + " \\")
    print("\hline")
    for i, gap in enumerate(GAPS):
        print(
            row_list[i+1][0] + " & " + row_list[i+1][1]
            + " & " + row_list[i+1][2] + " \\")

    print("\bbegin{table }[H]")
    print("\centering")
    print("\caption{Segundas integrais de campo}")
    print("\bbegin{tabular }{|c|c|c|}")
    print("\hline")
    print(row_list[0][0] + " & " + row_list[0][3]
          + " & " + row_list[0][4] + " \\")
    print("\hline")
    for i, gap in enumerate(GAPS):
        print(
            row_list[i+1][0] + " & " + row_list[i+1][3]
            + " & " + row_list[i+1][4] + " \\")


if __name__ == "__main__":
    """."""
    traj_data, *_ = load_rk_traj()

    for phase in [PHASES[2]]:
        print(phase)
        plot_rk_traj(traj_data[phase], phase, show_flag=False)
