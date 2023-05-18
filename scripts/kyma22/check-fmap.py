#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from idanalysis import IDKickMap
from idanalysis.fmap import FieldmapOnAxisAnalysis as fm

import utils


def calc_roll_off(fmap):
    by = fmap.By
    idxmax = np.argmax(by[2000:10000])
    idxmax += 2000
    by_x = np.array(fmap.field.by)
    byx = by_x[0, :, idxmax]
    rx = fmap.field.rx
    rx0idx = np.argmin(np.abs(rx))
    rx5idx = np.argmin(np.abs(rx - 5))
    roff = 100*(byx[rx0idx] - byx[rx5idx])/byx[rx0idx]
    # plt.figure(1)
    # plt.plot(rx, byx, label='roll off @ 5 mm= {:.2} %'.format(roff))
    # plt.legend()
    # plt.xlabel('rx [mm]')
    # plt.ylabel('By [T]')
    # plt.grid()
    # plt.show()
    return rx, byx, roff


def plot_field(fmap):
    by = fmap.By
    rz = fmap.rz
    plt.figure(1)
    plt.plot(rz, by)
    plt.xlabel('rz [mm]')
    plt.ylabel('By [T]')
    plt.grid()
    plt.show()


def load_fieldmap(phase):
    MEAS_FILE = utils.MEAS_FILE
    fmap_fname = MEAS_FILE
    fmap_fname = fmap_fname.replace('phase0', 'phase{}'.format(phase))
    fmap = fm(fieldmap=fmap_fname)
    fmap.load_fieldmap()
    return fmap


def compare_model(fmap, phase):
    kyma = utils.generate_radia_model(phase=phase, nr_periods=51)
    by_fmap = fmap.By
    bx_fmap = fmap.Bx
    bz_fmap = fmap.Bz
    rz_fmap = fmap.rz
    rz_model = np.linspace(rz_fmap[0], rz_fmap[-1], 701)
    field = kyma.get_field(0, 0, rz_model)
    by = field[:, 1]
    bx = field[:, 0]
    bz = field[:, 2]
    plt.figure(1)
    fcomp = 'y'
    plt.plot(rz_fmap, by_fmap, color='C0', label='by - measurement')
    plt.plot(rz_model, by, color='C1', label='by - model')
    plt.xlabel('rz [mm]')
    plt.ylabel('By [T]')
    plt.legend()
    plt.grid()
    figname = utils.FOLDER_DATA + 'B{}_comparison_phase{}'.format(fcomp, phase)
    plt.savefig(figname, dpi=300)

    plt.figure(2)
    fcomp = 'x'
    plt.plot(rz_fmap, bx_fmap, color='C0', label='bx - measurement')
    plt.plot(rz_model, bx, color='C1', label='bx - model')
    plt.xlabel('rz [mm]')
    plt.ylabel('Bx [T]')
    plt.legend()
    plt.grid()
    figname = utils.FOLDER_DATA + 'B{}_comparison_phase{}'.format(fcomp, phase)
    plt.savefig(figname, dpi=300)

    plt.figure(3)
    fcomp = 'z'
    plt.plot(rz_fmap, bz_fmap, color='C0', label='bz - measurement')
    plt.plot(rz_model, bz, color='C1', label='bz - model')
    plt.xlabel('rz [mm]')
    plt.ylabel('Bz [T]')
    plt.legend()
    plt.grid()
    figname = utils.FOLDER_DATA + 'B{}_comparison_phase{}'.format(fcomp, phase)
    plt.savefig(figname, dpi=300)

    rx, byx, roff_meas = calc_roll_off(fmap)

    period = kyma.period_length
    rz = np.linspace(-period/2, period/2, 100)
    field = kyma.get_field(0, 0, rz)
    by = field[:, 1]
    by_max_idx = np.argmax(by)
    rz_at_max = rz[by_max_idx]
    field = kyma.get_field(rx, 0, rz_at_max)
    by = field[:, 1]
    by_avg = by
    rx_avg = rx
    rx5_idx = np.argmin(np.abs(rx_avg - 5))
    rx0_idx = np.argmin(np.abs(rx_avg))
    roff = 100*np.abs(by_avg[rx5_idx]/by_avg[rx0_idx]-1)

    plt.figure(4)
    plt.plot(
        rx, byx, color='C0', label='roll off @ 5mm - meas  {:.2f} %'.format(roff_meas))
    plt.plot(
        rx_avg, by_avg, color='C1', label='roll off @ 5mm- model  {:.2f} %'.format(roff))
    plt.xlabel('rx [mm]')
    plt.ylabel('By [T]')
    plt.legend()
    plt.grid()
    figname = utils.FOLDER_DATA + 'Roll_off_phase{}'.format(phase)
    plt.savefig(figname, dpi=300)

    plt.figure(5)
    byx -= byx[int(len(byx)/2)] - by_avg[int(len(by_avg)/2)]
    plt.plot(
        rx, byx, color='C0', label='roll off @ 5mm - meas  {:.2f} %'.format(roff_meas))
    plt.plot(
        rx_avg, by_avg, '.-', color='C1', label='roll off @ 5mm- model  {:.2f} %'.format(roff))
    plt.xlabel('rx [mm]')
    plt.ylabel('By [T]')
    plt.legend()
    plt.grid()
    figname = utils.FOLDER_DATA + 'Roll_off_shift_phase{}'.format(phase)
    plt.savefig(figname, dpi=300)



    plt.show()


def run(phase):
    fmap = load_fieldmap(phase=phase)
    i1 = fmap.calc_first_integral(fmap.By, fmap.rz)
    i2 = fmap.calc_first_integral(i1, fmap.rz)
    print('First integral: {:.2} Tm  or {:.3f} urad'.format(
        i1[-1], 1e6*i1[-1]/10))
    print('Second integral: {:.2} Tm2  or {:.3f} mm'.format(
        i2[-1], 1e3*i2[-1]/10))
    # plot_field(fmap)

    # calc_roll_off(fmap)
    # compare_model(fmap, phase=phase)


if __name__ == "__main__":
    """."""

    phase = 11
    run(phase=phase)
