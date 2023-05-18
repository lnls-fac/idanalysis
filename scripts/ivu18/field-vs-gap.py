#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from imaids.blocks import Block as Block
from mathphys.functions import save_pickle, load_pickle

import utils


SOLVE_FLAG = utils.SOLVE_FLAG
NOMINAL_GAP = utils.NOMINAL_GAP


def get_termination_parameters(width=64):
    """."""
    fname = utils.FOLDER_DATA + 'respm_termination_{}.pickle'.format(width)
    term = load_pickle(fname)
    b1t, b2t, b3t, dist1, dist2 = term['results']
    return list([b1t, b2t, b3t, dist1, dist2])


def get_field_vs_gap(ivu, gaps, rx, peak_idx, data):
    """."""
    by_x, rx_dict, roll_off, field_amp = dict(), dict(), dict(), dict()
    period = ivu.period_length
    rz = np.linspace(-period/2, period/2, 100)

    # iterate on gaps
    for i, gap in enumerate(gaps):
        ivu.dg = gap - NOMINAL_GAP
        ivu.solve()
        field = ivu.get_field(0, 0, rz)
        by = field[:, 1]
        by_max_idx = np.argmax(by)
        rz_max = rz[by_max_idx] + peak_idx*period
        field = ivu.get_field(rx, 0, rz_max)
        by = field[:, 1]
        by_list = list()
        for i, _ in enumerate(rx):
            if i >= 6 and i <= len(rx)-7:
                by_temp = by[i-6] + by[i-5] + by[i-4] + by[i-3]
                by_temp += by[i-2] + by[i-1] + by[i] + by[i+1] + by[i+2]
                by_temp += by[i+3] + by[i+4] + by[i+5] + by[i+6]
                by_temp = by_temp/13
                by_list.append(by_temp)
        by_avg = np.array(by_list)
        rx_avg = rx[6:-6]
        rx6_idx = np.argmin(np.abs(rx_avg - utils.ROLL_OFF_RX))
        rx0_idx = np.argmin(np.abs(rx_avg))
        roff = np.abs(by_avg[rx6_idx]/by_avg[rx0_idx]-1)

        idx_zero = np.argmin(np.abs(rx))
        field_amp[gap] = by[idx_zero]

        by_x[gap] = by_avg
        rx_dict[gap] = rx_avg
        roll_off[gap] = roff

        ivu.dg = 0

    data['rolloff_by'] = by_x
    data['rolloff_rx'] = rx_dict
    data['rolloff_value'] = roll_off
    data['field_amp'] = field_amp

    return data


def run_generate_data(width, rx, gaps):
    """."""
    rx = rx or np.linspace(-40, 40, 4*81)
    gaps = gaps or np.arange(4, 21, 1)

    data = dict()
    termination_parameters = get_termination_parameters()
    ivu = utils.generate_radia_model(
        gap=NOMINAL_GAP, width=width,
        termination_parameters=termination_parameters,
        solve=SOLVE_FLAG)
    data = get_field_vs_gap(
        ivu=ivu, gaps=gaps, rx=rx,
        peak_idx=0, data=data)
    fname = utils.FOLDER_DATA + 'field-vs-gap-width{}.pickle'.format(width)
    save_pickle(data, fname,
                overwrite=True)


def load_data(width):
    fname = utils.FOLDER_DATA + 'field-vs-gap-width{}.pickle'.format(width)
    data = load_pickle(fname)
    field_amp = data['field_amp']
    roll_off = data['roll_off']  # This is the old name of this key

    return field_amp, roll_off


def plot_product_field_roff(widths):

    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    for width, color in zip(widths, colors):
        by, roff = load_data(width=width)
        gaps = np.array(list(by.keys()))
        by = np.array((list(by.values())))
        roff = np.array((list(roff.values())))
        plt.semilogy(
            gaps, 1e4*roff*by, '.-', color=color,
            label='{:d} mm'.format(width))
        plt.ylabel('Delta field [G]')
        plt.xlabel('Gap [mm]')
        plt.xlim(4, 20)
        plt.legend()
        plt.title('Absolute Field Rolloff @ x=6mm'.format(width))
        plt.tight_layout()

    plt.show()


def plot_field_roff(widths):

    gaps = np.linspace(1, 20, 20)
    for width in widths:
        by, roff = load_data(width=width)
        gaps = np.array(list(by.keys()))
        by = np.array((list(by.values())))
        roff = np.array((list(roff.values())))
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(gaps, by, label='Field amplitude')
        ax1.set_ylabel('Field [T]')
        ax1.set_xlabel('Gap [mm]')
        ax1.legend(loc='upper left')
        ax1.set_ylim(0, 1.5)
        ax2.plot(gaps, 100*roff, label='Field Roll-off at 6mm', color='C1')
        ax2.set_ylabel('Roll off [%]')
        ax2.legend(loc='upper right')
        plt.xlim(4, 20)
        plt.title('Field for width = {}'.format(width))
        plt.tight_layout()
        fpath = utils.FOLDER_BASE
        fpath = fpath.replace('/model/data', '/model')
        fig_path = fpath + 'width_{}/field_and_roff_width{}'.format(
            width, width)
        # plt.savefig(fig_path, dpi=300)
        plt.show()


if __name__ == '__main__':
    widths = [64]
    plot_field_roff(widths=widths)
