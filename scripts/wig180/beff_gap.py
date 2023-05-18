#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize

from idanalysis.fmap import FieldmapOnAxisAnalysis
from fieldmaptrack import FieldMap

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import ID_CONFIGS

from imaids import utils as ima_utils


def calc_beff(z,B):
    freqs = 2*np.pi*np.array([1/180,3/180,5/180])
    amps, *_ = ima_utils.fit_fourier_components(B,freqs,z)
    return amps


def fit_function(x, a, b, c):
        return a*np.exp(-b*x) + c


def fit_measurement(gap,beff):

    a0, b0, c0 = 3, 0.5, 0.5
    return optimize.curve_fit(
        fit_function, gap, beff)[0]


def readfield(idconfig, init):

    MEAS_FILE = ID_CONFIGS[idconfig]

    _, meas_id =  MEAS_FILE.split('ID=')
    meas_id = meas_id.replace('.dat', '')

    file_name = FOLDER_BASE + DATA_PATH + MEAS_FILE

    substr = "gap"
    if substr in file_name:
        idx = file_name.index(substr)
        gap = file_name[idx+4:idx+9]
        gap_value = float(gap)

    print(file_name)

    with open(file_name, encoding='utf-8', errors='ignore') as my_file:
        data_col1 = []
        data_col2 = []
        for i, line in enumerate(my_file):
            if i >= init:
                line = line.strip()
                for i in range(10):
                    line = line.replace('  ', '')
                    line = line.replace('\t\t', ' ')
                line = line.replace('\t', ' ')
                list_data = line.split() #returns a list
                try:
                    xvalue = float(list_data[0])
                except ValueError:
                    xvalue = ((list_data[0]))

                if xvalue == 0:
                    try:
                        data_col1.append(float(list_data[2]))
                        data_col2.append(float(list_data[4]))
                    except ValueError:
                        data_col1.append((list_data[2]))
                        data_col2.append((list_data[4]))


    my_file.close()
    z = np.array(data_col1)
    B = np.array(data_col2)
    return z, B, gap_value


def run():
    """."""
    config_list = ['ID3986','ID3987','ID3988','ID3989','ID3990',
        'ID4021','ID3991','ID4005','ID3992','ID3993','ID3994']

    period = 180
    beff_list = []
    keff_list = []
    gap_list = []
    for i, idconfig in enumerate(config_list):
        z, B, gap = readfield(idconfig, 2)
        fraction = int(len(z)/4)
        amps = calc_beff(z[fraction:3*fraction],B[fraction:3*fraction])
        beff = np.sqrt(amps[0]**2+amps[1]**2+amps[2]**2)
        keff = ima_utils.calc_deflection_parameter(beff, 0.18)
        beff_list.append(beff)
        keff_list.append(keff)
        gap_list.append(gap)

    gap_array = np.array(gap_list)
    # gap_array = gap_array/period
    curve_fit = fit_measurement(gap_array, beff_list)
    a=curve_fit[0]
    b=curve_fit[1]
    c = curve_fit[2]
    gaps = np.arange(23,300,1)
    # gaps = gaps/period
    fitted_curve = fit_function(gaps,a,b,c)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    label = 'B = {:.2f} exp(- {:.3f} * Gap / 180)'.format(a, 180*b)
    ax1.plot(gaps,fitted_curve, '--', color='C1', label=label)
    ax1.plot(gap_array,beff_list, 'o', color='b', label='Medida')
    ax2.plot(gap_array,keff_list, 'o', color='b')
    
    print(a,b,c)
    
    b_fit = fit_function(49.73,a,b,c)
    b_fit = fit_function(23,a,b,c)
    print(b_fit)
    k_fit = ima_utils.calc_deflection_parameter(b_fit, 0.18)
    print(k_fit)

    # curve_fit = fit_measurement(gap_array, keff_list)
    # a=curve_fit[0]
    # b=curve_fit[1]
    # c = curve_fit[2]
    # fitted_curve = fit_function(23,a,b,c)
    # print(fitted_curve)

    ax1.set_xlabel('Gap [mm]')
    ax1.set_ylabel('B [T]')
    ax2.set_ylabel('K')
    ax1.legend()
    ax1.grid()
    plt.title('Campo Magnético do WIG180')
    plt.show()
    gaps2 = np.linspace(23,300,1000)
    beff = fit_function(gaps2/180,a,b,c)
    # for i, gap in enumerate(gaps2):
    #     gap = format(gap, '03.2f')
    #     beff[i] = format(beff[i], '03.2f')
    #     print(gap,'&', beff[i])

    # fig.savefig('results/b_vs_gap.png', dpi=300)


def plot_field(idconfig):
    """."""
    rz, B, gap = readfield(idconfig, 2)
    plt.plot(rz, B, 'o')
    plt.show()


if __name__ == '__main__':
    """."""
    run()
    # idconfig = 'ID4020' # gap 045.00mm/2022-09-01_WigglerSTI_45mm_U+0.00_D+0.00_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=4020.dat',
    idconfig = 'ID4019' # gap 049.73mm/2022-09-01_WigglerSTI_49.73mm_U+0.00_D+0.00_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=4019.dat',
    # idconfig = 'ID3979' # gap 059.60mm/2022-08-25_WigglerSTI_059.60mm_U+0.00_D+0.00_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=3979.dat',

    # plot_field(idconfig=idconfig)
