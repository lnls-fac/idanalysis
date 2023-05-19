#!/usr/bin/env python-sirius

import numpy as np


from utils import FOLDER_BASE
from utils import MEAS_DATA_PATH

folder_m5 = FOLDER_BASE + MEAS_DATA_PATH + 'probes 133-14/gap 22.0mm/y=-0.5mm/'
folder_00 = FOLDER_BASE + MEAS_DATA_PATH + 'probes 133-14/gap 22.0mm/'
folder_p5 = FOLDER_BASE + MEAS_DATA_PATH + 'probes 133-14/gap 22.0mm/y=+0.5mm/'
folder_list = [folder_m5, folder_00, folder_p5]

fname_0_list = ['2022-10-28_EPU_gap22.0_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4113.dat', '2022-10-19_EPU_gap22.0_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4079.dat', '2022-10-31_EPU_gap22.0_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4114.dat']

fname_m16_list = ['2022-11-03_EPU_gap22.0_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4183.dat', '2022-10-20_EPU_gap22.0_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4081.dat', '2022-11-03_EPU_gap22.0_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4184.dat']

fname_p16_list = ['2022-11-03_EPU_gap22.0_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4181.dat', '2022-10-20_EPU_gap22.0_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4080.dat', '2022-11-03_EPU_gap22.0_fase16.39_X=-20_20mm_Z=-1810_1810mm_ID=4182.dat']

fname_m25_list = ['2022-11-04_EPU_gap22.0_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4187.dat', '2022-10-20_EPU_gap22.0_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4083.dat', '2022-11-04_EPU_gap22.0_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4188.dat']

fname_p25_list = ['2022-11-04_EPU_gap22.0_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4185.dat', '2022-10-20_EPU_gap22.0_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4082.dat', '2022-11-04_EPU_gap22.0_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4186.dat']


def readfield(file_name):
    """."""
    init = 2
    with open(file_name, encoding='utf-8', errors='ignore') as my_file:
        data_col1 = []
        data_col2 = []
        data_col3 = []
        data_col4 = []
        data_col5 = []
        data_col6 = []
        for i, line in enumerate(my_file):
            if i >= init:
                list_data = line.split('\t')  # returns a list
                if len(list_data) > 1:
                    try:
                        data_col1.append(float(list_data[0]))
                        data_col2.append(float(list_data[1]))
                        data_col3.append(float(list_data[2]))
                        data_col4.append(float(list_data[3]))
                        data_col5.append(float(list_data[4]))
                        data_col6.append(float(list_data[5]))
                    except ValueError:
                        print('Error')

    my_file.close()
    x = np.array(data_col1)
    y = np.array(data_col2)
    z = np.array(data_col3)
    bx = np.array(data_col4)
    by = np.array(data_col5)
    bz = np.array(data_col6)

    return x, y, z, bx, by, bz


def generate_fieldmap(phase):

    if phase == '00.00':
        fname_list = fname_0_list
    elif phase == '-16.39':
        fname_list = fname_m16_list
    elif phase == '16.39':
        fname_list = fname_p16_list
    elif phase == '-25.00':
        fname_list = fname_m25_list
    elif phase == '25.00':
        fname_list = fname_p25_list

    xlist, ylist, zlist = list(), list(), list()
    bxlist, bylist, bzlist = list(), list(), list()

    for i, folder in enumerate(folder_list):
        file_name = folder + fname_list[i]
        data = readfield(file_name)
        xlist.append(data[0])
        ylist.append(data[1])
        zlist.append(data[2])
        bxlist.append(data[3])
        bylist.append(data[4])
        bzlist.append(data[5])

    x_col1 = list()
    y_col2 = list()
    z_col3 = list()
    b_col4 = list()
    b_col5 = list()
    b_col6 = list()

    for i in np.arange(1801):
        for j, y in enumerate(ylist):
            for k in np.arange(41):
                x_col1.append(xlist[j][41*i+k])
                y_col2.append(ylist[j][41*i+k])
                z_col3.append(zlist[j][41*i+k])
                b_col4.append(bxlist[j][41*i+k])
                b_col5.append(bylist[j][41*i+k])
                b_col6.append(bzlist[j][41*i+k])

    fname = folder_00 + "fieldmaps/fieldmap_EPU_gap22.0_fase" + phase + ".dat"
    my_file = open(fname, "w")  # w=writing
    my_file.write('X[mm]\tY[mm]\tZ[mm]\tBx[T]\tBy[T]\tBz[T]\n')
    my_file.write('-------------------------------------------------------\n')
    for i in np.arange(len(x_col1)):
        my_file.write("{:.1f}\t{:.1f}\t{:.1f}\t{:.5e}\t{:.5e}\t{:.5e}\n".format(x_col1[i], y_col2[i], z_col3[i], b_col4[i], b_col5[i], b_col6[i]))
    my_file.close()


if __name__ == "__main__":
    """."""
    phase_list = ['00.00', '-16.39', '16.39', '-25.00', '25.00']
    for phase in phase_list:
        generate_fieldmap(phase)
