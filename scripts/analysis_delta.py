#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from idanalysis.trajectory import IDTrajectory

from pyaccel.optics.edwards_teng import estimate_coupling_parameters
from idanalysis.deltadata import DeltaData
from idanalysis.model import calc_optics, create_model, get_id_sabia_list

from idanalysis.utils import create_deltadata, FOLDER_BASE


def dGV_plot_config_space(configs, save=True):
    configs.plot_dGV_config_space(save=True)


def dGV_calc_posang_respm(configs, s_step=0.1):
    for config in configs:
        if configs.check_is_dGV(config):
            fmap, label = configs.get_fieldmap(config)
            print(label)
            traj = IDTrajectory(label=label, fieldmap=fmap)
            respm = traj.calc_posang_respm(s_step=s_step, save=True)


def dGV_correct_posang(configs, s_step=0.1):
    for config in configs:
        if configs.check_is_dGV(config):
            # create fieldmap
            fmap, label = configs.get_fieldmap(config)
            print(label)
            # create traj object
            traj = IDTrajectory(label=label, fieldmap=fmap)
            # calc initial traj and save result
            traj.calc_trajectory(s_step=s_step)
            traj.save_posang(label='uncorrected')
            # correct posang
            traj.output_folder = './results/dGV/posang_respm/'
            traj.correct_posang(s_step=s_step)
            # save result
            traj.output_folder = ''
            traj.save_posang(label='corrected')


def dGV_plot_rktraj(configs):

    data = dict()
    for config in configs:
        if configs.check_is_dGV(config):
            # create fieldmap
            label = configs.get_config_label(config)
            print(label)
            dP = configs.get_dP(config)
            datum = data.get(dP, dict(dGV=[], 
                hkicks_corrected=[], vkicks_corrected=[],
                hkicks_uncorrected=[], vkicks_uncorrected=[],
                ))
            # dGV
            dGV = configs.get_dGV(config)
            datum['dGV'].append(dGV)
            
            # create traj object
            fmap, label = configs.get_fieldmap(config)
            traj = IDTrajectory(label=label, fieldmap=fmap)
            traj.output_folder = './results/dGV/posang/'
            
            # corrected trajectory
            traj.load_posang(label='corrected')
            kicks = traj.kicks
            traj.fit_parabola()

            # corrected kicks
            datum['hkicks_corrected'].append(kicks[1])
            datum['vkicks_corrected'].append(kicks[2])
            # uncorrected kicks
            traj.load_posang(label='uncorrected')
            kicks = traj.kicks
            datum['hkicks_corrected'].append(kicks[1])
            datum['vkicks_corrected'].append(kicks[2])
            # add back to data dict
            data[dP] = datum

    for dP, datum in data.items():
        print(dP, datum)


def print_configs(configs):

    data = dict()
    for config in configs:
        label = configs.get_config_label(config)
        header = configs.get_header(config)
        fstr = '{:<40s}: Kh={:.2f} Kv={:.2f}'
        print(fstr.format(label, header['K_Horizontal'], header['K_Vertical']))


def run_calc_optics():

    folder = FOLDER_BASE + DeltaData.FOLDER_SABIA_ERR
    configs_dict = dict(folder=folder, configs=DeltaData.CONFIGS_ERR)
    configs = DeltaData(**configs_dict)
    # kickmap = configs._read_kickmap(configs[0])
    # print(kickmap.length)
    
    optics = []
    for config in configs:
        print(config)
        data = dict(config=config)
        kmap_fname = configs.get_kickmap_filename(config)
        ids = get_id_sabia_list(kmap_fname)
        data.update(calc_optics(ids, vchamber_on=False))
        optics.append(data)
    return configs, optics


# --- legacy (to be adapted and tested in new version lib version)

def run():
    
    configs = DeltaData(folder=DeltaData.FOLDER_SABIA)

    print_configs(configs)
    dGV_plot_config_space(configs)
    # dGV_calc_posang_respm(configs, s_step=0.1)
    # dGV_correct_posang(configs, s_step=0.1)
    # dGV_plot_rktraj(configs)


# def run_new():
    
#     configs_dict = dict(folder=DeltaData.FOLDER_SABIA_ERR, configs=DeltaData.CONFIGS_ERR)
#     configs = DeltaData(**configs_dict)

#     print_configs(configs)
#     dGV_plot_config_space(configs)
#     dGV_calc_posang_respm(configs, s_step=0.1)
#     # dGV_correct_posang(configs, s_step=0.1)
#     # dGV_plot_rktraj(configs)


# def plot_optics(configs, optics):
#     """."""

#     nominal = calc_optics(dict(), vchamber_on=False)

#     pos2config = {
#         (0, -1.0): 'Horiz K_zero',
#         (0, -0.5): 'Horiz K_mid',
#         (0, 0): 'Horiz K_max',
#         (0.5, -1.0): 'LHCirc K_zero',
#         (0.5, -0.5): 'LHCirc K_mid',
#         (0.5, 0.0): 'LHCirc K_max',
#         (1, -1): 'Vert K_zero',
#         (1, -0.5): 'Vert K_mid',
#         (1, 0.0): 'Vert K_max',
#         }

#     tune1 = dict()
#     tune2 = dict()
#     edt_tune1 = dict()
#     edt_tune2 = dict()
#     emit1 = dict()
#     emit2 = dict()
#     tune_sep = dict()
#     emit_ratio = dict()
#     for datum in optics:
#         config = datum['config']
#         dP = configs.get_dP(config)
#         dGV = configs.get_dGV(config)
#         # print('{}: (dP={}, dGV={})'.format(config, dP, dGV))
#         pos = (dP, dGV)
#         if pos not in tune1:
#             tune1[pos] = [datum['tune1'], ]
#             tune2[pos] = [datum['tune2'], ]
#             edt_tune1[pos] = [datum['edt_tune1'], ]
#             edt_tune2[pos] = [datum['edt_tune2'], ]
#             emit1[pos] = [datum['emit1'], ]
#             emit2[pos] = [datum['emit2'], ]
#             tune_sep[pos] = [datum['edt_tune_sep']]
#             emit_ratio[pos] = [datum['edt_emit_ratio'], ]
#         else:
#             tune1[pos].append(datum['tune1'])
#             tune2[pos].append(datum['tune2'])
#             edt_tune1[pos].append(datum['edt_tune1'])
#             edt_tune2[pos].append(datum['edt_tune2'])
#             emit1[pos].append(datum['emit1'])
#             emit2[pos].append(datum['emit2'])
#             tune_sep[pos].append(datum['edt_tune_sep'])
#             emit_ratio[pos].append(datum['edt_emit_ratio'])

#     # edwards-teng: emit_avg
#     _, ax = plt.subplots()
#     idx = 0
#     for i, pos in enumerate(tune1.keys()):
#         config = pos2config[pos]
#         data = emit_ratio[pos]
#         ind = range(idx, idx+len(data))
#         avg = 100 * np.average(data, axis=1)
#         std = 100 * np.std(data, axis=1)
#         max = 100 * np.max(data, axis=1)
#         min = 100 * np.min(data, axis=1)
#         for j in range(len(avg)):
#             ax.plot([ind[j], ind[j]], [avg[j] - std[j], avg[j] + std[j]], '-', color='C'+str(i))
#             ax.plot([ind[j], ind[j]], [min[j], max[j]], '--', color='C'+str(i))
#         ax.plot(ind, avg, 'o', label=config)
#         idx += len(data)
#     ax.set_yscale('log')
#     plt.legend()
#     plt.xlabel('ID configuration number')
#     plt.ylabel(r'Emittance ratio [%]')
#     plt.grid()
#     plt.title('Emittance ratio (Edwards-Teng)')
#     plt.show()
    
#     # edward-teng: tune_sep
#     idx = 0
#     for pos in tune1.keys():
#         config = pos2config[pos]
#         data = 100 * np.array(tune_sep[pos])
#         plt.plot(range(idx, idx+len(data)), data, 'o', label=config)
#         idx += len(data)
#     plt.legend()
#     plt.xlabel('ID configuration number')
#     plt.ylabel(r'Tune separation [%]')
#     plt.grid()
#     plt.title('Min. Tune Separation (Edwards-Teng)')
#     plt.show()

#     # edward-teng: tune1
#     idx = 0
#     for pos in edt_tune1.keys():
#         config = pos2config[pos]
#         data = 1000*(np.array(edt_tune1[pos]) - nominal['edt_tune1'])
#         plt.plot(range(idx, idx+len(data)), data, 'o', label=config)
#         idx += len(data)
#     plt.legend()
#     plt.xlabel('ID configuration number')
#     plt.ylabel(r'$\delta\nu_1 \times 1000$')
#     plt.grid()
#     plt.title('Tune1 Shift (Edwards-Teng)')
#     plt.show()

#     # edward-teng: tune2
#     idx = 0
#     for pos in edt_tune2.keys():
#         config = pos2config[pos]
#         data = 1000*(np.array(edt_tune2[pos]) - nominal['edt_tune2'])
#         plt.plot(range(idx, idx+len(data)), data, 'o', label=config)
#         idx += len(data)
#     plt.legend()
#     plt.xlabel('ID configuration number')
#     plt.ylabel(r'$\delta\nu_2 \times 1000$')
#     plt.grid()
#     plt.title('Tune2 Shift (Edwards-Teng)')
#     plt.show()

#     # tune1
#     idx = 0
#     for pos in tune1.keys():
#         config = pos2config[pos]
#         data = 1000*(np.array(tune1[pos]) - nominal['tune1'])
#         plt.plot(range(idx, idx+len(data)), data, 'o', label=config)
#         idx += len(data)
#     plt.legend()
#     plt.xlabel('ID configuration number')
#     plt.ylabel(r'$\delta\nu_1 \times 1000$')
#     plt.grid()
#     plt.title('Tune1 Shift (Ohmi Envelope)')
#     plt.show()

#     # tune2
#     idx = 0
#     for pos in tune2.keys():
#         config = pos2config[pos]
#         data = 1000*(np.array(tune2[pos]) - nominal['tune2'])
#         plt.plot(range(idx, idx+len(data)), data, 'o', label=config)
#         idx += len(data)
#     plt.legend()
#     plt.xlabel('ID configuration number')
#     plt.ylabel(r'$\delta\nu_2 \times 1000$')
#     plt.grid()
#     plt.title('Tune2 Shift (Ohmi Envelope)')
#     plt.show()

#     # coupling
#     idx = 0
#     for pos in emit1.keys():
#         config = pos2config[pos]
#         data = 100 * np.array(emit2[pos]) / np.array(emit1[pos])
#         plt.plot(range(idx, idx+len(data)), data, 'o', label=config)
#         idx += len(data)
#     plt.legend()
#     plt.xlabel('ID configuration number')
#     plt.ylabel(r'Emittance ratio [%]')
#     plt.grid()
#     plt.title('Emittance Ratio (Ohmi Envelope)')
#     plt.show()


# def run_err():

#     folder = FOLDER_BASE + DeltaData.FOLDER_SABIA_ERR

#     configs_dict = dict(folder=folder, configs=DeltaData.CONFIGS_ERR)
#     configs = DeltaData(**configs_dict)
#     kmap_fname = configs.get_kickmap_filename(configs[1])
#     ids = get_id_sabia_list(kmap_fname)
#     data = calc_optics(ids, vchamber_on=False)
#     edt = data['edt']
#     r = estimate_coupling_parameters(edt)
#     print(r)

#     # configs, optics = run_calc_optics()
#     # plot_optics(configs, optics)
#     # return optics



    
if __name__ == '__main__':
    
    configs = create_deltadata()
    # run()
    # run_err()
    # run_new()
    dGV_plot_config_space(configs)
