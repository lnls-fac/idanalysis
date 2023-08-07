#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as _plt
from scipy.optimize import curve_fit as _curve_fit

import pyaccel
import pymodels

from fieldmaptrack import Beam

from mathphys.functions import save_pickle as _save_pickle
from mathphys.functions import load_pickle as _load_pickle

from idanalysis import IDKickMap as _IDKickMap
from idanalysis import optics as optics
from idanalysis import orbcorr as orbcorr

from apsuite.dynap import DynapXY

import utils


class Tools:

    class CALC_TYPES:
        nominal = 0
        nonsymmetrized = 1
        symmetrized = 2

    @staticmethod
    def get_gap_str(gap):
        gap_str = '{:04.1f}'.format(gap).replace('.', 'p')
        return gap_str

    @staticmethod
    def get_phase_str(phase):
        phase_str = '{:+07.3f}'.format(phase).replace('.', 'p')
        phase_str = phase_str.replace('+', 'pos').replace('-', 'neg')
        return phase_str

    @staticmethod
    def create_config_path(**kwargs):
        path = ''
        if 'width' in kwargs.keys():
            path += 'widths/width_{}/'.format(kwargs.get('width'))
        if 'phase' in kwargs.keys():
            phase_str = Tools.get_phase_str(kwargs.get('phase'))
            path += 'phases/phase_{}/'.format(phase_str)
        if 'gap' in kwargs.keys():
            gap_str = Tools.get_gap_str(kwargs.get('gap'))
            path += 'gap_{}/'.format(gap_str)

        forbidden_list = ['width', 'phase', 'gap']
        items = list(kwargs.items())
        items_ = [elem for elem in items if elem[0] not in forbidden_list]
        for key, value in items_:
            path += '{}/{}_{}/'.format(key + 's', key, value)
        return path

    @staticmethod
    def get_data_path(folder_data, meas_flag, **kwargs):
        fpath = folder_data
        if meas_flag:
            fpath = fpath.replace('model/', 'measurements/')
        config = Tools.create_config_path(**kwargs)
        fpath += config
        return fpath

    @staticmethod
    def get_meas_data_path(folder_data, **kwargs):
        fpath = Tools.get_data_path(folder_data, meas_flag=True, **kwargs)
        return fpath

    @staticmethod
    def get_kmap_filename(folder_data, shift_flag=False, filter_flag=False,
                          linear=False, meas_flag=False, **kwargs):
        fpath = folder_data + 'kickmaps/'
        fpath = fpath.replace('model/data/', 'model/')
        if meas_flag:
            fpath = fpath.replace('model/', 'measurements/')

        fname = 'kickmap-ID'
        if 'width' in kwargs.keys():
            fname += '_width{}/'.format(kwargs.get('width'))
        if 'phase' in kwargs.keys():
            phase_str = Tools.get_phase_str(kwargs.get('phase'))
            fname += '_phase{}'.format(phase_str)
        if 'gap' in kwargs.keys():
            gap_str = Tools.get_gap_str(kwargs.get('gap'))
            fname += '_gap{}'.format(gap_str)

        forbidden_list = ['width', 'phase', 'gap']
        items = list(kwargs.items())
        items_ = [elem for elem in items if elem[0] not in forbidden_list]
        for key, value in items_:
            fname += '_{}{}'.format(key, value)
        fname += '.txt'

        if linear:
            fname = fname.replace('.txt', '-linear.txt')
            return fname
        if shift_flag:
            fname = fname.replace('.txt', '-shifted_on_axis.txt')
        if filter_flag:
            fname = fname.replace('.txt', '-filtered.txt')
        fname = fpath + fname
        print(fname)
        return fname

    @staticmethod
    def create_model_ids(
            fname, rescale_kicks, rescale_length,
            fitted_model, linear, create_ids):
        if linear:
            rescale_kicks = 1
            rescale_length = 1
        if create_ids is None:
            raise ValueError
        ids = create_ids(
            fname, rescale_kicks=rescale_kicks,
            rescale_length=rescale_length)
        model = pymodels.si.create_accelerator(ids=ids)
        if fitted_model:
            famdata = pymodels.si.families.get_family_data(model)
            idcs_qn = _np.array(famdata['QN']['index']).ravel()
            idcs_qs = _np.array(famdata['QS']['index']).ravel()
            data = _load_pickle(utils.FIT_PATH)
            kl = data['KL']
            ksl = data['KsL']
            pyaccel.lattice.set_attribute(model, 'KL', idcs_qn, kl)
            pyaccel.lattice.set_attribute(model, 'KsL', idcs_qs, ksl)
        model.cavity_on = False
        model.radiation_on = 0
        twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
        print('\nModel with ID:')
        print('length : {:.4f} m'.format(model.length))
        print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/_np.pi))
        print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/_np.pi))
        return model, ids

    @staticmethod
    def get_field_component_idx(field_component):
        components = {'bx': 0, 'by': 1, 'bz': 2}
        return components[field_component]

    @staticmethod
    def calc_radia_roll_off(field_component, model, rt, roff_pos,
                            plane='x', peak_idx=0):
        period = model.period_length
        comp_idx = Tools.get_field_component_idx(field_component)
        rz = _np.linspace(-period/2, period/2, 201)
        field = model.get_field(0, 0, rz)
        b = field[:, comp_idx]
        b_max_idx = _np.argmax(b)
        rz_at_max = rz[b_max_idx] + peak_idx*period

        if plane == 'x':
            field = model.get_field(rt, 0, rz_at_max)
        elif plane == 'y':
            field = model.get_field(0, rt, rz_at_max)
        b = field[:, comp_idx]

        roff_idx = _np.argmin(_np.abs(rt-roff_pos))
        rt0_idx = _np.argmin(_np.abs(rt))
        roff = b[roff_idx]/b[rt0_idx]-1
        return b, roff, rz_at_max

    @staticmethod
    def get_meas_idconfig(config_keys, config_dict):
        idconfig = config_dict['configs'][config_keys]
        print(idconfig)
        return idconfig

    @staticmethod
    def get_fmap_fname(config_keys=None, config_dict=None, idconfig=None):
        if idconfig is None:
            if config_dict is None:
                raise ValueError
            idconfig = Tools.get_meas_idconfig(config_keys, config_dict)
        MEAS_FILE = utils.ID_CONFIGS[idconfig]
        _, meas_id = MEAS_FILE.split('ID=')
        meas_id = meas_id.replace('.dat', '')
        fmap_fname = utils.MEAS_DATA_PATH + MEAS_FILE
        return fmap_fname

    @staticmethod
    def get_fmap(fmap_fname):
        idkickmap = _IDKickMap()
        idkickmap.fmap_fname = fmap_fname
        fmap = idkickmap.fmap_config.fmap
        return fmap

    @staticmethod
    def get_fmap_roll_off(field_component, fmap, roff_pos,
                          period, plane='x'):

        if field_component == 'by':
            b = fmap.by[fmap.ry_zero][fmap.rx_zero][:]
        elif field_component == 'bx':
            b = fmap.bx[fmap.ry_zero][fmap.rx_zero][:]
        elif field_component == 'bz':
            b = fmap.bz[fmap.ry_zero][fmap.rx_zero][:]

        if plane == 'x':
            rt = fmap.rx
        elif plane == 'y':
            rt = fmap.ry

        idxmax = _np.argmax(_np.abs(
            b[int(len(b)/2-period):int(len(b)/2+period)]))
        idxmax += int(len(b)/2-period)

        rzmax = fmap.rz[idxmax]
        print('rz peak: {} mm'.format(rzmax))
        print('b peak: {} T'.format(b[idxmax]))

        if field_component == 'by':
            b = _np.array(fmap.by)
        elif field_component == 'bx':
            b = _np.array(fmap.bx)
        elif field_component == 'bz':
            b = _np.array(fmap.bz)

        if plane == 'x':
            b_rt = b[fmap.ry_zero, :, idxmax]
        elif plane == 'y':
            b_rt = b[:, fmap.rx_zero, idxmax]

        rt0_idx = _np.argmin(_np.abs(rt))
        rt_idx = _np.argmin(_np.abs(rt - roff_pos))
        roff = 100*(b_rt[rt0_idx] - b_rt[rt_idx])/b_rt[rt0_idx]

        return rt, b_rt, rzmax, roff

    @staticmethod
    def mkdir_function(mypath):
        from errno import EEXIST
        from os import makedirs, path

        try:
            makedirs(mypath)
        except OSError as exc:
            if exc.errno == EEXIST and path.isdir(mypath):
                pass
            else:
                raise


class FieldAnalysisFromFieldmap(Tools):

    def __init__(self):
        """."""
        # fmap attributes
        self._fmaps = None
        self._fmaps_names = None
        self.config_idx = None
        self.data = dict()

        # Trajectory attributes
        self._idkickmap = None
        self.beam_energy = utils.BEAM_ENERGY  # [Gev]
        self.rk_s_step = utils.DEF_RK_S_STEP  # [mm]
        self.traj_init_rz = None  # [mm]
        self.traj_max_rz = None  # [mm]
        self.kmap_idlen = None  # [m]
        self.gridx = None
        self.gridy = None

        self.FOLDER_DATA = './results/measurements/data/'

    @property
    def idkickmap(self):
        """Return an object of IDKickMap class.

        Returns:
            IDKickMap object:
        """
        return self._idkickmap

    @property
    def fmaps(self):
        """Return a dictionary with all ID fieldmaps.

        Returns:
            Dictionary: A dictionary in which keys are
                the variable parameters and the values are the field maps.
                example: {(2, 40): fmap1, (2, 45): fmap2}, the keys could be
                gap and phase for example.
        """
        return self._fmaps

    @fmaps.setter
    def fmaps(self, fieldmaps):
        self._fmaps = fieldmaps

    @property
    def fmaps_names(self):
        """Return a dictionary with all ID fieldmaps names.

        Returns:
            Dictionary: A dictionary in which keys are
                the variable parameters and the values are the field maps.
                example: {(2, 40): fmap_name1, (2, 45): fmap_name2}, the keys could be gap and phase for example.
        """
        return self._fmaps_names

    @fmaps_names.setter
    def fmaps_names(self, fieldmap_names):
        self._fmaps_names = fieldmap_names

    @idkickmap.setter
    def idkickmap(self, fmap_fname):
        """Set idkickmap config for trajectory."""
        self._idkickmap = _IDKickMap()
        self._idkickmap.fmap_fname = fmap_fname
        self._idkickmap.beam_energy = self.beam_energy
        self._idkickmap.rk_s_step = self.rk_s_step
        self._idkickmap.kmap_idlen = self.kmap_idlen

    def _get_fmaps(self):
        config_dict = utils.CONFIG_DICT
        fmaps_ = {'params': None,
                  'configs': dict()}
        fmaps_names_ = {'params': None,
                        'configs': dict()}
        for config_key in config_dict['configs'].keys():
            fmap_fname = self.get_fmap_fname(config_keys=config_key,
                                             config_dict=config_dict)
            fmap = self.get_fmap(fmap_fname=fmap_fname)
            fmaps_['configs'][config_key] = fmap
            fmaps_names_['configs'][config_key] = fmap_fname
        fmaps_['params'] = config_dict['params']
        fmaps_names_['params'] = config_dict['params']
        self.fmaps = fmaps_
        self.fmaps_names = fmaps_names_

    def _generate_kickmap(self, fmap_fname, **kwargs):
        self.idkickmap = fmap_fname
        self.idkickmap.fmap_calc_kickmap(posx=self.gridx, posy=self.gridy)
        fname = self.get_kmap_filename(folder_data=utils.FOLDER_DATA,
                                       meas_flag=True, **kwargs)
        self.idkickmap.save_kickmap_file(kickmap_filename=fname)

    def run_generate_kickmap(self):

        kwargs = dict()
        self._get_fmaps()
        param_names = self.fmaps_names['params']
        for configs, value in self.fmaps_names['configs'].items():
            stg = 'calc kickmap for '
            for i, config in enumerate(configs):
                stg += param_names[i]
                stg += f' {config}, '
                kwargs[param_names[i]] = config
            print(stg)
            self._generate_kickmap(fmap_fname=value, **kwargs)

    def _get_field_roll_off(self):
        """Calculate the roll-off of a field component.

        Args:
            rt (numpy 1D array): array with positions where the field will be
                calculated
        """
        field_component = utils.FIELD_COMPONENT
        period = utils.ID_PERIOD
        roff_pos = utils.ROLL_OFF_POS
        plane = utils.ROLL_OFF_PLANE

        for config, fmap in self.fmaps['configs'].items():
            rt, b, rzmax, roff = self.get_fmap_roll_off(
                field_component=field_component,
                period=period, roff_pos=roff_pos,
                plane=plane, fmap=fmap)

            self.data[config] = dict()
            self.data[config].update({'rolloff_rt': rt})
            self.data[config].update({'rolloff_{}'.format(field_component): b})
            self.data[config].update({'rolloff_value': roff})

    def _get_field_on_axis(self):
        """Get the field along z axis."""

        print('Calculating field on axis...')
        for config, fmap in self.fmaps['configs'].items():
            print(config)
            bx = fmap.bx[fmap.ry_zero][fmap.rx_zero][:]
            by = fmap.by[fmap.ry_zero][fmap.rx_zero][:]
            bz = fmap.bz[fmap.ry_zero][fmap.rx_zero][:]
            rz = fmap.rz
            self.data[config].update({'onaxis_bx': bx})
            self.data[config].update({'onaxis_by': by})
            self.data[config].update({'onaxis_bz': bz})
            self.data[config].update({'onaxis_rz': rz})

    def _get_field_on_trajectory(self):
        print('Calculating field on trajectory...')
        for config, fmap_name in self.fmaps_names['configs'].items():
            # create IDKickMap and calc trajectory
            self.idkickmap = fmap_name
            self.idkickmap.fmap_calc_trajectory(
                traj_init_rx=0, traj_init_ry=0,
                traj_init_px=0, traj_init_py=0)
            traj = self.idkickmap.traj

            self.data[config].update({'ontraj_bx': traj.bx})
            self.data[config].update({'ontraj_by': traj.by})
            self.data[config].update({'ontraj_bz': traj.bz})
            self.data[config].update({'ontraj_rx': traj.rx})
            self.data[config].update({'ontraj_ry': traj.ry})
            self.data[config].update({'ontraj_rz': traj.rz})
            self.data[config].update({'ontraj_px': traj.px})
            self.data[config].update({'ontraj_py': traj.py})
            self.data[config].update({'ontraj_pz': traj.pz})
            self.data[config].update({'ontraj_s': traj.s})

    def _generate_field_data_fname(self, params, config):
        fpath = self.FOLDER_DATA
        fname = 'field_data'

        if 'phase' in params:
            idx = params.index('phase')
            phase_str = self.get_phase_str(config[idx])
            fpath += 'phases/phase_{}/'.format(phase_str)
            fname += '_phase{}'.format(phase_str)

        if 'gap' in params:
            idx = params.index('gap')
            gap_str = self.get_gap_str(config[idx])
            fpath += 'gap_{}/'.format(gap_str)
            fname += '_gap{}'.format(gap_str)

        forbidden_list = ['phase', 'gap']
        params_ = [elem for elem in params if elem not in forbidden_list]
        for param in params_:
            idx = params.index(param)
            value = config[idx]
            fpath += '{}/{}_{}/'.format(param + 's', param, value)
            fname += '_{}{}'.format(param, value)

        return fpath + fname

    def _save_field_data(self):
        #  re-organize data
        fdata = {'params': self.fmaps['params'],
                 'configs': self.data}

        params = fdata['params']
        for config, data in fdata['configs'].items():
            # get filename
            fname = self._generate_field_data_fname(params, config)
            print(fname)
            _save_pickle(data, fname, overwrite=True, makedirs=True)

    def run_calc_fields(self):

        self._get_fmaps()
        self._get_field_roll_off()
        self._get_field_on_axis()
        self._get_field_on_trajectory()

        self._save_field_data()

    def _get_field_data_fname(self, **kwargs):

        folder_data = self.FOLDER_DATA
        fpath = self.get_meas_data_path(folder_data=folder_data, **kwargs)

        fname = 'field_data'

        if 'phase' in kwargs.keys():
            phase_str = self.get_phase_str(kwargs.get('phase'))
            fname += '_phase{}'.format(phase_str)

        if 'gap' in kwargs.keys():
            gap_str = self.get_gap_str(kwargs.get('gap'))
            fname += '_gap{}'.format(gap_str)

        forbidden_list = ['phase', 'gap']
        items = list(kwargs.items())
        items_ = [elem for elem in items if elem[0] not in forbidden_list]
        for key, value in items_:
            fname += '_{}{}'.format(key, value)

        return fpath + fname

    def get_data_plot(self, **kwargs):
        data_plot = dict()
        fdata_dict = dict()
        var_param = [item for item in kwargs.items() if type(item[1]) == list]
        var_param = var_param[0]
        var_param_name = var_param[0]
        var_param_values = var_param[1]
        del kwargs[var_param_name]

        for var_param_value in var_param_values:
            kwargs[var_param_name] = var_param_value
            fname = self._get_field_data_fname(**kwargs)
            try:
                fdata_dict[var_param_value] = _load_pickle(fname)
            except FileNotFoundError:
                print('File does not exist.')
        data_plot[var_param_name] = fdata_dict

        return data_plot

    def _plot_field_on_axis(self, data, sulfix=None):
        colors = ['C0', 'b', 'g', 'y', 'C1', 'r', 'k']
        fig, axs = _plt.subplots(3, 1, sharex=True, figsize=(12, 8))
        output_dir = utils.FOLDER_DATA + 'general'
        output_dir = output_dir.replace('model', 'measurements')
        filename = output_dir + '/field-profile'
        if sulfix is not None:
            filename += sulfix
        self.mkdir_function(output_dir)
        var_parameter_name = list(data.keys())[0]
        for i, value in enumerate(data[var_parameter_name].keys()):
            label = var_parameter_name + ' {}'.format(value)
            bx = data[var_parameter_name][value]['onaxis_bx']
            by = data[var_parameter_name][value]['onaxis_by']
            bz = data[var_parameter_name][value]['onaxis_bz']
            rz = data[var_parameter_name][value]['onaxis_rz']
            axs[0].plot(rz, bx, label=label, color=colors[i])
            axs[0].set_ylabel('bx [T]')
            axs[1].plot(rz, by, label=label, color=colors[i])
            axs[1].set_ylabel('by [T]')
            axs[2].plot(rz, bz, label=label, color=colors[i])
            axs[2].set_ylabel('bz [T]')
        for j in _np.arange(3):
            axs[j].set_xlabel('z [mm]')
            axs[j].legend()
            axs[j].grid()
        _plt.savefig(filename, dpi=300)
        _plt.show()

    def _plot_rk_traj(self, data, sulfix=None):
        colors = ['C0', 'b', 'g', 'y', 'C1', 'r', 'k']
        output_dir = utils.FOLDER_DATA + 'general'
        output_dir = output_dir.replace('model', 'measurements')
        var_parameter_name = list(data.keys())[0]
        for i, value in enumerate(data[var_parameter_name].keys()):
            s = data[var_parameter_name][value]['ontraj_s']
            rx = data[var_parameter_name][value]['ontraj_rx']
            ry = data[var_parameter_name][value]['ontraj_ry']
            px = 1e6*data[var_parameter_name][value]['ontraj_px']
            py = 1e6*data[var_parameter_name][value]['ontraj_py']
            label = var_parameter_name + ' {}'.format(value)

            _plt.figure(1)
            _plt.plot(s, 1e3*rx, color=colors[i], label=label)
            _plt.xlabel('s [mm]')
            _plt.ylabel('x [um]')

            _plt.figure(2)
            _plt.plot(s, 1e3*ry, color=colors[i], label=label)
            _plt.xlabel('s [mm]')
            _plt.ylabel('y [um]')
            _plt.legend()

            _plt.figure(3)
            _plt.plot(s, px, color=colors[i], label=label)
            _plt.xlabel('s [mm]')
            _plt.ylabel('px [urad]')

            _plt.figure(4)
            _plt.plot(s, py, color=colors[i], label=label)
            _plt.xlabel('s [mm]')
            _plt.ylabel('py [urad]')
        sulfix_ = ['/traj-rx', '/traj-ry', '/traj-px', '/traj-py']
        for i in [1, 2, 3, 4]:
            _plt.figure(i)
            _plt.legend()
            _plt.grid()
            filename = output_dir + sulfix_[i-1]
            if sulfix is not None:
                filename += sulfix
            _plt.savefig(filename, dpi=300)
        _plt.show()

    def _read_data_roll_off(self, data, value):
        field_component = utils.FIELD_COMPONENT
        if 'rolloff_{}'.format(field_component) in data[value]:
            b = data[value]['rolloff_{}'.format(field_component)]
            if 'rolloff_rt' in data[value]:
                rt = data[value]['rolloff_rt']
            elif 'rolloff_rx' in data[value]:
                rt = data[value]['rolloff_rx']
            elif 'rolloff_ry' in data[value]:
                rt = data[value]['rolloff_ry']
            rtp_idx = _np.argmin(_np.abs(rt - utils.ROLL_OFF_POS))
            rt0_idx = _np.argmin(_np.abs(rt))
            roff = _np.abs(b[rtp_idx]/b[rt0_idx]-1)
            b0 = b[rt0_idx]
            roll_off = 100*(b/b0 - 1)
            return rt, b, roll_off, roff

    def _plot_field_roll_off(self, data, sulfix=None):
        field_component = utils.FIELD_COMPONENT
        _plt.figure(1)
        output_dir = utils.FOLDER_DATA + 'general'
        output_dir = output_dir.replace('model', 'measurements')
        filename = output_dir + '/field-rolloff'
        if sulfix is not None:
            filename += sulfix
        colors = ['C0', 'b', 'g', 'y', 'C1', 'r', 'k']
        var_parameter_name = list(data.keys())[0]
        for i, value in enumerate(data[var_parameter_name].keys()):
            roff_data = self._read_data_roll_off(
                data[var_parameter_name], value)
            if roff_data is None:
                continue
            else:
                rt, b, roll_off, roff = roff_data
            label = var_parameter_name +\
                " {} mm, roll-off = {:.2f} %".format(value, 100*roff)
            _plt.plot(rt, roll_off, '.-', label=label, color=colors[i])
        if field_component == 'by':
            _plt.xlabel('x [mm]')
        else:
            _plt.xlabel('y [mm]')
        _plt.ylabel('Field roll off [%]')
        _plt.xlim(-utils.ROLL_OFF_POS, utils.ROLL_OFF_POS)
        _plt.ylim(-101*roff, 20*roff)
        if field_component == 'by':
            _plt.title(
                'Field roll-off at x = {} mm'.format(utils.ROLL_OFF_POS))
        elif field_component == 'bx':
            _plt.title(
                'Field roll-off at y = {} mm'.format(utils.ROLL_OFF_POS))
        _plt.legend()
        _plt.grid()
        _plt.savefig(filename, dpi=300)
        _plt.show()

    def run_plot_data(self, sulfix=None, **kwargs):
        data_plot = self.get_data_plot(**kwargs)
        self._plot_field_on_axis(data=data_plot,
                                 sulfix=sulfix)
        self._plot_rk_traj(data=data_plot, sulfix=sulfix)
        self._plot_field_roll_off(data=data_plot, sulfix=sulfix)


class RadiaModelCalibration(Tools):

    def __init__(self, fmap, model):
        """."""
        self._fmap = fmap
        self._model = model
        self._rz_meas = None
        self._rx_meas = None
        self._ry_meas = None
        self._bx_meas = None
        self._by_meas = None
        self._byx_meas = None
        self._bz_meas = None
        self._df_bmeas = None
        self._rz_model = None
        self._rx_model = None
        self._ry_model = None
        self._bx_model = None
        self._byx_model = None
        self._by_model = None
        self._bz_model = None
        self._df_bmodel = None
        self._nrselblocks = 1
        self._neg_flag = 1

    def _set_rz_model(self, nr_pts_period=9):
        """."""
        model = self._model
        field_length = 1.1*(2 + model.period_length)*model.nr_periods
        field_length += 2*5*model.gap
        z_step = self._model.period_length / (nr_pts_period - 1)
        maxind = int(field_length / z_step)
        inds = _np.arange(-maxind, maxind + 1)
        rz = inds * z_step
        self._rz_model = rz
        self._by_model = None  # reset field model

    def _init_fields_rz(self, nr_pts_period=9):
        """Initialize model and measurement field arrays"""
        self._set_rz_model(nr_pts_period=nr_pts_period)
        field = self._model.get_field(0, 0, self._rz_model)  # On-axis
        self._bz_model = self._neg_flag*field[:, 2]
        self._by_model = self._neg_flag*field[:, 1]
        self._bx_model = self._neg_flag*field[:, 0]
        fmap = self._fmap
        self._rz_meas = fmap.rz
        self._bx_meas = fmap.bx[fmap.ry_zero][fmap.rx_zero][:]
        self._by_meas = fmap.by[fmap.ry_zero][fmap.rx_zero][:]
        self._bz_meas = fmap.bz[fmap.ry_zero][fmap.rx_zero][:]

    def _init_fields_rt(self, rx, b):
        self._rx_meas = rx
        self._rx_model = rx
        self._byx_meas = b
        b_model, _, rzmax = self.calc_radia_roll_off(
            field_component=utils.FIELD_COMPONENT, plane=utils.ROLLOFF_PLANE,
            rt=rx, roff_pos=utils.ROLL_OFF_POS, model=self._model)
        self._byx_model = b_model
        self._df_bmeas = _np.diff(self._byx_meas)/_np.diff(self._rx_meas)
        self._df_bmodel = _np.diff(self._byx_model)/_np.diff(self._rx_model)
        res = _np.sum((self._df_bmeas-self._df_bmodel)**2)/len(self._df_bmodel)
        return res

    def _plot_fields_rz(self):
        fig, axs = _plt.subplots(2, sharex=True)
        by_meas_fit = _np.interp(
            self._rz_model, self._rz_meas, self._by_meas)
        bx_meas_fit = _np.interp(
            self._rz_model, self._rz_meas, self._bx_meas)
        b_model = _np.concatenate((self._by_model, self._bx_model))
        b_meas = _np.concatenate((by_meas_fit, bx_meas_fit))
        rms = 100*_np.std((b_model-b_meas)/b_meas)/len(b_meas)
        axs[0].plot(self._rz_meas, self._by_meas, label='Measurements')
        axs[0].plot(self._rz_model, self._by_model, label='Model')
        axs[1].plot(self._rz_meas, self._bx_meas, label='Measurements')
        axs[1].plot(
            self._rz_model, self._bx_model,
            label='Model - r.m.s = {:.2f} %'.format(rms))
        axs[0].set(ylabel='By [T]')
        axs[1].set(xlabel='rz [mm]', ylabel='Bx [T]')
        axs[0].grid()
        axs[1].grid()
        _plt.xlim(-1600, 1600)
        _plt.legend(loc='upper right')
        _plt.show()

    def _plot_fields_rt(self):
        fig, axs = _plt.subplots(2, sharex=True)
        rx = self._rx_meas[:-1]
        axs[0].plot(self._rx_meas, self._byx_meas, label='Meas.')
        axs[0].plot(self._rx_model, self._byx_model, label='Model')
        axs[1].plot(
            rx, self._df_bmeas, label='Meas')
        axs[1].plot(rx, self._df_bmodel, label='Model')
        axs[0].set(ylabel='By [T]')
        axs[1].set(xlabel='rx [mm]', ylabel='Field gradient [T/mm]')
        axs[0].grid()
        axs[1].grid()
        axs[0].legend()
        axs[1].legend()
        rms = 100*_np.std(
            (self._df_bmodel-self._df_bmeas)/self._df_bmeas)/len(
                self._df_bmeas)
        # _plt.plot(rx, 1e4*self._df_bmeas, label='Measurements')
        # _plt.plot(
        #     rx, 1e4*self._df_bmodel, label='Model - r.m.s = {:.2f} %'.format(
        #         rms))
        # _plt.xlabel('rx [mm]')
        # _plt.ylabel('By gradient [G/mm]')
        # _plt.legend(loc='upper right')
        # _plt.grid()
        _plt.show()

    def _shiftscale_calc_residue(self, shift):
        """Calculate correction scale and residue for a given shift in data."""
        if self._by_model is None and self._bx_model is None:
            raise ValueError('Field model not yet initialized!')
        by_meas_fit = _np.interp(
            self._rz_model, self._rz_meas + shift, self._by_meas)
        bx_meas_fit = _np.interp(
            self._rz_model, self._rz_meas + shift, self._bx_meas)
        bf1 = _np.concatenate((by_meas_fit, bx_meas_fit))
        bf2 = _np.concatenate((self._by_model, self._bx_model))
        scale = _np.dot(bf1, bf2) / _np.dot(bf2, bf2)
        residue = _np.sum((bf1 - scale * bf2)**2)/len(self._rz_model)
        return residue, scale, by_meas_fit, bx_meas_fit

    def _shiftscale_plot_fields(self, shift):
        results = self._shiftscale_calc_residue(shift)
        residue, scale, by_meas_fit, bx_meas_fit = results
        fig, axs = _plt.subplots(2, sharex=True)
        axs[0].plot(self._rz_model, by_meas_fit, label='meas.')
        axs[0].plot(self._rz_model, scale * self._by_model, label='model')
        axs[1].plot(self._rz_model, bx_meas_fit, label='meas.')
        axs[1].plot(self._rz_model, scale * self._bx_model, label='model')
        sfmt = 'shift: {:.4f} mm, scale: {:.4f}, residue: {:.4f} T'
        fig.suptitle(sfmt.format(shift, scale, residue))
        axs[0].set(ylabel='By [T]')
        axs[1].set(xlabel='rz [mm]', ylabel='Bx [T]')
        _plt.xlim(-1600, 1600)
        _plt.legend()
        _plt.show()

    def _shiftscale_mag_set(self, scale):
        """Incorporate fitted scale as effective remanent magnetization."""
        for cas in self._model.cassettes_ref.values():
            mags_old = _np.array(cas.magnetization_list)
            mags_new = (scale*mags_old).tolist()
            cas.create_radia_object(magnetization_list=mags_new)
        mag_dict = self._model.magnetization_dict
        self._model.create_radia_object(magnetization_dict=mag_dict)

    def _model_set_config(self, width, phase, gap, roff_delta, mr_scale):
        """Incorporate fitted scale as effective remanent magnetization
           in the model construction moment."""
        self._model = utils.generate_radia_model(
            width=width, phase=phase, gap=gap,
            nr_periods=utils.NR_PERIODS_REAL_ID,
            solve=False, roff_calibration=roff_delta,
            mr_scale=mr_scale)


class FieldAnalysisFromRadia(Tools):

    def __init__(self):
        # Model attributes
        self.rt_field_max = None  # [mm]
        self.rt_field_nrpts = None
        self.rz_field_max = None  # [mm]
        self.rz_field_nrpts = None
        self.roll_off_rt = utils.ROLL_OFF_RT  # [mm]
        self.data = dict()
        self._models = dict()

        # Trajectory attributes
        self._idkickmap = None
        self.beam_energy = utils.BEAM_ENERGY  # [Gev]
        self.rk_s_step = utils.DEF_RK_S_STEP  # [mm]
        self.traj_init_rz = None  # [mm]
        self.traj_max_rz = None  # [mm]
        self.kmap_idlen = None  # [m]
        self.gridx = None
        self.gridy = None

        # fmap attributes
        self.config_idx = None
        self.roff_deltas = _np.linspace(1, 1, 1)

        self.FOLDER_DATA = './results/model/data/'

    @property
    def idkickmap(self):
        """Return an object of IDKickMap class.

        Returns:
            IDKickMap object:
        """
        return self._idkickmap

    @property
    def models(self):
        """Return a dictionary with all ID models.

        Returns:
            Dictionary: A dictionary in which keys are
                the variable parameters and the values are the models.
                example: {(2, 40): model1, (2, 45): model2}, the keys could be
                gap and width for example.
        """
        return self._models

    @models.setter
    def models(self, ids):
        """Set models attribute.

        Args:
            models (dictionary): A dictionary in which keys are
                the variable parameters and the values are the models.
                example: {(2, 40): model1, (2, 45): model2}, the keys could be
                gap and width for example.
        """
        self._models = ids

    @idkickmap.setter
    def idkickmap(self, id):
        """Set idkickmap config for trajectory."""
        self._idkickmap = _IDKickMap()
        self._idkickmap.radia_model = id
        self._idkickmap.beam_energy = self.beam_energy
        self._idkickmap.rk_s_step = self.rk_s_step
        self._idkickmap.traj_init_rz = self.traj_init_rz
        self._idkickmap.traj_rk_min_rz = self.traj_max_rz
        self._idkickmap.kmap_idlen = self.kmap_idlen

    def _generate_kickmap(self, key, id):
        width = key[0][1]
        phase = key[1][1]
        gap = key[2][1]
        self.idkickmap = id
        self.idkickmap.fmap_calc_kickmap(posx=self.gridx, posy=self.gridy)
        fname = self.get_kmap_filename(
            folder_data=utils.FOLDER_DATA, width=width, phase=phase, gap=gap)
        self.idkickmap.save_kickmap_file(kickmap_filename=fname)

    def _create_models(self):
        models_ = dict()
        for width in utils.widths:
            for phase in utils.phases:
                for gap in utils.gaps:
                    print(
                        f'creating model for gap {gap} mm, phase {phase} mm' +
                        f' and width {width} mm')
                    id = utils.generate_radia_model(
                        width=width,
                        phase=phase,
                        gap=gap,
                        solve=utils.SOLVE_FLAG)
                    key = (('width', width), ('phase', phase), ('gap', gap))
                    models_[(key)] = id
        self.models = models_

    def _get_field_roll_off(self, rt, peak_idx=0):
        """Calculate the roll-off of a field component.

        Args:
            rt (numpy 1D array): array with positions where the field will be
                calculated
            peak_idx (int): Peak index where the roll-off will be calculated.
                Defaults to 0.
        """
        b_ = dict()
        rt_dict = dict()
        roll_off = dict()
        for var_params, id in self.models.items():
            b, roff, _ = self.calc_radia_roll_off(
                field_component=utils.FIELD_COMPONENT,
                plane=utils.ROLL_OFF_PLANE,
                roff_pos=utils.ROLL_OFF_POS,
                model=id, rt=rt, peak_idx=peak_idx)
            b_[var_params] = b
            rt_dict[var_params] = rt
            roll_off[var_params] = roff
        self.data['rolloff_rt'] = rt_dict
        self.data['rolloff_{}'.format(utils.FIELD_COMPONENT)] = b_
        self.data['rolloff_value'] = roll_off

    def _get_field_on_axis(self, rz):
        """Get the field along z axis.

        Args:
            rz (numpy 1D array): array with longitudinal positions where the
                field will be calculated
        """
        bx_dict, by_dict, bz_dict, rz_dict = dict(), dict(), dict(), dict()
        for var_params, id in self.models.items():
            field = id.get_field(0, 0, rz)
            bx_dict[var_params] = field[:, 0]
            by_dict[var_params] = field[:, 1]
            bz_dict[var_params] = field[:, 2]
            rz_dict[var_params] = rz
        self.data['onaxis_bx'] = bx_dict
        self.data['onaxis_by'] = by_dict
        self.data['onaxis_bz'] = bz_dict
        self.data['onaxis_rz'] = rz_dict

    def _get_field_on_trajectory(self):
        bx, by, bz = dict(), dict(), dict()
        rz, rx, ry = dict(), dict(), dict()
        px, py, pz = dict(), dict(), dict()
        s = dict()
        traj_ = dict()
        for var_params, id in self.models.items():
            # create IDKickMap and calc trajectory
            self.idkickmap = id
            self.idkickmap.fmap_calc_trajectory(
                traj_init_rx=0, traj_init_ry=0,
                traj_init_px=0, traj_init_py=0)
            traj = self.idkickmap.traj
            bx[var_params], by[var_params], bz[var_params] =\
                traj.bx, traj.by, traj.bz
            rx[var_params], ry[var_params], rz[var_params] =\
                traj.rx, traj.ry, traj.rz
            px[var_params], py[var_params], pz[var_params] =\
                traj.px, traj.py, traj.pz
            s[var_params] = traj.s
            traj_[var_params] = traj

        self.data['ontraj_bx'] = bx
        self.data['ontraj_by'] = by
        self.data['ontraj_bz'] = bz

        self.data['ontraj_rx'] = rx
        self.data['ontraj_ry'] = ry
        self.data['ontraj_rz'] = rz

        self.data['ontraj_px'] = px
        self.data['ontraj_py'] = py
        self.data['ontraj_pz'] = pz

        self.data['ontraj_s'] = s
        self.data['traj'] = traj_

    def _generate_field_data_fname(self, keys):
        fname = self.FOLDER_DATA
        sulfix = 'field_data'
        for parameter in keys:
            if parameter[0] == 'width':
                width = parameter[1]
                fname = fname.replace(
                    'data/', 'data/widths/width_{}/'.format(width))
                sulfix += '_width{}'.format(width)

            if parameter[0] == 'phase':
                phase_str = self.get_phase_str(parameter[1])
                fname += 'phases/phase_{}/'.format(phase_str)
                sulfix += '_phase{}'.format(phase_str)

            if parameter[0] == 'gap':
                gap_str = self.get_gap_str(parameter[1])
                fname += 'gap_{}/'.format(gap_str)
                sulfix += '_gap{}'.format(gap_str)
        return fname + sulfix

    def _get_field_data_fname(self, width, phase, gap):
        fpath = self.get_data_path(folder_data=utils.FOLDER_DATA,
                                   width=width, phase=phase, gap=gap)

        sulfix = 'field_data'
        sulfix += '_width{}'.format(width)

        phase_str = self.get_phase_str(phase)
        sulfix += '_phase{}'.format(phase_str)

        gap_str = self.get_gap_str(gap)
        sulfix += '_gap{}'.format(gap_str)
        return fpath + sulfix

    def _save_field_data(self):
        #  re-organize data
        for keys in list(self.models.keys()):
            fdata = dict()
            for info, value in self.data.items():
                fdata[info] = value[keys]
            # get filename
            fname = self._generate_field_data_fname(keys=keys)
            print(fname)
            _save_pickle(fdata, fname, overwrite=True, makedirs=True)

    def run_calc_fields(self):
        if not self.models:
            self._create_models()

        rt = _np.linspace(
            -self.rt_field_max, self.rt_field_max, self.rt_field_nrpts)
        rz = _np.linspace(
            -self.rz_field_max, self.rz_field_max, self.rz_field_nrpts)

        self._get_field_roll_off(rt=rt)
        self._get_field_on_axis(rz=rz)
        self._get_field_on_trajectory()

        self._save_field_data()

    def get_data_plot(self, width=0, phase=0, gap=0):
        data_plot = dict()
        if utils.var_param == 'gap':
            for gap_ in utils.gaps:
                fname = self._get_field_data_fname(width, phase, gap_)
                try:
                    fdata = _load_pickle(fname)
                    data_plot[gap_] = fdata
                except FileNotFoundError:
                    print('File does not exist.')
        if utils.var_param == 'phase':
            for phase_ in utils.phases:
                fname = self._get_field_data_fname(width, phase_, gap)
                try:
                    fdata = _load_pickle(fname)
                    data_plot[phase_] = fdata
                except FileNotFoundError:
                    print('File does not exist.')
        if utils.var_param == 'width':
            for width_ in utils.widths:
                fname = self._get_field_data_fname(width_, phase, gap)
                try:
                    fdata = _load_pickle(fname)
                    data_plot[width_] = fdata
                except FileNotFoundError:
                    print('File does not exist.')
        return data_plot

    def run_plot_data(self, width=0, phase=0, gap=0, sulfix=None):
        data_plot = self.get_data_plot(width=width, phase=phase, gap=gap)
        self._plot_field_on_axis(data=data_plot, sulfix=sulfix)
        self._plot_rk_traj(data=data_plot, sulfix=sulfix)
        self._plot_field_roll_off(data=data_plot, sulfix=sulfix)

    def run_generate_kickmap(self):
        if self.models:
            print("models ready")
        else:
            self._create_models()
            print("models ready")

        for key, id in self.models.items():
            print(f'calc kickmap for gap {key[2][1]} mm, ' +
                  f'phase {key[1][1]} mm ' +
                  f'and width {key[0][1]} mm')
            self._generate_kickmap(key, id)

    def _plot_field_on_axis(self, data, sulfix=None):
        colors = ['b', 'g', 'y', 'C1', 'r', 'k']
        field_component = utils.FIELD_COMPONENT
        fig, axs = _plt.subplots(3, 1, sharex=True, figsize=(12, 8))
        output_dir = utils.FOLDER_DATA + 'general'
        filename = output_dir + '/field-profile'
        if sulfix is not None:
            filename += sulfix
        self.mkdir_function(output_dir)
        var_parameters = list(data.keys())
        for i, parameter in enumerate(var_parameters):
            label = utils.var_param + ' {}'.format(parameter)
            bx = data[parameter]['onaxis_bx']
            by = data[parameter]['onaxis_by']
            bz = data[parameter]['onaxis_bz']
            rz = data[parameter]['onaxis_rz']
            axs[0].plot(rz, bx, label=label, color=colors[i])
            axs[0].set_ylabel('bx [T]')
            axs[1].plot(rz, by, label=label, color=colors[i])
            axs[1].set_ylabel('by [T]')
            axs[2].plot(rz, bz, label=label, color=colors[i])
            axs[2].set_ylabel('bz [T]')
            for j in _np.arange(3):
                axs[j].set_xlabel('z [mm]')
                axs[j].legend()
                axs[j].grid()
        _plt.savefig(filename, dpi=300)
        _plt.show()

    def _plot_rk_traj(self, data, sulfix=None):
        colors = ['b', 'g', 'y', 'C1', 'r', 'k']
        var_parameters = list(data.keys())
        output_dir = utils.FOLDER_DATA + 'general'
        for i, parameter in enumerate(var_parameters):
            s = data[parameter]['ontraj_s']
            rx = data[parameter]['ontraj_rx']
            ry = data[parameter]['ontraj_ry']
            px = 1e6*data[parameter]['ontraj_px']
            py = 1e6*data[parameter]['ontraj_py']
            label = utils.var_param + ' {}'.format(parameter)

            _plt.figure(1)
            _plt.plot(s, 1e3*rx, color=colors[i], label=label)
            _plt.xlabel('s [mm]')
            _plt.ylabel('x [um]')

            _plt.figure(2)
            _plt.plot(s, 1e3*ry, color=colors[i], label=label)
            _plt.xlabel('s [mm]')
            _plt.ylabel('y [um]')
            _plt.legend()

            _plt.figure(3)
            _plt.plot(s, px, color=colors[i], label=label)
            _plt.xlabel('s [mm]')
            _plt.ylabel('px [urad]')

            _plt.figure(4)
            _plt.plot(s, py, color=colors[i], label=label)
            _plt.xlabel('s [mm]')
            _plt.ylabel('py [urad]')
        sulfix_ = ['/traj-rx', '/traj-ry', '/traj-px', '/traj-py']
        for i in [1, 2, 3, 4]:
            _plt.figure(i)
            _plt.legend()
            _plt.grid()
            filename = output_dir + sulfix_[i-1]
            if sulfix is not None:
                filename += sulfix
            _plt.savefig(filename, dpi=300)
        _plt.show()

    def _read_data_roll_off(self, data, parameter):
        field_component = utils.FIELD_COMPONENT
        if 'rolloff_{}'.format(field_component) in data[parameter]:
            b = data[parameter]['rolloff_{}'.format(field_component)]
            if 'rolloff_rt' in data[parameter]:
                rt = data[parameter]['rolloff_rt']
            elif 'rolloff_rx' in data[parameter]:
                rt = data[parameter]['rolloff_rx']
            elif 'rolloff_ry' in data[parameter]:
                rt = data[parameter]['rolloff_ry']
            rtp_idx = _np.argmin(_np.abs(rt - utils.ROLL_OFF_RT))
            rt0_idx = _np.argmin(_np.abs(rt))
            roff = b[rtp_idx]/b[rt0_idx]-1
            b0 = b[rt0_idx]
            roll_off = 100*(b/b0 - 1)
            return rt, b, roll_off, roff

        return

    def _plot_field_roll_off(self, data, sulfix=None):
        field_component = utils.FIELD_COMPONENT
        _plt.figure(1)
        output_dir = utils.FOLDER_DATA + 'general'
        filename = output_dir + '/field-rolloff'
        if sulfix is not None:
            filename += sulfix
        colors = ['b', 'g', 'y', 'C1', 'r', 'k']
        var_parameters = list(data.keys())
        for i, parameter in enumerate(var_parameters):
            roff_data = self._read_data_roll_off(data, parameter)
            if roff_data is None:
                continue
            else:
                rt, b, roll_off, roff = roff_data
            label = utils.var_param +\
                " {} mm, roll-off = {:.2f} %".format(parameter, 100*roff)
            _plt.plot(rt, roll_off, '.-', label=label, color=colors[i])
        if field_component == 'by':
            _plt.xlabel('x [mm]')
        else:
            _plt.xlabel('y [mm]')
        _plt.ylabel('Field roll off [%]')
        _plt.xlim(-utils.ROLL_OFF_RT, utils.ROLL_OFF_RT)
        if roff >= 0:
            _plt.ylim(-0.1, 100*roff)
        else:
            _plt.ylim(100*roff, 0.1)
        if field_component == 'by':
            _plt.title('Field roll-off at x = {} mm'.format(utils.ROLL_OFF_RT))
        elif field_component == 'bx':
            _plt.title('Field roll-off at y = {} mm'.format(utils.ROLL_OFF_RT))
        _plt.legend()
        _plt.grid()
        _plt.savefig(filename, dpi=300)
        _plt.show()

    def _get_planar_id_features(self,
                                width=0, phase=0, gap=0,
                                plot_flag=False):

        def cos(x, b0, kx):
            b = b0*_np.cos(kx*x)
            return b

        data = self.get_data_plot(width=width, phase=phase, gap=gap)
        if utils.var_param == 'width':
            rt, b, *_ = self._read_data_roll_off(data, width)
        elif utils.var_param == 'phase':
            rt, b, *_ = self._read_data_roll_off(data, phase)
        elif utils.var_param == 'gap':
            rt, b, *_ = self._read_data_roll_off(data, gap)
        idxi = _np.argmin(_np.abs(rt+1))
        idxf = _np.argmin(_np.abs(rt-1))
        rt = rt[idxi:idxf]
        b = b[idxi:idxf]
        opt = _curve_fit(cos, rt, b)[0]
        k = opt[1]
        b0 = opt[0]
        print('field amplitude = {:.3f} T'.format(b0))
        if plot_flag:
            _plt.plot(rt, b, label='model')
            b_fitted = cos(rt, opt[0], opt[1])
            _plt.plot(rt, b_fitted, label='fitting')
            if utils.FIELD_COMPONENT == 'by':
                _plt.xlabel('x [mm]')
            elif utils.FIELD_COMPONENT == 'bx':
                _plt.xlabel('y [mm]')
            _plt.ylabel('B [T]')
            _plt.title('Field Roll-off fitting')
            _plt.grid()
            _plt.legend()
            _plt.show()

        kz = 2*_np.pi/(utils.ID_PERIOD*1e-3)
        kx = k*1e3
        ky = _np.sqrt(kx**2+kz**2)
        if utils.FIELD_COMPONENT == 'by':
            print('kx = {:.2f}'.format(kx))
            print('ky = {:.2f}'.format(ky))
        elif utils.FIELD_COMPONENT == 'bx':
            print('kx = {:.2f}'.format(ky))
            print('ky = {:.2f}'.format(kx))
        print('kz = {:.2f}'.format(kz))

        return b0, kx, ky, kz

    def _get_id_quad_coef(self,
                          width=0, phase=0, gap=0,
                          plot_flag=False):
        id_len = utils.SIMODEL_ID_LEN
        b0, kx, ky, kz = self._get_planar_id_features(width=width,
                                                      phase=0, gap=gap,
                                                      plot_flag=plot_flag)
        beam = Beam(energy=3)
        brho = beam.brho
        phase = phase*1e-3
        factor = 1 + _np.cos(kz*phase)
        factor -= (kx**2)/(kz**2+kx**2)*(1-_np.cos(kz*phase))
        a = (1/(kz**2))*id_len*(b0**2)/(4*brho**2)
        if utils.FIELD_COMPONENT == 'by':
            coefx = a*factor*kx**2
            coefy = -a*2*ky**2
        elif utils.FIELD_COMPONENT == 'bx':
            coefx = -a*factor*ky**2
            coefy = a*2*kx**2

        return coefx, coefy

    def get_id_estimated_focusing(self, betax, betay,
                                  width=0, phase=0, gap=0, plot_flag=False):

        coefx, coefy = self._get_id_quad_coef(width=width, phase=phase,
                                              gap=gap, plot_flag=plot_flag)
        dtunex = -coefx*betax/(4*_np.pi)
        dtuney = -coefy*betay/(4*_np.pi)

        print('horizontal quadrupolar term KL: {:.5f}'.format(coefx))
        print('vertical quadrupolar term KL: {:.5f}'.format(coefy))
        print('horizontal delta tune: {:.5f}'.format(dtunex))
        print('vertical delta tune: {:.5f}'.format(dtuney))

        return coefx, coefy, dtunex, dtuney

    def generate_linear_kickmap(self, width=0, phase=0, gap=0,
                                cxy=0, cyx=0):
        beam = Beam(energy=3)
        brho = beam.brho
        idkickmap = _IDKickMap()
        cxx, cyy = self._get_id_quad_coef(width=width,
                                          phase=phase, gap=gap)

        idkickmap.generate_linear_kickmap(brho=brho, posx=self.gridx,
                                          posy=self.gridy, cxx=cxx, cyy=cyy,
                                          cxy=cxy, cyx=cyx)
        fname = self.get_kmap_filename(
            folder_data=utils.FOLDER_DATA, width=width, phase=phase, gap=gap)
        fname = fname.replace('.txt', '-linear.txt')
        idkickmap.kmap_idlen = utils.ID_KMAP_LEN
        idkickmap.save_kickmap_file(kickmap_filename=fname)

    def calibrate_models(self, plot_flag=False):
        field_component = utils.FIELD_COMPONENT
        period = utils.ID_PERIOD
        roff_pos = utils.ROLL_OFF_POS
        plane = utils.ROLL_OFF_PLANE
        config_dict = utils.CONFIG_DICT

        models_ = dict()
        for width in utils.widths:
            for phase in utils.phases:
                for gap in utils.gaps:
                    print(
                        f'calibrating model for gap {gap} mm, phase {phase}' +
                        f' mm and width {width} mm')

                    fmap_fname = self.get_fmap_fname(config_keys=(phase, gap),
                                                     config_dict=config_dict)
                    fmap = self.get_fmap(fmap_fname=fmap_fname)
                    rx, byx, rzmax, roff = self.get_fmap_roll_off(
                        field_component=field_component,
                        period=period, roff_pos=roff_pos,
                        plane=plane, fmap=fmap)
                    model = utils.generate_radia_model(
                        width=width, phase=phase, gap=gap,
                        nr_periods=utils.NR_PERIODS_REAL_ID,
                        solve=False)
                    cm = RadiaModelCalibration(fmap, model)
                    cm._init_fields_rz()

                    # search for best shift and calc scale
                    shifts = _np.linspace(-0.25, 0.25, 31)*model.period_length
                    minshift, minscale, minresidue = shifts[0], 1, float('inf')
                    for shift in shifts:
                        residue, scale, *_ = cm._shiftscale_calc_residue(
                            shift=shift)
                        if residue < minresidue:
                            minshift, minscale = shift, scale
                            minresidue = residue
                    if minscale < 0:
                        mr_scale = -1*minscale
                        cm._neg_flag = -1
                    else:
                        mr_scale = minscale
                        cm._neg_flag = 1

                    # search for best delta in widths
                    residue = cm._init_fields_rt(rx, byx)
                    mindelta, minresidue = self.roff_deltas[0], float('inf')
                    for delta in self.roff_deltas:
                        cm._model_set_config(width=width, phase=phase,
                                             gap=gap, roff_delta=delta,
                                             mr_scale=mr_scale)

                        cm._init_fields_rt(rx, byx)
                        residue = _np.sum(
                            (cm._df_bmodel-cm._df_bmeas)**2)/len(cm._df_bmodel)
                        if residue < minresidue:
                            mindelta = delta
                            minresidue = residue
                    cm._model_set_config(width=width, phase=phase,
                                         gap=gap, roff_delta=mindelta,
                                         mr_scale=mr_scale*1.05)

                    cm._init_fields_rz()

                    if plot_flag:
                        cm._plot_fields_rz()
                    residue = cm._init_fields_rt(rx, byx)
                    if plot_flag:
                        cm._plot_fields_rt()

                    key = (('width', width), ('phase', phase), ('gap', gap))
                    models_[(key)] = cm._model
        self.models = models_


class AnalysisKickmap(Tools):

    def __init__(self):
        self._idkickmap = None
        self.save_flag = False
        self.plot_flag = False
        self.filter_flag = False
        self.shift_flag = False
        self.linear = False
        self.meas_flag = False
        self.FOLDER_DATA = utils.FOLDER_DATA

    def _get_figname_plane(self, kick_plane, var='X',
                           width=None, phase=None, gap=None):
        fpath = utils.FOLDER_DATA
        if self.meas_flag:
            fpath = fpath.replace('model/', 'measurements/')
        fpath = fpath.replace('data/', 'data/general/')

        if utils.var_param == 'gap':
            phase_str = self.get_phase_str(phase)
            fname = fpath + 'kick{}-vs-{}-width{}mm-phase{}.png'.format(
                kick_plane, var.lower(), width, phase_str)

        if utils.var_param == 'phase':
            gap_str = self.get_gap_str(gap)
            fname = fpath + 'kick{}-vs-{}-width{}mm-gap{}.png'.format(
                kick_plane, var.lower(), width, gap_str)

        if utils.var_param == 'width':
            phase_str = self.get_phase_str(phase)
            gap_str = self.get_gap_str(gap)
            fname = fpath + 'kick{}-vs-{}-phase{}-gap{}.png'.format(
                kick_plane, var.lower(), phase_str, gap_str)
        if self.linear:
            fname = fname.replace('.png', '-linear.png')

        return fname

    def _get_figname_allplanes(self, kick_plane, var='X',
                               width=None, phase=None, gap=None):
        fname = utils.FOLDER_DATA
        if self.meas_flag:
            fname = fname.replace('model/', 'measurements/')
        sulfix = 'kick{}-vs-{}-all-planes'.format(
            kick_plane.lower(), var.lower())

        fname = fname.replace(
            'data/', 'data/widths/width_{}/'.format(width))

        phase_str = self.get_phase_str(phase)
        fname += 'phases/phase_{}/'.format(phase_str)

        gap_str = self.get_gap_str(gap)
        fname += 'gap_{}/'.format(gap_str)

        fname += sulfix
        if self.linear:
            fname += '-linear'

        return fname

    def run_shift_kickmap(self, widths, phases, gaps):
        for width in widths:
            for phase in phases:
                for gap in gaps:
                    fname = self.get_kmap_filename(
                        folder_data=utils.FOLDER_DATA, width=width,
                        phase=phase, gap=gap,
                        meas_flag=self.meas_flag)
                    self._idkickmap = _IDKickMap(
                        kmap_fname=fname, shift_on_axis=True)
                    fname = fname.replace('.txt', '-shifted_on_axis.txt')
                    self._idkickmap.save_kickmap_file(fname)

    def run_filter_kickmap(self, widths, phases, gaps,
                           rx, ry, filter_order=5, is_shifted=True):
        for width in widths:
            for phase in phases:
                for gap in gaps:
                    fname = self.get_kmap_filename(
                        folder_data=utils.FOLDER_DATA, width=width,
                        phase=phase, gap=gap,
                        shift_flag=is_shifted, meas_flag=self.meas_flag)
                    self._idkickmap = _IDKickMap(fname)
                    self._idkickmap.filter_kmap(
                        posx=rx, posy=ry, order=filter_order, plot_flag=True)
                    self._idkickmap.kmap_idlen = utils.ID_KMAP_LEN
                    fname = fname.replace('.txt', '-filtered.txt')
                    self._idkickmap.save_kickmap_file(fname)

    def _calc_idkmap_kicks(self, plane_idx=0, var='X'):
        beam = Beam(energy=3)
        brho = beam.brho
        rx0 = self._idkickmap.posx
        ry0 = self._idkickmap.posy
        fposx = self._idkickmap.fposx
        fposy = self._idkickmap.fposy
        kickx = self._idkickmap.kickx
        kicky = self._idkickmap.kicky
        if var.lower() == 'x':
            rxf = fposx[plane_idx, :]
            ryf = fposy[plane_idx, :]
            pxf = kickx[plane_idx, :]/brho**2
            pyf = kicky[plane_idx, :]/brho**2
        elif var.lower() == 'y':
            rxf = fposx[:, plane_idx]
            ryf = fposy[:, plane_idx]
            pxf = kickx[:, plane_idx]/brho**2
            pyf = kicky[:, plane_idx]/brho**2

        return rx0, ry0, pxf, pyf, rxf, ryf

    def check_kick_at_plane(
            self, width=None, phase=None, gap=None,
            planes=['X', 'Y'], kick_planes=['X', 'Y']):

        if utils.var_param == 'width':
            var_params = utils.widths
        if utils.var_param == 'phase':
            var_params = utils.phases
        if utils.var_param == 'gap':
            var_params = utils.gaps

        for var in planes:
            for kick_plane in kick_planes:
                colors = ['b', 'g', 'y', 'C1', 'r', 'k']
                for var_param, color in zip(var_params, colors):

                    if utils.var_param == 'width':
                        fname = self.get_kmap_filename(
                            folder_data=utils.FOLDER_DATA, width=var_param,
                            gap=gap, phase=phase,
                            shift_flag=self.shift_flag,
                            filter_flag=self.filter_flag,
                            linear=self.linear,
                            meas_flag=self.meas_flag)
                        fname_fig = self._get_figname_plane(
                            kick_plane=kick_plane, var=var,
                            width=var_param, gap=gap, phase=phase)
                    if utils.var_param == 'phase':
                        fname = self.get_kmap_filename(
                            folder_data=utils.FOLDER_DATA, width=width,
                            gap=gap, phase=var_param,
                            shift_flag=self.shift_flag,
                            filter_flag=self.filter_flag,
                            linear=self.linear,
                            meas_flag=self.meas_flag)
                        fname_fig = self._get_figname_plane(
                            kick_plane=kick_plane, var=var,
                            width=width, gap=gap, phase=var_param)
                    if utils.var_param == 'gap':
                        fname = self.get_kmap_filename(
                            folder_data=utils.FOLDER_DATA, width=width,
                            phase=phase, gap=var_param,
                            shift_flag=self.shift_flag,
                            filter_flag=self.filter_flag,
                            linear=self.linear,
                            meas_flag=self.meas_flag)
                        fname_fig = self._get_figname_plane(
                            kick_plane=kick_plane, var=var,
                            width=width, gap=var_param, phase=phase)

                    self._idkickmap = _IDKickMap(fname)
                    if var.lower() == 'x':
                        pos_zero_idx = list(self._idkickmap.posy).index(0)
                    elif var.lower() == 'y':
                        pos_zero_idx = list(self._idkickmap.posx).index(0)
                    rx0, ry0, pxf, pyf, *_ = self._calc_idkmap_kicks(
                                                    plane_idx=pos_zero_idx,
                                                    var=var)
                    if not self.linear:
                        pxf *= utils.RESCALE_KICKS
                        pyf *= utils.RESCALE_KICKS

                    pf, klabel =\
                        (pxf, 'px') if kick_plane.lower() == 'x' else (
                         pyf, 'py')

                    if var.lower() == 'x':
                        r0, xlabel, rvar = (rx0, 'x0 [mm]', 'y')
                        pfit = _np.polyfit(r0, pf, len(r0)-1)
                    else:
                        r0, xlabel, rvar = (ry0, 'y0 [mm]', 'x')
                        pfit = _np.polyfit(r0, pf, len(r0)-1)
                    pf_fit = _np.polyval(pfit, r0)

                    label = utils.var_param + ' = {} mm'.format(var_param)
                    _plt.figure(1)
                    _plt.plot(
                        1e3*r0, 1e6*pf, '.-', color=color, label=label)
                    _plt.plot(
                        1e3*r0, 1e6*pf_fit, '-', color=color, alpha=0.6)
                    print('kick plane: ', kick_plane)
                    print('plane: ', var)
                    print('Fitting:')
                    print(pfit[::-1])

                _plt.figure(1)
                _plt.xlabel(xlabel)
                _plt.ylabel('final {} [urad]'.format(klabel))
                _plt.title('Kick{} for gap {} mm, at pos{} {:+.3f} mm'.format(
                    kick_plane.upper(), gap, rvar, 0))
                _plt.legend()
                _plt.grid()
                _plt.tight_layout()
                if self.save_flag:
                    _plt.savefig(fname_fig, dpi=300)
                if self.plot_flag:
                    _plt.show()
                _plt.close()

    def check_kick_all_planes(
            self, width=None, phase=None, gap=None,
            planes=['X', 'Y'], kick_planes=['X', 'Y']):
        for var in planes:
            for kick_plane in kick_planes:
                fname = self.get_kmap_filename(folder_data=utils.FOLDER_DATA,
                                               width=width,
                                               phase=phase,
                                               gap=gap,
                                               linear=self.linear,
                                               meas_flag=self.meas_flag,
                                               shift_flag=self.shift_flag,
                                               filter_flag=self.filter_flag)
                self._idkickmap = _IDKickMap(fname)
                fname_fig = self._get_figname_allplanes(
                    width=width, phase=phase, gap=gap,
                    var=var, kick_plane=kick_plane)

                if var.lower() == 'x':
                    kmappos = self._idkickmap.posy
                else:
                    kmappos = self._idkickmap.posx

                for plane_idx, pos in enumerate(kmappos):
                    # if pos > 0:
                        # continue
                    rx0, ry0, pxf, pyf, *_ = self._calc_idkmap_kicks(
                        var=var, plane_idx=plane_idx)
                    if not self.linear:
                        pxf *= utils.RESCALE_KICKS
                        pyf *= utils.RESCALE_KICKS
                    pf, klabel =\
                        (pxf, 'px') if kick_plane.lower() == 'x' else (
                         pyf, 'py')
                    if var.lower() == 'x':
                        r0, xlabel, rvar = (rx0, 'x0 [mm]', 'y')
                    else:
                        r0, xlabel, rvar = (ry0, 'y0 [mm]', 'x')
                    rvar = 'y' if var.lower() == 'x' else 'x'
                    label = 'pos{} = {:+.3f} mm'.format(rvar, 1e3*pos)
                    _plt.plot(
                        1e3*r0, 1e6*pf, '.-', label=label)
                    _plt.xlabel(xlabel)
                    _plt.ylabel('final {} [urad]'.format(klabel))
                    _plt.title(
                        'Kicks for gap {} mm, width {} mm'.format(gap, width))
                _plt.legend()
                _plt.grid()
                _plt.tight_layout()
                if self.save_flag:
                    _plt.savefig(fname_fig, dpi=300)
                if self.plot_flag:
                    _plt.show()
                _plt.close()

    def check_kick_at_plane_trk(self, width, gap, phase):
        fname = self.get_kmap_filename(folder_data=utils.FOLDER_DATA,
                                       width=width, gap=gap,
                                       phase=phase, linear=self.linear,
                                       meas_flag=self.meas_flag,
                                       shift_flag=self.shift_flag,
                                       filter_flag=self.filter_flag)
        self._idkickmap = _IDKickMap(fname)
        plane_idx = list(self._idkickmap.posy).index(0)
        out = self._calc_idkmap_kicks(plane_idx=plane_idx)
        rx0, ry0 = out[0], out[1]
        pxf, pyf = out[2], out[3]
        rxf, ryf = out[4], out[5]
        if not self.linear:
            pxf *= utils.RESCALE_KICKS
            pyf *= utils.RESCALE_KICKS

        # lattice with IDs
        model, _ = self.create_model_ids(fname, linear=self.linear,
                                         create_ids=utils.create_ids)

        famdata = pymodels.si.get_family_data(model)

        # shift model
        idx = famdata[utils.ID_FAMNAME]['index']
        idx_begin = idx[0][0]
        idx_end = idx[0][-1]
        idx_dif = idx_end - idx_begin

        model = pyaccel.lattice.shift(model, start=idx_begin)

        rxf_trk, ryf_trk = _np.ones(len(rx0)), _np.ones(len(rx0))
        pxf_trk, pyf_trk = _np.ones(len(rx0)), _np.ones(len(rx0))
        for i, x0 in enumerate(rx0):
            coord_ini = _np.array([x0, 0, 0, 0, 0, 0])
            coord_fin, *_ = pyaccel.tracking.line_pass(
                model, coord_ini, indices='open')

            rxf_trk[i] = coord_fin[0, idx_dif+1]
            ryf_trk[i] = coord_fin[2, idx_dif+1]
            pxf_trk[i] = coord_fin[1, idx_dif+1]
            pyf_trk[i] = coord_fin[3, idx_dif+1]

        _plt.plot(
            1e3*rx0, 1e6*(rxf - rx0), '.-', color='C1', label='Pos X  kickmap')
        _plt.plot(1e3*rx0, 1e6*ryf, '.-', color='b', label='Pos Y  kickmap')
        _plt.plot(
            1e3*rx0, 1e6*(rxf_trk - rx0), 'o', color='C1',
            label='Pos X  tracking')
        _plt.plot(
            1e3*rx0, 1e6*ryf_trk, 'o', color='b', label='Pos Y  tracking')
        _plt.xlabel('x0 [mm]')
        _plt.ylabel('final dpos [um]')
        _plt.title('dPos')
        _plt.legend()
        _plt.grid()
        _plt.show()

        _plt.plot(1e3*rx0, 1e6*pxf, '.-', color='C1', label='Kick X  kickmap')
        _plt.plot(1e3*rx0, 1e6*pyf, '.-', color='b', label='Kick Y  kickmap')
        _plt.plot(
            1e3*rx0, 1e6*pxf_trk, 'o', color='C1', label='Kick X  tracking')
        _plt.plot(
            1e3*rx0, 1e6*pyf_trk, 'o', color='b', label='Kick Y  tracking')
        _plt.xlabel('x0 [mm]')
        _plt.ylabel('final px [urad]')
        _plt.title('Kicks')
        _plt.legend()
        _plt.grid()
        _plt.show()


class AnalysisEffects(Tools):

    def __init__(self):
        self._idkickmap = None
        self._model_id = None
        self._ids = None
        self._twiss0 = None
        self._twiss1 = None
        self._twiss2 = None
        self._twiss3 = None
        self._beta_flag = None
        self._stg = None
        self.id_famname = utils.ID_FAMNAME
        self.fitted_model = False
        self.calc_type = 0
        self.corr_system = 'SOFB'
        self.filter_flag = False
        self.shift_flag = True
        self.orbcorr_plot_flag = False
        self.bb_plot_flag = False
        self.meas_flag = False
        self.linear = False

    def _create_model_nominal(self):
        model0 = pymodels.si.create_accelerator()
        if self.fitted_model:
            famdata0 = pymodels.si.families.get_family_data(model0)
            idcs_qn0 = _np.array(famdata0['QN']['index']).ravel()
            idcs_qs0 = _np.array(famdata0['QS']['index']).ravel()
            data = _load_pickle(utils.FIT_PATH)
            kl = data['KL']
            ksl = data['KsL']
            pyaccel.lattice.set_attribute(model0, 'KL', idcs_qn0, kl)
            pyaccel.lattice.set_attribute(model0, 'KsL', idcs_qs0, ksl)

        model0.cavity_on = False
        model0.radiation_on = 0
        return model0

    def _get_knobs_locs(self):

        straight_nr = dict()
        knobs = dict()
        locs_beta = dict()
        for id_ in self._ids:
            straight_nr_ = int(id_.subsec[2:4])

            # get knobs and beta locations
            if straight_nr_ is not None:
                _, knobs_, _ = optics.symm_get_knobs(
                        self._model_id, straight_nr_)
                locs_beta_ = optics.symm_get_locs_beta(knobs_)
            else:
                knobs_, locs_beta_ = None, None

            straight_nr[id_.subsec] = straight_nr_
            knobs[id_.subsec] = knobs_
            locs_beta[id_.subsec] = locs_beta_

        return knobs, locs_beta, straight_nr

    def _calc_action_ratio(self, x0, nturns=1000):
        coord_ini = _np.array([x0, 0, 0, 0, 0, 0])
        coord_fin, *_ = pyaccel.tracking.ring_pass(
            self._model_id, coord_ini, nr_turns=nturns,
            turn_by_turn=True, parallel=True)
        rx = coord_fin[0, :]
        ry = coord_fin[2, :]
        twiss, *_ = pyaccel.optics.calc_twiss(self._model_id)
        betax, betay = twiss.betax, twiss.betay  # Beta functions
        jx = 1/(betax[0]*nturns)*(_np.sum(rx**2))
        jy = 1/(betay[0]*nturns)*(_np.sum(ry**2))

        print('action ratio = {:.3f}'.format(jy/jx))
        return jy/jx

    def _calc_coupling(self):
        ed_tang, *_ = pyaccel.optics.calc_edwards_teng(self._model_id)
        min_tunesep, ratio =\
            pyaccel.optics.estimate_coupling_parameters(ed_tang)
        print('Minimum tune separation = {:.3f}'.format(min_tunesep))

        return min_tunesep

    def _calc_dtune_betabeat(self, twiss1):
        dtunex = (twiss1.mux[-1] - self._twiss0.mux[-1]) / 2 / _np.pi
        dtuney = (twiss1.muy[-1] - self._twiss0.muy[-1]) / 2 / _np.pi
        bbeatx = (twiss1.betax - self._twiss0.betax)/self._twiss0.betax
        bbeaty = (twiss1.betay - self._twiss0.betay)/self._twiss0.betay
        bbeatx *= 100
        bbeaty *= 100
        bbeatx_rms = _np.std(bbeatx)
        bbeaty_rms = _np.std(bbeaty)
        bbeatx_absmax = _np.max(_np.abs(bbeatx))
        bbeaty_absmax = _np.max(_np.abs(bbeaty))
        return (
            dtunex, dtuney, bbeatx, bbeaty,
            bbeatx_rms, bbeaty_rms, bbeatx_absmax, bbeaty_absmax)

    def _analysis_uncorrected_perturbation(self, plot_flag=True):
        self._twiss1, *_ = pyaccel.optics.calc_twiss(
            self._model_id, indices='closed')

        results = self._calc_dtune_betabeat(twiss1=self._twiss1)
        dtunex, dtuney = results[0], results[1]
        bbeatx, bbeaty = results[2], results[3]
        bbeatx_rms, bbeaty_rms = results[4], results[5]
        bbeatx_absmax, bbeaty_absmax = results[6], results[7]

        if plot_flag:

            print(f'dtunex: {dtunex:+.6f}')
            print(f'dtunex: {dtuney:+.6f}')
            txt = f'bbetax: {bbeatx_rms:04.1f} % rms, '
            txt += f'{bbeatx_absmax:04.1f} % absmax'
            print(txt)
            txt = f'bbetay: {bbeaty_rms:04.1f} % rms, '
            txt += f'{bbeaty_absmax:04.1f} % absmax'
            print(txt)

            labelx = f'X ({bbeatx_rms:.1f} % rms)'
            labely = f'Y ({bbeaty_rms:.1f} % rms)'
            _plt.plot(self._twiss1.spos, bbeatx,
                      color='b', alpha=1, label=labelx)
            _plt.plot(self._twiss1.spos, bbeaty,
                      color='r', alpha=0.8, label=labely)
            _plt.xlabel('spos [m]')
            _plt.ylabel('Beta Beat [%]')
            _plt.title('Beta Beating from ' + self.id_famname)
            _plt.legend()
            _plt.grid()
            _plt.show()

    def _plot_beta_beating(self, width, phase, gap, xlim=None):
        fpath = self.get_data_path(folder_data=utils.FOLDER_DATA,
                                   width=width, phase=phase,
                                   gap=gap, meas_flag=self.meas_flag)
        # Compare optics between nominal value and uncorrect optics due ID
        results = self._calc_dtune_betabeat(twiss1=self._twiss1)
        dtunex, dtuney = results[0], results[1]
        bbeatx, bbeaty = results[2], results[3]
        bbeatx_rms, bbeaty_rms = results[4], results[5]
        bbeatx_absmax, bbeaty_absmax = results[6], results[7]
        print('Not symmetrized optics :')
        print(f'dtunex: {dtunex:+.2e}')
        print(f'dtuney: {dtuney:+.2e}')
        print(
            f'bbetax: {bbeatx_rms:04.3f} % rms,' +
            f' {bbeatx_absmax:04.3f} % absmax')
        print(
            f'bbetay: {bbeaty_rms:04.3f} % rms,' +
            f' {bbeaty_absmax:04.3f} % absmax')
        print()

        _plt.clf()

        label1 = {False: '-nominal', True: '-fittedmodel'}[self.fitted_model]

        _plt.figure(1)
        stg_tune = f'\u0394\u03BDx: {dtunex:+0.04f}\n'
        stg_tune += f'\u0394\u03BDy: {dtuney:+0.04f}'
        labelx = f'X ({bbeatx_rms:.3f} % rms)'
        labely = f'Y ({bbeaty_rms:.3f} % rms)'
        figname = fpath + 'opt{}-ids-nonsymm'.format(label1)
        if self.linear:
            figname += '-linear'
        _plt.plot(self._twiss0.spos, bbeatx,
                  color='b', alpha=1.0, label=labelx)
        _plt.plot(self._twiss0.spos, bbeaty,
                  color='r', alpha=0.8, label=labely)
        bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        bbmin = _np.min(_np.concatenate((bbeatx, bbeaty)))
        ylim = (1.1*bbmin, 1.1*bbmax)
        if xlim is None:
            xlim = (0, self._twiss0.spos[-1])
        pyaccel.graphics.draw_lattice(
            self._model_id, offset=bbmin, height=bbmax/8, gca=True)
        _plt.xlim(xlim)
        _plt.ylim(ylim)
        _plt.xlabel('spos [m]')
        _plt.ylabel('Beta Beating [%]')
        _plt.title('Tune shift:' + '\n' + stg_tune)
        _plt.suptitle(self.id_famname + ' - Non-symmetrized optics')
        _plt.tight_layout()
        _plt.legend()
        _plt.grid()
        _plt.savefig(figname, dpi=300)

        # Compare optics between nominal value and symmetrized optics
        results = self._calc_dtune_betabeat(twiss1=self._twiss2)
        dtunex, dtuney = results[0], results[1]
        bbeatx, bbeaty = results[2], results[3]
        bbeatx_rms, bbeaty_rms = results[4], results[5]
        bbeatx_absmax, bbeaty_absmax = results[6], results[7]
        print('symmetrized optics but uncorrect tunes:')
        print(f'dtunex: {dtunex:+.0e}')
        print(f'dtuney: {dtuney:+.0e}')
        print(
            f'bbetax: {bbeatx_rms:04.3f} % rms,' +
            f' {bbeatx_absmax:04.3f} % absmax')
        print(
            f'bbetay: {bbeaty_rms:04.3f} % rms,' +
            f' {bbeaty_absmax:04.3f} % absmax')
        print()

        _plt.figure(2)
        labelx = f'X ({bbeatx_rms:.3f} % rms)'
        labely = f'Y ({bbeaty_rms:.3f} % rms)'
        figname = fpath + 'opt{}-ids-symm'.format(label1)
        if self.linear:
            figname += '-linear'
        _plt.plot(self._twiss0.spos, bbeatx,
                  color='b', alpha=1.0, label=labelx)
        _plt.plot(self._twiss0.spos, bbeaty,
                  color='r', alpha=0.8, label=labely)
        bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        bbmin = _np.min(_np.concatenate((bbeatx, bbeaty)))
        ylim = (1.1*bbmin, 1.1*bbmax)
        _plt.xlim(xlim)
        _plt.ylim(ylim)
        # bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        pyaccel.graphics.draw_lattice(
            self._model_id, offset=bbmin, height=bbmax/8, gca=True)
        _plt.xlabel('spos [m]')
        _plt.ylabel('Beta Beating [%]')
        _plt.title('Beta Beating')
        _plt.suptitle(self.id_famname +
                      '- Symmetrized optics and uncorrected tunes')
        _plt.legend()
        _plt.grid()
        _plt.tight_layout()
        _plt.savefig(figname, dpi=300)

        # Compare optics between nominal value and all corrected
        results = self._calc_dtune_betabeat(twiss1=self._twiss3)
        dtunex, dtuney = results[0], results[1]
        bbeatx, bbeaty = results[2], results[3]
        bbeatx_rms, bbeaty_rms = results[4], results[5]
        bbeatx_absmax, bbeaty_absmax = results[6], results[7]
        print('symmetrized optics and corrected tunes:')
        print(f'dtunex: {dtunex:+.0e}')
        print(f'dtuney: {dtuney:+.0e}')
        print(
            f'bbetax: {bbeatx_rms:04.3f} % rms,' +
            f' {bbeatx_absmax:04.3f} % absmax')
        print(
            f'bbetay: {bbeaty_rms:04.3f} % rms,' +
            f' {bbeaty_absmax:04.3f} % absmax')

        _plt.figure(3)
        labelx = f'X ({bbeatx_rms:.3f} % rms)'
        labely = f'Y ({bbeaty_rms:.3f} % rms)'
        figname = fpath + 'opt{}-ids-symm-tunes'.format(label1)
        if self.linear:
            figname += '-linear'
        _plt.plot(self._twiss0.spos, bbeatx,
                  color='b', alpha=1.0, label=labelx)
        _plt.plot(self._twiss0.spos, bbeaty,
                  color='r', alpha=0.8, label=labely)
        bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        bbmin = _np.min(_np.concatenate((bbeatx, bbeaty)))
        ylim = (1.1*bbmin, 1.1*bbmax)
        _plt.xlim(xlim)
        _plt.ylim(ylim)
        # bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        pyaccel.graphics.draw_lattice(
            self._model_id, offset=bbmin, height=bbmax/8, gca=True)
        _plt.xlabel('spos [m]')
        _plt.ylabel('Beta Beating [%]')
        _plt.title('Corrections:' + '\n' + self._stg)
        _plt.suptitle(self.id_famname +
                      '- Symmetrized optics and corrected tunes')
        _plt.legend()
        _plt.grid()
        _plt.tight_layout()
        _plt.savefig(figname, dpi=300)
        _plt.show()
        _plt.clf()

    def _correct_beta(self, straight_nr, knobs, goal_beta, goal_alpha):
        dk_tot = _np.zeros(len(knobs))
        for i in range(7):
            dk, k = optics.correct_symmetry_withbeta(
                self._model_id, straight_nr, goal_beta, goal_alpha)
            if i == 0:
                k0 = k
            print('iteration #{}, \u0394K: {}'.format(i+1, dk))
            dk_tot += dk
        delta = dk_tot/k0
        self._stg = str()
        for i, fam in enumerate(knobs):
            self._stg += '{:<9s} \u0394K: {:+9.3f} % \n'.format(
                fam, 100*delta[i])
        print(self._stg)
        self._twiss2, *_ = pyaccel.optics.calc_twiss(
            self._model_id, indices='closed')
        print()

    def _correct_tunes(self, goal_tunes, knobs_out=None):
        tunes = self._twiss1.mux[-1]/_np.pi/2, self._twiss1.muy[-1]/_np.pi/2
        print('init    tunes: {:.9f} {:.9f}'.format(tunes[0], tunes[1]))
        for i in range(2):
            optics.correct_tunes_twoknobs(
                self._model_id, goal_tunes, knobs_out)
            twiss, *_ = pyaccel.optics.calc_twiss(self._model_id)
            tunes = twiss.mux[-1]/_np.pi/2, twiss.muy[-1]/_np.pi/2
            print('iter #{} tunes: {:.9f} {:.9f}'.format(
                i+1, tunes[0], tunes[1]))
        print('goal    tunes: {:.9f} {:.9f}'.format(
            goal_tunes[0], goal_tunes[1]))
        self._twiss3, *_ = pyaccel.optics.calc_twiss(
            self._model_id, indices='closed')
        print()

    def _correct_optics(self, width, phase, gap):

        # create unperturbed model for reference
        model0 = self._create_model_nominal()
        self._twiss0, *_ = pyaccel.optics.calc_twiss(model0, indices='closed')
        print('Model without ID:')
        print('length : {:.4f} m'.format(model0.length))
        print('tunex  : {:.6f}'.format(self._twiss0.mux[-1]/2/_np.pi))
        print('tuney  : {:.6f}'.format(self._twiss0.muy[-1]/2/_np.pi))
        # create model with ID
        fname = self.get_kmap_filename(folder_data=utils.FOLDER_DATA,
                                       width=width, phase=phase, gap=gap,
                                       shift_flag=self.shift_flag,
                                       filter_flag=self.filter_flag,
                                       linear=self.linear,
                                       meas_flag=self.meas_flag)
        self._model_id, self._ids = self.create_model_ids(fname=fname,
                                                          linear=self.linear,
                                                          create_ids=utils.create_ids)
        knobs, locs_beta, straight_nr = self._get_knobs_locs()

        print('element indices for straight section begin and end:')
        for idsubsec, locs_beta_ in locs_beta.items():
            print(idsubsec, locs_beta_)

        print('local quadrupole fams: ')
        for idsubsec, knobs_ in knobs.items():
            print(idsubsec, knobs_)

        # correct orbit
        kicks, spos_bpms, codx_c, cody_c, codx_u, cody_u, bpms = \
            orbcorr.correct_orbit_fb(
                model0, self._model_id, corr_system=self.corr_system,
                nr_steps=1, plot_flag=self.orbcorr_plot_flag)

        # calculate beta beating and delta tunes
        self._analysis_uncorrected_perturbation(plot_flag=False)

        # get list of ID model indices and set rescale_kicks to zero
        ids_ind_all = orbcorr.get_ids_indices(self._model_id)
        rescale_kicks_orig = list()
        for idx in range(len(ids_ind_all)//2):
            ind_id = ids_ind_all[2*idx:2*(idx+1)]
            rescale_kicks_orig.append(self._model_id[ind_id[0]].rescale_kicks)
            self._model_id[ind_id[0]].rescale_kicks = 0
            self._model_id[ind_id[1]].rescale_kicks = 0


        # loop over IDs turning rescale_kicks on, one by one.
        for idx in range(len(ids_ind_all)//2):
            # turn rescale_kicks on for ID index idx
            ind_id = ids_ind_all[2*idx:2*(idx+1)]
            self._model_id[ind_id[0]].rescale_kicks = rescale_kicks_orig[idx]
            self._model_id[ind_id[1]].rescale_kicks = rescale_kicks_orig[idx]
            fam_name = self._model_id[ind_id[0]].fam_name
            # print(idx, ind_id)
            # continue

            # search knob and straight_nr for ID index idx
            for subsec in knobs:
                straight_nr_ = straight_nr[subsec]
                knobs_ = knobs[subsec]
                locs_beta_ = locs_beta[subsec]
                if min(locs_beta_) < ind_id[0] and ind_id[1] < max(locs_beta_):
                    break
            k = self._calc_coupling()
            print()
            print('symmetrizing ID {} in subsec {}'.format(fam_name, subsec))

            # calculate nominal twiss
            goal_tunes = _np.array(
                [self._twiss0.mux[-1]/2/_np.pi, self._twiss0.muy[-1]/2/_np.pi])
            goal_beta = _np.array(
                [self._twiss0.betax[locs_beta_],
                 self._twiss0.betay[locs_beta_]])
            goal_alpha = _np.array(
                [self._twiss0.alphax[locs_beta_],
                 self._twiss0.alphay[locs_beta_]])
            print('goal_beta:')
            print(goal_beta)

            # symmetrize optics (local quad fam knobs)
            if self._beta_flag:

                self._correct_beta(
                    straight_nr_, knobs_, goal_beta, goal_alpha)

                self._correct_tunes(goal_tunes)

                if self.bb_plot_flag:
                    # plot results
                    self._plot_beta_beating(width=width, phase=phase, gap=gap)

        return self._model_id

    def _analysis_dynapt(self, model, width, phase, gap):

        model.radiation_on = 0
        model.cavity_on = False
        model.vchamber_on = True

        dynapxy = DynapXY(model)
        dynapxy.params.x_nrpts = 40
        dynapxy.params.y_nrpts = 20
        dynapxy.params.nrturns = 2*1024
        print(dynapxy)
        dynapxy.do_tracking()
        dynapxy.process_data()
        fig, axx, ayy = dynapxy.make_figure_diffusion(
            orders=(1, 2, 3, 4),
            nuy_bounds=(14.12, 14.45),
            nux_bounds=(49.05, 49.50))

        fpath = self.get_data_path(folder_data=utils.FOLDER_DATA, width=width,
                                   phase=phase, gap=gap, meas_flag=self.meas_flag)
        print(fpath)
        label1 = ['', '-ids-nonsymm', '-ids-symm'][self.calc_type]
        label2 = {False: '-nominal', True: '-fittedmodel'}[self.fitted_model]
        fig_name = fpath + 'dynapt{}{}.png'.format(label2, label1)
        if self.linear:
            fig_name = fig_name.replace('.png', '-linear.png')
        fig.savefig(fig_name, dpi=300, format='png')
        fig.clf()

    def run_analysis_dynapt(self, width, phase, gap):
        if self.calc_type == self.CALC_TYPES.nominal:
            model = self._create_model_nominal()
        elif self.calc_type in (
                self.CALC_TYPES.symmetrized, self.CALC_TYPES.nonsymmetrized):
            self._beta_flag = self.calc_type == self.CALC_TYPES.symmetrized
            model = self._correct_optics(
                width=width, phase=phase, gap=gap)
        else:
            raise ValueError('Invalid calc_type')

        self._analysis_dynapt(model=model, width=width, phase=phase, gap=gap)
