#!/usr/bin/env python-sirius
"""IDKickMap class."""


import numpy as _np
import matplotlib.pyplot as _plt
from scipy.optimize import curve_fit as _curve_fit

import fieldmaptrack as _fmaptrack

from . import utils as _utils


class IDKickMap:
    """ID KickMap and FieldMap."""

    DEF_AUTHOR = '# Author: FAC idanalysis.IDKickMap'
    DEF_BEAM_ENERGY = 3  # [GeV]
    DEF_RK_S_STEP = 0.2  # [mm]

    def __init__(self, kmap_fname=None, author=None, shift_on_axis=False):
        """."""
        self._kmap_fname = kmap_fname
        self.fmap_idlen = None  # [m]
        self.kmap_idlen = None  # [m]
        self.posx = None  # [m]
        self.posy = None  # [m]
        self.kickx = None  # [T².m²]
        self.kicky = None  # [T².m²]
        self.fposx = None  # [m]
        self.fposy = None  # [m]
        self.period_len = None  # [mm]
        self._fmap_config = None
        self.author = author or IDKickMap.DEF_AUTHOR
        self.shift_on_axis = shift_on_axis
        self.kickx_upstream = None
        self.kicky_upstream = None
        self.kickx_downstream = None
        self.kicky_downstream = None
        self._radia_model_config = None
        self._traj_init_rz = None
        self._traj_rk_min_rz = None
        self._config = None

        # load kickmap
        self._load_kmap()

    @property
    def kmap_fname(self):
        """Kickmap filename."""
        return self._kmap_fname

    @kmap_fname.setter
    def kmap_fname(self, value):
        """Set kickmap filename and load file."""
        self._kmap_fname = value
        self._load_kmap()

    @property
    def fmap_fname(self):
        """Fieldmap filename."""
        if self._fmap_config:
            return self._fmap_config.fmap.filename
        else:
            return None

    @fmap_fname.setter
    def fmap_fname(self, value):
        """Set fieldmap filename and load file."""
        self._fmap_config = IDKickMap._create_fmap_config(
            fmap_fname=value,
            beam_energy=self.beam_energy, rk_s_step=self.rk_s_step)

    @property
    def radia_model(self):
        """Fieldmap filename."""
        if self._radia_model:
            return self._radia_model
        else:
            return None

    @radia_model.setter
    def radia_model(self, value):
        """Set fieldmap filename and load file."""
        self._radia_model_config = IDKickMap._create_radia_model_config(
            radia_model=value,
            rk_s_step=self.rk_s_step)

    @property
    def beam_energy(self):
        """."""
        if self._fmap_config:
            return self._fmap_config.beam.energy
        elif self._radia_model_config:
            return self._radia_model_config.beam.energy
        else:
            return None

    @beam_energy.setter
    def beam_energy(self, value):
        """."""
        if not self._fmap_config and not self._radia_model_config:
            raise AttributeError('Undefined configuration!')
        elif not self._radia_model_config:
            IDKickMap._update_fmap_energy(self._fmap_config, value)
        else:
            IDKickMap._update_radia_model_energy(
                self._radia_model_config, value)

    @property
    def brho(self):
        """."""
        if self._fmap_config:
            return self._fmap_config.beam.brho
        elif self._radia_model_config:
            return self._radia_model_config.beam.brho
        else:
            return None

    @property
    def rk_s_step(self):
        """."""
        if self._fmap_config:
            return self._fmap_config.traj_rk_s_step
        elif self._radia_model_config:
            return self._radia_model_config.traj_rk_s_step
        else:
            return None

    @rk_s_step.setter
    def rk_s_step(self, value):
        """."""
        if not self._fmap_config and not self._radia_model_config:
            raise AttributeError('Undefined fieldmap configuration!')
        elif not self._radia_model_config:
            self._fmap_config.traj_rk_s_step = value
        else:
            self._radia_model_config.traj_rk_s_step = value

    @property
    def traj_init_rz(self):
        """."""
        if self._fmap_config:
            return self._fmap_config.traj_init_rz
        elif self._radia_model_config:
            return self._radia_model_config.traj_init_rz
        else:
            return None

    @traj_init_rz.setter
    def traj_init_rz(self, value):
        """."""
        if not self._fmap_config and not self._radia_model_config:
            raise AttributeError('Undefined configuration!')
        elif not self._radia_model_config:
            self._fmap_config.traj_init_rz = value
        else:
            self._radia_model_config.traj_init_rz = value

    @property
    def traj_rk_min_rz(self):
        """."""
        if self._fmap_config:
            return self._fmap_config.traj_rk_min_rz
        elif self._radia_model_config:
            return self._radia_model_config.traj_rk_min_rz
        else:
            return None

    @traj_rk_min_rz.setter
    def traj_rk_min_rz(self, value):
        """."""
        if not self._fmap_config and not self._radia_model_config:
            raise AttributeError('Undefined configuration!')
        elif not self._radia_model_config:
            self._fmap_config.traj_rk_min_rz = value
        else:
            self._radia_model_config.traj_rk_min_rz = value

    @property
    def radia_model_config(self):
        """Return Radia Model Config."""
        return self._radia_model_config

    @property
    def fmap_config(self):
        """Return fieldmap Config."""
        return self._fmap_config

    @property
    def fmap(self):
        """Return FieldMap."""
        return self._fmap_config.fmap

    @property
    def traj(self):
        """Return RK Trajectory."""
        config = self._fmap_config or self._radia_model_config
        return config.traj

    def fmap_calc_trajectory(
            self, traj_init_rx, traj_init_ry,
            traj_init_px=0, traj_init_py=0,
            traj_init_rz=None, traj_rk_min_rz=None,
            rk_s_step=None, **kwargs):
        """."""
        if rk_s_step is not None:
            self.rk_s_step = rk_s_step

        config = self._fmap_config or self._radia_model_config
        config.traj_init_rx = traj_init_rx * 1e3
        config.traj_init_ry = traj_init_ry * 1e3
        config.traj_init_px = traj_init_px
        config.traj_init_py = traj_init_py
        if traj_init_rz is not None:
            config.traj_init_rz = traj_init_rz
        if traj_init_rz is not None:
            config.traj_init_rz = traj_init_rz
        if traj_rk_min_rz is not None:
            config.traj_rk_min_rz = traj_rk_min_rz
        config = IDKickMap._fmap_calc_traj(config)
        return config

    def generate_linear_kickmap(
            self, brho, posx, posy, cxx, cyy, cxy=0, cyx=0):
        """Generate a linear kickmap based on Ellaume formalism"""
        self.posx = _np.array(posx)  # [m]
        self.posy = _np.array(posy)  # [m]
        self.kickx = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.kicky = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.fposx = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.fposy = _np.full((len(self.posy), len(self.posx)), _np.inf)
        for i, ryi in enumerate(self.posy):
            for j, rxi in enumerate(self.posx):
                pxf = cxx*rxi + cxy*ryi
                pyf = cyx*rxi + cyy*ryi
                stg = 'rx = {:.01f} mm, ry = {:.01f}: '.format(
                    rxi*1e3, ryi*1e3)
                stg += 'px = {:.01f} urad, py = {:.01f} urad'.format(
                    pxf*1e6, pyf*1e6)
                print(stg)
                self.kickx[i, j] = pxf*brho**2
                self.kicky[i, j] = pyf*brho**2

    def fmap_calc_kickmap(
            self, posx, posy, beam_energy=None, rk_s_step=None):
        """."""
        self.posx = _np.array(posx)  # [m]
        self.posy = _np.array(posy)  # [m]

        if beam_energy is not None:
            self.beam_energy = beam_energy
        if rk_s_step is not None:
            self.rk_s_step = rk_s_step
        brho = self.brho
        self.kickx = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.kicky = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.fposx = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.fposy = _np.full((len(self.posy), len(self.posx)), _np.inf)
        config = self._fmap_config or self._radia_model_config
        self._config = config
        for i, ryi in enumerate(self.posy):
            for j, rxi in enumerate(self.posx):
                self._config.traj_init_rx = 1e3 * rxi
                self._config.traj_init_ry = 1e3 * ryi
                IDKickMap._fmap_calc_traj(self._config)
                pxf = self._config.traj.px[-1]
                pyf = self._config.traj.py[-1]
                rxf = self._config.traj.rx[-1]
                ryf = self._config.traj.ry[-1]
                stg = 'rx = {:.01f} mm, ry = {:.01f}: '.format(
                    rxi*1e3, ryi*1e3)
                stg += 'px = {:.01f} urad, py = {:.01f} urad'.format(
                    pxf*1e6, pyf*1e6)
                print(stg)
                self.kickx[i, j] = pxf * brho**2
                self.kicky[i, j] = pyf * brho**2
                self.fposx[i, j] = rxf / 1e3
                self.fposy[i, j] = ryf / 1e3

    def find_fit_plateau(self, p):
        """."""
        opt = _curve_fit(self._plateau_function, self.posx, p)[0]
        return opt

    def filter_kmap(self, posx=None, posy=None, order=4, plot_flag=False):
        self._load_kmap()
        if posx is not None:
            kickx = _np.zeros((len(self.posy), len(posx)))
            fposx = _np.zeros((len(self.posy), len(posx)))
            for i, ryi in enumerate(self.posy):
                opt = _np.polyfit(self.posx, self.kickx[i, :], order)
                pxf = _np.polyval(opt, posx)
                xfit = _np.polyfit(self.posx, self.fposx[i, :], order)
                xf = _np.polyval(xfit, posx)
                kickx[i, :] = pxf
                fposx[i, :] = xf
                label = 'y = {:.2f} mm'.format(1e3*ryi)
                if plot_flag:
                    _plt.plot(1e3*self.posx, 1e6*self.kickx[i, :],
                              '.-', label=label)
                    _plt.plot(1e3*posx, 1e6*kickx[i, :])
            if plot_flag:
                _plt.xlabel('x pos [mm]')
                _plt.ylabel('kicks [Tm²]')
                _plt.legend()
                _plt.show()

            kicky = _np.zeros((len(self.posy), len(posx)))
            fposy = _np.zeros((len(self.posy), len(posx)))
            for i, ryi in enumerate(self.posy):
                opt = _np.polyfit(self.posx, self.kicky[i, :], order)
                pyf = _np.polyval(opt, posx)
                yfit = _np.polyfit(self.posx, self.fposy[i, :], order)
                yf = _np.polyval(yfit, posx)
                kicky[i, :] = pyf
                fposy[i, :] = yf
                label = 'y = {:.2f} mm'.format(1e3*ryi)
                if plot_flag:
                    _plt.plot(1e3*self.posx, 1e6*self.kicky[i, :],
                              '.-', label=label)
                    _plt.plot(1e3*posx, 1e6*kicky[i, :])
            if plot_flag:
                _plt.xlabel('x pos [mm]')
                _plt.ylabel('kicks [Tm²]')
                _plt.legend()
                _plt.show()

            self.posx = posx

            self.kickx = kickx
            self.fposx = fposx

            self.kicky = kicky
            self.fposy = fposy

        if posy is not None:
            kickx = _np.zeros((len(posy), len(self.posx)))
            fposx = _np.zeros((len(posy), len(self.posx)))
            for i, rxi in enumerate(self.posx):
                opt = _np.polyfit(self.posy, self.kickx[:, i], order)
                pxf = _np.polyval(opt, posy)
                xfit = _np.polyfit(self.posy, self.fposx[:, i], order)
                xf = _np.polyval(xfit, posy)
                kickx[:, i] = pxf
                fposx[:, i] = xf
                label = 'x = {:.2f} mm'.format(1e3*rxi)
                if plot_flag:
                    _plt.plot(1e3*self.posy, 1e6*self.kickx[:, i],
                              '.-', label=label)
                    _plt.plot(1e3*posy, 1e6*kickx[:, i])
            if plot_flag:
                _plt.xlabel('y pos [mm]')
                _plt.ylabel('kicks [Tm²]')
                _plt.legend()
                _plt.show()

            kicky = _np.zeros((len(posy), len(self.posx)))
            fposy = _np.zeros((len(posy), len(self.posx)))
            for i, rxi in enumerate(self.posx):
                opt = _np.polyfit(self.posy, self.kicky[:, i], order)
                pyf = _np.polyval(opt, posy)
                yfit = _np.polyfit(self.posy, self.fposy[:, i], order)
                yf = _np.polyval(yfit, posy)
                kicky[:, i] = pyf
                fposy[:, i] = yf
                label = 'x = {:.2f} mm'.format(1e3*rxi)
                if plot_flag:
                    _plt.plot(1e3*self.posy, 1e6*self.kicky[:, i],
                              '.-', label=label)
                    _plt.plot(1e3*posy, 1e6*kicky[:, i])
            if plot_flag:
                _plt.xlabel('y pos [mm]')
                _plt.ylabel('kicks [Tm²]')
                _plt.legend()
                _plt.show()

            self.posy = posy

            self.kickx = kickx
            self.fposx = fposx

            self.kicky = kicky
            self.fposy = fposy

    def save_kickmap_file(self, kickmap_filename):
        """."""
        rst = self.__str__()
        my_file = open(kickmap_filename, "w")  # w=writing
        my_file.write(rst)
        my_file.close()

    def calc_KsL_kickx_at_x(self, ix, plot=True):
        """."""
        posy = self.posy  # [m]
        posx = self.posx[ix]
        kickx = self.kickx[:, ix] / self.brho**2  # [rad]
        poly = _np.polyfit(posy, kickx, len(posy)-5)
        if plot:
            kickx_fit = _np.polyval(poly, posy)
            _plt.clf()
            _plt.plot(1e3*posy, 1e6*kickx, 'o', label='data')
            _plt.plot(1e3*posy, 1e6*kickx_fit, label='fit')
            _plt.xlabel('posy [mm]')
            _plt.ylabel('kickx [urad]')
            _plt.title('Kickx @ x = {:.1f} mm'.format(1e3*posx))
            _plt.legend()
            _plt.grid()
            _plt.savefig('kickx_ix_{}.png'.format(ix))
            # plt.show()
        KsL = poly[-2] * self.brho
        return KsL

    def calc_KsL_kicky_at_y(self, iy, plot=True):
        """."""
        posx = self.posx  # [m]
        posy = self.posy[iy]
        kicky = self.kicky[iy, :] / self.brho**2  # [rad]
        poly = _np.polyfit(posx, kicky, len(posx)-5)
        if plot:
            kicky_fit = _np.polyval(poly, posx)
            _plt.clf()
            _plt.plot(1e3*posx, 1e6*kicky, 'o', label='data')
            _plt.plot(1e3*posx, 1e6*kicky_fit, label='fit')
            _plt.xlabel('posx [mm]')
            _plt.ylabel('kicky [urad]')
            _plt.title('Kicky @ y = {:.1f} mm'.format(1e3*posy))
            _plt.legend()
            _plt.grid()
            _plt.savefig('kicky_iy_{}.png'.format(iy))
            # plt.show()
        KsL = poly[-2] * self.brho
        return KsL

    def calc_KsL_kickx(self):
        """."""
        posx = self.posx  # [m]
        ksl = []
        for ix, _ in enumerate(posx):
            ksl_ = self.calc_KsL_kickx_at_x(ix, False)
            ksl.append(ksl_)
        return posx, _np.array(ksl)

    def calc_KsL_kicky(self):
        """."""
        posy = self.posy  # [m]
        ksl = []
        for iy, _ in enumerate(posy):
            ksl_ = self.calc_KsL_kicky_at_y(iy, False)
            ksl.append(ksl_)
        return posy, _np.array(ksl)

    def model_rz_field_center(self):
        """Return rz pos of field center."""
        model = self.radia_model
        # rz = fmap.rz
        # bx = radia_model.get_field[fmap.ry_zero][fmap.rx_zero][:]
        # by = radia_model.get_field[fmap.ry_zero][fmap.rx_zero][:]
        # bz = radia_model.get_field[fmap.ry_zero][fmap.rx_zero][:]
        # rz_center = _utils.calc_rz_of_field_center(rz, bx, by, bz)
        # return rz_center

    def fmap_rz_field_center(self):
        """Return rz pos of field center."""
        fmap = self.fmap_config.fmap
        rz = fmap.rz
        bx = fmap.bx[fmap.ry_zero][fmap.rx_zero][:]
        by = fmap.by[fmap.ry_zero][fmap.rx_zero][:]
        bz = fmap.bz[fmap.ry_zero][fmap.rx_zero][:]
        rz_center = _utils.calc_rz_of_field_center(rz, bx, by, bz)
        return rz_center

    def calc_id_termination_kicks(
            self, period_len=None, kmap_idlen=None, plot_flag=False):
        """."""
        # get parameters
        kmap_idlen = kmap_idlen or self.kmap_idlen
        self.kmap_idlen = kmap_idlen
        period_len = period_len or self.period_len
        self.period_len = period_len
        nr_central_periods = int(kmap_idlen*1e3/period_len) - 4

        config = self.fmap_calc_trajectory(traj_init_rx=0, traj_init_ry=0)
        self._config = config

        # get indices for central part of ID
        if self._fmap_config:
            rz_center = self.fmap_rz_field_center()
        elif self._radia_model_config:
            rz_center = 0
        rz = self._config.traj.rz
        px = self._config.traj.px
        py = self._config.traj.py
        idx_begin_fit = self._find_value_idx(
            rz, rz_center - period_len*nr_central_periods/2)
        idx_end_fit = self._find_value_idx(
            rz, rz_center + period_len*nr_central_periods/2)

        for idx, pxy in enumerate([px, py]):
            rz_sample = rz[idx_begin_fit:idx_end_fit]
            p_sample = pxy[idx_begin_fit:idx_end_fit]
            opt = self.find_fit(rz_sample, p_sample)
            idx_begin_ID = self._find_value_idx(rz, -kmap_idlen * 1e3/2)
            idx_end_ID = self._find_value_idx(rz, +kmap_idlen * 1e3/2)
            linefit = self._linear_function(rz, opt[2], opt[3])
            kick_begin = linefit[idx_begin_ID] - pxy[0]
            kick_end = pxy[-1] - linefit[idx_end_ID]
            if plot_flag:
                _plt.plot(rz, pxy)
                _plt.plot(rz_sample, p_sample, '.')
                _plt.plot(rz, linefit)
                _plt.show()
            if idx == 0:
                self.kickx_upstream = kick_begin*self.brho**2
                self.kickx_downstream = kick_end*self.brho**2
                print('ID length: {:.3f} m'.format(kmap_idlen))
                print("kickx_upstream: {:11.4e}  T2m2".format(
                    self.kickx_upstream))
                print("kickx_downstream: {:11.4e}  T2m2".format(
                    self.kickx_downstream))
            elif idx == 1:
                self.kicky_upstream = kick_begin*self.brho**2
                self.kicky_downstream = kick_end*self.brho**2
                print("kicky_upstream: {:11.4e}  T2m2".format(
                    self.kicky_upstream))
                print("kicky_downstream: {:11.4e}  T2m2".format(
                    self.kicky_downstream))

    def plot_kickx_vs_posy(self, indx, title=''):
        """."""
        posx = self.posx
        posy = self.posy
        kickx = self.kickx / self.brho**2
        colors = _plt.cm.jet(_np.linspace(0, 1, len(indx)))
        _plt.figure(figsize=(8, 6))
        for c, ix in enumerate(indx):
            x = posx[ix]
            _plt.plot(
                1e3*posy, 1e6*kickx[:, ix], '-', color=colors[c])
            _plt.plot(
                1e3*posy, 1e6*kickx[:, ix], 'o', color=colors[c],
                label='posx = {:+.1f} mm'.format(1e3*x))
        _plt.xlabel('posy [mm]')
        _plt.ylabel('kickx [urad]')
        _plt.title(title)
        _plt.grid()
        _plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.05))
        _plt.tight_layout(True)
        _plt.show()

    def plot_kicky_vs_posx(self, indy, title=''):
        """."""
        posx = self.posx
        posy = self.posy
        kicky = self.kicky / self.brho**2
        colors = _plt.cm.jet(_np.linspace(0, 1, len(indy)))
        _plt.figure(figsize=(8, 6))
        for c, iy in enumerate(indy):
            y = posy[iy]
            _plt.plot(1e3*posx, 1e6*kicky[iy, :], '-', color=colors[c])
            _plt.plot(
                1e3*posx, 1e6*kicky[iy, :], 'o', color=colors[c],
                label='posy = {:+.1f} mm'.format(1e3*y))
        _plt.xlabel('posx [mm]')
        _plt.ylabel('kicky [urad]')
        _plt.title(title)
        _plt.grid()
        _plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.05))
        _plt.tight_layout(True)
        _plt.show()

    def plot_examples(self):
        """."""
        self.load_kmap_delta(idx=0)
        self.calc_KsL_kickx_at_x(ix=14, plot=True)
        self.calc_KsL_kicky_at_y(iy=8, plot=True)

    def fit_function(self, rz, amp1, phi1, a, b):
        """."""
        f = amp1 * _np.sin(2*_np.pi/self.period_len * rz + phi1)
        f += a * rz + b
        return f

    def find_fit(self, rz, pvec):
        """."""
        opt = _curve_fit(self.fit_function, rz, pvec)[0]
        return opt

    def load_kmap_delta(self, idx):
        """Load Delta ID kickmap defined by configuration index 'idx'."""
        configs = _utils.create_deltadata()
        kmap_fname = configs.get_kickmap_filename(configs[idx])
        self.kmap_fname = kmap_fname

    def _load_kmap(self):
        """."""
        if not self.kmap_fname:
            return
        info = IDKickMap._load_kmap_info(self.kmap_fname)
        self.fmap_idlen = info['id_length']
        self.posx, self.posy = info['posx'], info['posy']
        self.kickx, self.kicky = info['kickx'], info['kicky']
        self.fposx, self.fposy = info['fposx'], info['fposy']
        self.kickx_upstream = info['kickx_upstream']
        self.kicky_upstream = info['kicky_upstream']
        self.kickx_downstream = info['kickx_downstream']
        self.kicky_downstream = info['kicky_downstream']

        if self.shift_on_axis:
            # find indices of central line
            try:
                indx = list(self.posx).index(0)
                indy = list(self.posy).index(0)
            except ValueError:
                raise ValueError('Kickmap does not have central transverse line!')
            # shift kicks on axis
            kickx0 = self.kickx[indy][indx]
            kicky0 = self.kicky[indy][indx]
            self.kickx -= kickx0
            self.kicky -= kicky0

    def __str__(self):
        """."""
        rst = ''
        # header
        rst += self.author
        if self.kickx_upstream is not None:
            flag = True
            if flag:
                fmt = (
                    '\n# Termination_kicks [T2m2]: '
                    '{:+11.4e} {:+11.4e} {:+11.4e} {:+11.4e} '
                    )
                rst += fmt.format(
                    self.kickx_upstream, self.kicky_upstream,
                    self.kickx_downstream, self.kicky_downstream)
        else:
            rst += '\n# '

        id_len = self.kmap_idlen or self.fmap_idlen
        rst += '\n# Total Length of Longitudinal Interval [m]'
        rst += '\n{}'.format(id_len)
        rst += '\n# Number of Horizontal Points'
        rst += '\n{}'.format(len(self.posx))
        rst += '\n# Number of Vertical Points'
        rst += '\n{}'.format(len(self.posy))

        kickx_end = (self.kickx_upstream or 0) + (self.kickx_downstream or 0)
        kicky_end = (self.kicky_upstream or 0) + (self.kicky_downstream or 0)

        rst += '\n# Total Horizontal 2nd Order Kick [T2m2]'
        rst += '\nSTART'
        # first line
        rst += '\n{:11s} '.format('')
        for rxi in self.posx:
            rst += '{:+011.5f} '.format(rxi)
        # table
        for i, ryi in enumerate(self.posy[::-1]):
            rst += '\n{:+011.5f} '.format(ryi)
            for j, rxi in enumerate(self.posx):
                rst += '{:+11.4e} '.format(self.kickx[-i-1, j] - kickx_end)

        rst += '\n# Total Vertical 2nd Order Kick [T2m2]'
        rst += '\nSTART'
        # first line
        rst += '\n{:11s} '.format('')
        for rxi in self.posx:
            rst += '{:+011.5f} '.format(rxi)
        # table
        for i, ryi in enumerate(self.posy[::-1]):
            rst += '\n{:+011.5f} '.format(ryi)
            for j, rxi in enumerate(self.posx):
                rst += '{:+11.4e} '.format(self.kicky[-i-1, j] - kicky_end)

        rst += '\n# Horizontal Final Position [m]'
        rst += '\nSTART'
        # first line
        rst += '\n{:11s} '.format('')
        for rxi in self.posx:
            rst += '{:+011.5f} '.format(rxi)
        # table
        for i, ryi in enumerate(self.posy[::-1]):
            rst += '\n{:+011.5f} '.format(ryi)
            for j, rxi in enumerate(self.posx):
                rst += '{:+11.4e} '.format(self.fposx[-i-1, j])

        rst += '\n# Vertical Final Position [m]'
        rst += '\nSTART'
        # first line
        rst += '\n{:11s} '.format('')
        for rxi in self.posx:
            rst += '{:+011.5f} '.format(rxi)
        # table
        for i, ryi in enumerate(self.posy[::-1]):
            rst += '\n{:+011.5f} '.format(ryi)
            for j, rxi in enumerate(self.posx):
                rst += '{:+11.4e} '.format(self.fposy[-i-1, j])
        return rst

    @staticmethod
    def _linear_function(x, a, b):
        return a*x + b

    @staticmethod
    def _find_value_idx(data, value):
        diff_array = _np.absolute(data-value)
        index = diff_array.argmin()
        return index

    @staticmethod
    def _load_kmap_info(kmap_fname):
        """."""
        kickx_up = kickx_down = 0
        kicky_up = kicky_down = 0

        with open(kmap_fname) as fp:
            lines = fp.readlines()

        tables = []
        params = []
        for line in lines:
            line = line.strip()
            if line.startswith('START'):
                pass
            elif line.startswith('#'):
                if 'Termination_kicks' in line:
                    *_, kicks = line.split('Termination_kicks')
                    _, k1, k2, k3, k4 = kicks.strip().split(' ')
                    kickx_up = float(k1)
                    kicky_up = float(k2)
                    kickx_down = float(k3)
                    kicky_down = float(k4)
            else:
                data = [float(val) for val in line.split()]
                if len(data) == 1:
                    params.append(data[0])
                elif len(data) == int(params[1]):
                    posx = _np.array(data)
                else:
                    # print(data)
                    # return
                    tables.append(data)

        id_length = params[0]
        nrpts_y = int(params[2])
        tables = _np.array(tables)
        posy = tables[:nrpts_y, 0]
        tables = tables[:, 1:]

        kickx = tables[0*nrpts_y:1*nrpts_y, :]
        kicky = tables[1*nrpts_y:2*nrpts_y, :]
        fposx = tables[2*nrpts_y:3*nrpts_y, :]
        fposy = tables[3*nrpts_y:4*nrpts_y, :]
        if posy[-1] < posy[0]:
            posy = posy[::-1]
            kickx = kickx[::-1, :]
            kicky = kicky[::-1, :]
            fposx = fposx[::-1, :]
            fposy = fposy[::-1, :]
        info = dict()
        info['id_length'] = id_length
        info['posx'], info['posy'] = posx, posy
        info['kickx'], info['kicky'] = kickx, kicky
        info['fposx'], info['fposy'] = fposx, fposy
        info['kickx_upstream'] = kickx_up
        info['kicky_upstream'] = kicky_up
        info['kickx_downstream'] = kickx_down
        info['kicky_downstream'] = kicky_down
        return info

    @staticmethod
    def _create_fmap_config(fmap_fname, beam_energy, rk_s_step):
        config = _fmaptrack.common_analysis.Config()
        config.config_label = 'id-3gev'
        config.magnet_type = 'insertion-device'  # not necessary
        config.interactive_mode = True
        config.fmap_filename = fmap_fname
        config.fmap_extrapolation_flag = False
        config.not_raise_range_exceptions = True

        transforms = dict()
        config.fmap = _fmaptrack.FieldMap(
            config.fmap_filename,
            transforms=transforms,
            not_raise_range_exceptions=config.not_raise_range_exceptions)

        config.radia_model = None
        config.traj_load_filename = None
        config.traj_is_reference_traj = True
        config.traj_init_rz = min(config.fmap.rz)
        config.traj_final_rz = max(config.fmap.rz)
        config.traj_rk_s_step = rk_s_step
        config.traj_rk_length = None
        config.traj_rk_nrpts = None
        config.traj_force_midplane_flag = False

        return config

    @staticmethod
    def _create_radia_model_config(radia_model, rk_s_step):
        config = _fmaptrack.common_analysis.Config()
        config.config_label = 'id-3gev'
        config.magnet_type = 'insertion-device'  # not necessary
        config.interactive_mode = True
        config.radia_model = radia_model
        config.fmap_extrapolation_flag = False
        config.not_raise_range_exceptions = True

        config.fmap = None
        config.traj_load_filename = None
        config.traj_is_reference_traj = True
        # config.traj_init_rz = min(config.fmap.rz)
        config.traj_rk_s_step = rk_s_step
        config.traj_rk_length = None
        config.traj_rk_nrpts = None
        config.traj_force_midplane_flag = False

        return config

    @staticmethod
    def _fmap_calc_traj(config, **kwargs):
        """Calcs trajectory."""
        config.beam = _fmaptrack.Beam(energy=config.beam_energy)
        config.traj = _fmaptrack.Trajectory(
            beam=config.beam,
            fieldmap=config.fmap,
            radia_model=config.radia_model,
            not_raise_range_exceptions=config.not_raise_range_exceptions)
        if config.traj_init_rx is not None:
            init_rx = config.traj_init_rx
        else:
            init_rx = 0.0
        if hasattr(config, 'traj_init_ry'):
            init_ry = config.traj_init_ry
        else:
            config.traj_init_ry = init_ry = 0.0
        if hasattr(config, 'traj_init_rz'):
            init_rz = config.traj_init_rz
        else:
            config.traj_init_rz = init_rz = 0.0
        if hasattr(config, 'traj_init_px'):
            init_px = config.traj_init_px  # * 180/_np.pi
        else:
            config.traj_init_px = init_px = 0.0
        if hasattr(config, 'traj_init_py'):
            init_py = config.traj_init_py  # * 180/_np.pi
        else:
            config.traj_init_py = init_py = 0.0
        init_pz = _np.sqrt(1.0 - init_px**2 - init_py**2)
        has_rk_min_rz = hasattr(config, 'traj_rk_min_rz')
        if has_rk_min_rz and config.traj_rk_min_rz is not None:
            rk_min_rz = config.traj_rk_min_rz
        elif config.traj_rk_s_step > 0.0:
            rk_min_rz = -1*config.traj_init_rz
        else:
            rk_min_rz = config.traj_init_rz
        config.traj.calc_trajectory(
            init_rx=init_rx, init_ry=init_ry, init_rz=init_rz,
            init_px=init_px, init_py=init_py, init_pz=init_pz,
            s_step=config.traj_rk_s_step,
            s_length=config.traj_rk_length,
            s_nrpts=config.traj_rk_nrpts,
            min_rz=rk_min_rz,
            force_midplane=config.traj_force_midplane_flag,
            **kwargs)

        return config

    @staticmethod
    def multipoles_analysis(config):
        """Multipoles analysis."""
        # calcs multipoles around reference trajectory
        # ============================================
        multi_perp = config.multipoles_perpendicular_grid
        multi_norm = config.multipoles_normal_field_fitting_monomials
        multi_skew = config.multipoles_skew_field_fitting_monomials
        config.multipoles = _fmaptrack.Multipoles(
            trajectory=config.traj,
            perpendicular_grid=multi_perp,
            normal_field_fitting_monomials=multi_norm,
            skew_field_fitting_monomials=multi_skew)
        config.multipoles.calc_multipoles(is_ref_trajectory_flag=False)
        config.multipoles.calc_multipoles_integrals()
        config.multipoles.calc_multipoles_integrals_relative(
            config.multipoles.normal_multipoles_integral,
            main_monomial=0,
            r0=config.multipoles_r0,
            is_skew=False)

        # calcs effective length

        # main_monomial = config.normalization_monomial
        # monomials = config.multipoles.normal_field_fitting_monomials
        # idx_n = monomials.index(main_monomial)
        # idx_z = list(config.traj.s).index(0.0)
        # main_multipole_center = config.multipoles.normal_multipoles[idx_n,idx_z]
        # config.multipoles.effective_length = config.multipoles.normal_multipoles_integral[idx_n] / main_multipole_center

        main_monomial = config.normalization_monomial
        monomials = config.multipoles.normal_field_fitting_monomials
        idx_n = monomials.index(main_monomial)

        if hasattr(config, 'hardedge_half_region'):
            sel = config.traj.s < config.hardedge_half_region
            s = config.traj.s[sel]
            field = config.multipoles.normal_multipoles[idx_n, sel]
            integrated_field = _np.trapz(field, s)
            hardedge_field = integrated_field / config.hardedge_half_region
            config.multipoles.effective_length = \
                config.multipoles.normal_multipoles_integral[idx_n] / \
                hardedge_field
        else:
            idx_z = list(config.traj.s).index(0.0)
            main_multipole_center = \
                config.multipoles.normal_multipoles[idx_n, idx_z]
            config.multipoles.effective_length = \
                config.multipoles.normal_multipoles_integral[idx_n] / \
                main_multipole_center

        # saves multipoles to file
        if not config.interactive_mode:
            config.multipoles.save('multipoles.txt')

        # prints basic information on multipoles
        # ======================================
        print('--- multipoles on reference trajectory (rz > 0) ---')
        print(config.multipoles)

        if not config.interactive_mode:
            comm_analysis = _fmaptrack.common_analysis
            # plots normal multipoles
            config = comm_analysis.plot_normal_multipoles(config)
            # plots skew multipoles
            config = comm_analysis.plot_skew_multipoles(config)
            # plots residual normal field
            # config = plot_residual_field_in_curvilinear_system(config)
            config = comm_analysis.plot_residual_normal_field(config)
            # plots residual skew field
            config = comm_analysis.plot_residual_skew_field(config)
        return config

    @staticmethod
    def _update_fmap_energy(fmap_config, beam_energy):
        if not fmap_config:
            return
        fmap_config.beam_energy = beam_energy
        fmap_config.beam = _fmaptrack.Beam(energy=beam_energy)
        fmap_config.traj = _fmaptrack.Trajectory(
            beam=fmap_config.beam,
            fieldmap=fmap_config.fmap,
            not_raise_range_exceptions=fmap_config.not_raise_range_exceptions)

    @staticmethod
    def _update_radia_model_energy(radia_model_config, beam_energy):
        if not radia_model_config:
            return
        radia_model_config.beam_energy = beam_energy
        radia_model_config.beam = _fmaptrack.Beam(energy=beam_energy)
