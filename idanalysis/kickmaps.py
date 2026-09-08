#!/usr/bin/env python-sirius
"""IDKickMap class."""

import fieldmaptrack as _fmaptrack
import matplotlib.pyplot as _plt
import multiprocessing
from copy import deepcopy
import numpy as _np
from scipy.optimize import curve_fit as _curve_fit
from imaids.insertiondevice import InsertionDeviceModel as _IDModel

from . import utils as _utils


class IDKickMap:
    """ID KickMap and FieldMap."""

    DEF_AUTHOR = '# Author: FAC idanalysis.IDKickMap'
    DEF_BEAM_ENERGY = 3  # [GeV]
    DEF_RK_S_STEP = 0.2  # [mm]

    def __init__(self, kmap_fname=None, author=None):
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
        self.shift_on_axis = False
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
            beam_energy=self.beam_energy,
            rk_s_step=self.rk_s_step,
        )

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
            radia_model=value, rk_s_step=self.rk_s_step
        )

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
                self._radia_model_config, value
            )

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
        self,
        traj_init_rx,
        traj_init_ry,
        traj_init_px=0,
        traj_init_py=0,
        traj_init_rz=None,
        traj_rk_min_rz=None,
        rk_s_step=None,
        **kwargs,
    ):
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
        self, brho, posx, posy, cxx, cyy, cxy=0, cyx=0, verbose=False
    ):
        """Generate a linear kickmap based on Ellaume formalism"""
        self.posx = _np.array(posx)  # [m]
        self.posy = _np.array(posy)  # [m]
        self.kickx = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.kicky = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.fposx = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.fposy = _np.full((len(self.posy), len(self.posx)), _np.inf)
        for i, ryi in enumerate(self.posy):
            for j, rxi in enumerate(self.posx):
                pxf = cxx * rxi + cxy * ryi
                pyf = cyx * rxi + cyy * ryi
                stg = 'rx = {:.01f} mm, ry = {:.01f}: '.format(
                    rxi * 1e3, ryi * 1e3
                )
                stg += 'px = {:.01f} urad, py = {:.01f} urad'.format(
                    pxf * 1e6, pyf * 1e6
                )
                if verbose:
                    print(stg)
                self.kickx[i, j] = pxf * brho**2
                self.kicky[i, j] = pyf * brho**2

    @staticmethod
    def _calc_kickmap_mp(args):
        (rxi, ryi, config) = args
        config.traj_init_rx = rxi
        config.traj_init_ry = ryi
        IDKickMap._fmap_calc_traj(config)
        pxf = config.traj.px[-1]
        pyf = config.traj.py[-1]
        rxf = config.traj.rx[-1]
        ryf = config.traj.ry[-1]
        data = (pxf, pyf, rxf, ryf)
        return data

    def _addoutput_to_kickmap(self, i, j, pxf, pyf, rxf, ryf):
        brho = self.brho
        rxi, ryi = self.posx[j], self.posy[i]
        stg = 'rx = {:.01f} mm, ry = {:.01f}: '.format(rxi * 1e3, ryi * 1e3)
        stg += 'px = {:.01f} urad, py = {:.01f} urad'.format(
            pxf * 1e6, pyf * 1e6
        )
        print(stg)
        self.kickx[i, j] = pxf * brho**2
        self.kicky[i, j] = pyf * brho**2
        self.fposx[i, j] = rxf / 1e3
        self.fposy[i, j] = ryf / 1e3

    def fmap_calc_kickmap(
        self, posx, posy, beam_energy=None, rk_s_step=None, parallelize=False
    ):
        """."""
        self.posx = _np.array(posx)  # [m]
        self.posy = _np.array(posy)  # [m]

        if beam_energy is not None:
            self.beam_energy = beam_energy
        if rk_s_step is not None:
            self.rk_s_step = rk_s_step
        self.kickx = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.kicky = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.fposx = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.fposy = _np.full((len(self.posy), len(self.posx)), _np.inf)
        config = self._fmap_config or self._radia_model_config
        self._config = config
        if parallelize:
            arglist = []
            for ryi in self.posy:
                for rxi in self.posx:
                    config = deepcopy(self._config)
                    arglist += [(1e3 * rxi, 1e3 * ryi, config)]
            num_processes = multiprocessing.cpu_count()
            data = []
            with multiprocessing.Pool(processes=num_processes - 1) as parallel:
                data = parallel.map(self._calc_kickmap_mp, arglist)
            for i, _ in enumerate(self.posy):
                for j, _ in enumerate(self.posx):
                    output = data[i * len(self.posx) + j]
                    pxf, pyf, rxf, ryf = output
                    self._addoutput_to_kickmap(i, j, pxf, pyf, rxf, ryf)
        else:
            for i, ryi in enumerate(self.posy):
                for j, rxi in enumerate(self.posx):
                    IDKickMap._fmap_calc_traj(self._config)
                    pxf, pyf, rxf, ryf = self._calc_kickmap_mp(
                        1e3 * rxi, 1e3 * ryi, self._config
                    )
                    self._addoutput_to_kickmap(i, j, pxf, pyf, rxf, ryf)

    def filter_kmap(self, posx=None, posy=None, order=5, plot_flag=False):
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
                label = 'y = {:.2f} mm'.format(1e3 * ryi)
                if plot_flag:
                    _plt.plot(
                        1e3 * self.posx,
                        1e6 * self.kickx[i, :],
                        '.',
                        label=label,
                    )
                    _plt.plot(1e3 * posx, 1e6 * kickx[i, :])
            if plot_flag:
                _plt.xlabel('x pos [mm]')
                _plt.ylabel('kicks x [Tm²]')
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
                label = 'y = {:.2f} mm'.format(1e3 * ryi)
                if plot_flag:
                    _plt.plot(
                        1e3 * self.posx,
                        1e6 * self.kicky[i, :],
                        '.',
                        label=label,
                    )
                    _plt.plot(1e3 * posx, 1e6 * kicky[i, :])
            if plot_flag:
                _plt.xlabel('x pos [mm]')
                _plt.ylabel('kicks y [Tm²]')
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
                label = 'x = {:.2f} mm'.format(1e3 * rxi)
                if plot_flag:
                    _plt.plot(
                        1e3 * self.posy,
                        1e6 * self.kickx[:, i],
                        '.',
                        label=label,
                    )
                    _plt.plot(1e3 * posy, 1e6 * kickx[:, i])
            if plot_flag:
                _plt.xlabel('y pos [mm]')
                _plt.ylabel('kicks x [Tm²]')
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
                label = 'x = {:.2f} mm'.format(1e3 * rxi)
                if plot_flag:
                    _plt.plot(
                        1e3 * self.posy,
                        1e6 * self.kicky[:, i],
                        '.',
                        label=label,
                    )
                    _plt.plot(1e3 * posy, 1e6 * kicky[:, i])
            if plot_flag:
                _plt.xlabel('y pos [mm]')
                _plt.ylabel('kicks y [Tm²]')
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
        my_file = open(kickmap_filename, 'w')  # w=writing
        my_file.write(rst)
        my_file.close()

    def calc_KsL_kickx_at_x(self, ix, plot=True):
        """."""
        posy = self.posy  # [m]
        posx = self.posx[ix]
        kickx = self.kickx[:, ix] / self.brho**2  # [rad]
        poly = _np.polyfit(posy, kickx, len(posy) - 5)
        if plot:
            kickx_fit = _np.polyval(poly, posy)
            _plt.clf()
            _plt.plot(1e3 * posy, 1e6 * kickx, 'o', label='data')
            _plt.plot(1e3 * posy, 1e6 * kickx_fit, label='fit')
            _plt.xlabel('posy [mm]')
            _plt.ylabel('kickx [urad]')
            _plt.title('Kickx @ x = {:.1f} mm'.format(1e3 * posx))
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
        poly = _np.polyfit(posx, kicky, len(posx) - 5)
        if plot:
            kicky_fit = _np.polyval(poly, posx)
            _plt.clf()
            _plt.plot(1e3 * posx, 1e6 * kicky, 'o', label='data')
            _plt.plot(1e3 * posx, 1e6 * kicky_fit, label='fit')
            _plt.xlabel('posx [mm]')
            _plt.ylabel('kicky [urad]')
            _plt.title('Kicky @ y = {:.1f} mm'.format(1e3 * posy))
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
        self, period_len=None, kmap_idlen=None, plot_flag=False
    ):
        """."""
        # get parameters
        kmap_idlen = kmap_idlen or self.kmap_idlen
        self.kmap_idlen = kmap_idlen
        period_len = period_len or self.period_len
        self.period_len = period_len
        nr_central_periods = int(kmap_idlen * 1e3 / period_len) - 4

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
            rz, rz_center - period_len * nr_central_periods / 2
        )
        idx_end_fit = self._find_value_idx(
            rz, rz_center + period_len * nr_central_periods / 2
        )

        for idx, pxy in enumerate([px, py]):
            rz_sample = rz[idx_begin_fit:idx_end_fit]
            p_sample = pxy[idx_begin_fit:idx_end_fit]
            opt = self.find_fit(rz_sample, p_sample)
            idx_begin_ID = self._find_value_idx(rz, -kmap_idlen * 1e3 / 2)
            idx_end_ID = self._find_value_idx(rz, +kmap_idlen * 1e3 / 2)
            linefit = self._linear_function(rz, opt[2], opt[3])
            kick_begin = linefit[idx_begin_ID] - pxy[0]
            kick_end = pxy[-1] - linefit[idx_end_ID]
            if plot_flag:
                _plt.plot(rz, pxy)
                _plt.plot(rz_sample, p_sample, '.')
                _plt.plot(rz, linefit)
                _plt.show()
            if idx == 0:
                self.kickx_upstream = kick_begin * self.brho**2
                self.kickx_downstream = kick_end * self.brho**2
                print('ID length: {:.3f} m'.format(kmap_idlen))
                print(
                    'kickx_upstream: {:11.4e}  T2m2'.format(
                        self.kickx_upstream
                    )
                )
                print(
                    'kickx_downstream: {:11.4e}  T2m2'.format(
                        self.kickx_downstream
                    )
                )
            elif idx == 1:
                self.kicky_upstream = kick_begin * self.brho**2
                self.kicky_downstream = kick_end * self.brho**2
                print(
                    'kicky_upstream: {:11.4e}  T2m2'.format(
                        self.kicky_upstream
                    )
                )
                print(
                    'kicky_downstream: {:11.4e}  T2m2'.format(
                        self.kicky_downstream
                    )
                )

    def plot_kickx_vs_posy(self, indx, title=''):
        """."""
        posx = self.posx
        posy = self.posy
        kickx = self.kickx / self.brho**2
        colors = _plt.cm.jet(_np.linspace(0, 1, len(indx)))
        _plt.figure(figsize=(8, 6))
        for c, ix in enumerate(indx):
            x = posx[ix]
            _plt.plot(1e3 * posy, 1e6 * kickx[:, ix], '-', color=colors[c])
            _plt.plot(
                1e3 * posy,
                1e6 * kickx[:, ix],
                'o',
                color=colors[c],
                label='posx = {:+.1f} mm'.format(1e3 * x),
            )
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
            _plt.plot(1e3 * posx, 1e6 * kicky[iy, :], '-', color=colors[c])
            _plt.plot(
                1e3 * posx,
                1e6 * kicky[iy, :],
                'o',
                color=colors[c],
                label='posy = {:+.1f} mm'.format(1e3 * y),
            )
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
        f = amp1 * _np.sin(2 * _np.pi / self.period_len * rz + phi1)
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

        if self.shift_on_axis:
            # find indices of central line
            try:
                indx = list(self.posx).index(0)
                indy = list(self.posy).index(0)
            except ValueError:
                raise ValueError(
                    'Kickmap does not have central transverse line!'
                )
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
        rst += '\n# '
        id_len = self.kmap_idlen or self.fmap_idlen
        rst += '\n# Total Length of Longitudinal Interval [m]'
        rst += '\n{}'.format(id_len)
        rst += '\n# Number of Horizontal Points'
        rst += '\n{}'.format(len(self.posx))
        rst += '\n# Number of Vertical Points'
        rst += '\n{}'.format(len(self.posy))

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
                rst += '{:+11.4e} '.format(self.kickx[-i - 1, j])

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
                rst += '{:+11.4e} '.format(self.kicky[-i - 1, j])

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
                rst += '{:+11.4e} '.format(self.fposx[-i - 1, j])

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
                rst += '{:+11.4e} '.format(self.fposy[-i - 1, j])
        return rst

    @staticmethod
    def _linear_function(x, a, b):
        return a * x + b

    @staticmethod
    def _find_value_idx(data, value):
        diff_array = _np.absolute(data - value)
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

        kickx = tables[0 * nrpts_y : 1 * nrpts_y, :]
        kicky = tables[1 * nrpts_y : 2 * nrpts_y, :]
        fposx = tables[2 * nrpts_y : 3 * nrpts_y, :]
        fposy = tables[3 * nrpts_y : 4 * nrpts_y, :]
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
            not_raise_range_exceptions=config.not_raise_range_exceptions,
        )

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
            not_raise_range_exceptions=config.not_raise_range_exceptions,
        )
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
            rk_min_rz = -1 * config.traj_init_rz
        else:
            rk_min_rz = config.traj_init_rz
        config.traj.calc_trajectory(
            init_rx=init_rx,
            init_ry=init_ry,
            init_rz=init_rz,
            init_px=init_px,
            init_py=init_py,
            init_pz=init_pz,
            s_step=config.traj_rk_s_step,
            s_length=config.traj_rk_length,
            s_nrpts=config.traj_rk_nrpts,
            min_rz=rk_min_rz,
            force_midplane=config.traj_force_midplane_flag,
            **kwargs,
        )

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
            skew_field_fitting_monomials=multi_skew,
        )
        config.multipoles.calc_multipoles(is_ref_trajectory_flag=False)
        config.multipoles.calc_multipoles_integrals()
        config.multipoles.calc_multipoles_integrals_relative(
            config.multipoles.normal_multipoles_integral,
            main_monomial=0,
            r0=config.multipoles_r0,
            is_skew=False,
        )

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
            config.multipoles.effective_length = (
                config.multipoles.normal_multipoles_integral[idx_n]
                / hardedge_field
            )
        else:
            idx_z = list(config.traj.s).index(0.0)
            main_multipole_center = config.multipoles.normal_multipoles[
                idx_n, idx_z
            ]
            config.multipoles.effective_length = (
                config.multipoles.normal_multipoles_integral[idx_n]
                / main_multipole_center
            )

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
            not_raise_range_exceptions=fmap_config.not_raise_range_exceptions,
        )

    @staticmethod
    def _update_radia_model_energy(radia_model_config, beam_energy):
        if not radia_model_config:
            return
        radia_model_config.beam_energy = beam_energy
        radia_model_config.beam = _fmaptrack.Beam(energy=beam_energy)


class EllaumeKickMap:
    """Class to generante kickmaps from Ellaume formalism."""

    def __init__(self, fieldsource, kmap_fname=None, author=None):
        """."""
        self._kmap_fname = kmap_fname
        self.fmap_idlen = None  # [m]
        self.kmap_idlen = None  # [m]
        self.posx = None  # [m]
        self.posy = None  # [m]
        self.posx_fit = None  # [m]
        self.posy_fit = None  # [m]
        self.kickx = None  # [T².m²]
        self.kicky = None  # [T².m²]
        self.period_len = None  # [mm]
        self._fieldsource_type = None
        self.potential = None
        self.matrix_poly = None
        self.fit_coefs = None
        self.potential_fit = None
        self.author = author or IDKickMap.DEF_AUTHOR
        self.fieldsource = fieldsource
        self.brho = _fmaptrack.Beam(energy=3).brho  # [Tm]

    @property
    def fieldsource(self):
        return self._fieldsource

    @fieldsource.setter
    def fieldsource(self, value):
        self._fieldsource = value
        if isinstance(value, _fmaptrack.FieldMap):
            self._fieldsource_type = 'Fieldmap'
        elif isinstance(value, _IDModel):
            self._fieldsource_type = 'RADIA'
        else:
            raise ValueError('Invalid fieldsource')

    def get_field_at_xy(self, x, y, rz):
        if self._fieldsource_type == 'RADIA':
            b = self.fieldsource.get_field(x, y, rz)
            return b[:, 0], b[:, 1]
        else:
            idx = _np.argwhere(self.fieldsource.rx == x)[0][0]
            idy = _np.argwhere(self.fieldsource.ry == y)[0][0]
            return self.fieldsource.bx[idy, idx, :], self.fieldsource.by[
                idy, idx, :
            ]

    def get_oneperiod_field(self, x, y, rz, period):
        bx, by = self.get_field_at_xy(x, y, rz)
        if self._fieldsource_type == 'Fieldmap':
            rz = self.fieldsource.rz
        idx_begin = _np.argmin(_np.abs(rz + period / 2))
        idx_end = _np.argmin(_np.abs(rz - period / 2))
        z = rz[idx_begin:idx_end]
        bx = bx[idx_begin:idx_end]
        by = by[idx_begin:idx_end]
        return z, bx, by

    def fit_fourier_coefs(self, z, b, period, nr_harms):
        modes_matrix = _np.zeros((len(z), 2 * nr_harms))
        ks = _np.arange(
            2 * _np.pi / period,
            (nr_harms + 1) * 2 * _np.pi / period,
            2 * _np.pi / period,
        )
        for i, k in enumerate(ks):
            modes_matrix[:, 2 * i] = _np.cos(k * z)
            modes_matrix[:, 2 * i + 1] = _np.sin(k * z)
        invmat = _np.linalg.pinv(modes_matrix)
        coefs = invmat @ b
        return coefs, modes_matrix

    def get_field_amps(self, coefs, nr_harms):
        coefs = _np.reshape(coefs, (nr_harms, 2))
        field_amps = _np.sqrt(_np.sum(coefs**2, axis=1))
        return field_amps

    def calc_kickmap_potential_at_xy(
        self, x, y, rz, nr_periods, period, nr_harms
    ):
        z, bx, by = self.get_oneperiod_field(x, y, rz, period)
        coefs_by, _ = self.fit_fourier_coefs(z, by, period, nr_harms)
        coefs_bx, _ = self.fit_fourier_coefs(z, bx, period, nr_harms)
        period *= 1e-3
        by_amps = self.get_field_amps(coefs_by, nr_harms)
        bx_amps = self.get_field_amps(coefs_bx, nr_harms)
        n = _np.arange(1, nr_harms + 1, 1)
        by_amps_n = by_amps / n
        bx_amps_n = bx_amps / n
        phi = (
            nr_periods
            * (period / 2)
            * (period / (2 * _np.pi)) ** 2
            * _np.sum(by_amps_n**2 + bx_amps_n**2)
        )
        return phi

    def calc_full_potential(self, nr_periods, period, nr_harms, rz):
        if self.posx is None or self.posy is None:
            raise ValueError('posx and posy must be set before calculating potential.')
        rx = 1e3*self.posx  # convert [m] to [mm]
        ry = 1e3*self.posy  # convert [m] to [mm]
        potential = _np.zeros((len(rx), len(ry)))
        for i, x in enumerate(rx):
            for j, y in enumerate(ry):
                x_ = round(x, 10)
                y_ = round(y, 10)
                progress = 100 * (i * len(ry) + j + 1) / (len(rx) * len(ry))
                print(
                    'Calculating potential... Progress: {:.2f}%'.format(progress),
                    end='\r',
                    flush=True)
                potential[i, j] = self.calc_kickmap_potential_at_xy(x_, y_, rz, nr_periods, period, nr_harms)
        self.potential = potential
        return potential

    def plot_potential(self, fitted=False):
        if fitted:
            if self.potential_fit is None:
                raise ValueError('Fitted potential has not been calculated yet. Call calc_full_potential first.')
            X, Y = _np.meshgrid(self.posx_fit, self.posy_fit)
            potential = self.potential_fit
        else:
            if self.potential is None:
                raise ValueError('Potential has not been calculated yet. Call calc_full_potential first.')
            X, Y = _np.meshgrid(self.posx, self.posy)
            potential = self.potential

        fig = _plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(
            1e3*X,
            1e3*Y,
            potential.T,
            edgecolor='none',
            antialiased=True,
        )

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel(r'$K_x$ [$T^2\,m^3$]')
        ax.set_title('Kickmap potential')
        ax.view_init(elev=30, azim=-30)
        _plt.tight_layout()
        _plt.show()

    def calc_2d_polyfit_matrix(self, degree, posx=None, posy=None):
        if posx is None or posy is None:
            if self.posx is None or self.posy is None:
                raise ValueError('posx and posy must be set before calculating potential.')
            else:
                posx = 1e3*self.posx  # convert [m] to [mm]
                posy = 1e3*self.posy  # convert [m] to [mm]
        n = degree + 1
        nr_coefs = int(n*(n+1)/2)
        matrix = _np.zeros((len(posx)*len(posy), nr_coefs))
        y_vec = _np.tile(posy, len(posx))
        x_vec = _np.repeat(posx, len(posy))
        idx = 0
        for i in _np.arange(degree + 1):
            for j in _np.arange(degree + 1 -i):
                matrix[:, idx] = (x_vec**i) * (y_vec**j)
                idx += 1
        return matrix

    def fit_2d_polyfit(self, degree):
        if self.posx is None or self.posy is None:
            raise ValueError('posx and posy must be set before calculating potential.')
        if self.potential is None:
            raise ValueError('Potential has not been calculated yet. Call calc_full_potential first.')
        potential = self.potential
        posx = self.posx
        posy = self.posy
        matrix = self.calc_2d_polyfit_matrix(degree)
        invmat = _np.linalg.pinv(matrix)
        pot_vec = _np.reshape(potential, len(posx)*len(posy), order='C')
        coefs = _np.dot(invmat, pot_vec)
        potential_fit = _np.reshape(_np.dot(matrix, coefs), (len(posx), len(posy)), order='C')
        residue = _np.sqrt(_np.sum(potential_fit-potential)**2)
        self.matrix_poly = matrix
        self.fit_coefs = coefs
        return coefs, residue

    def calc_potential_fit(self, degree):
        if self.posx_fit is None or self.posy_fit is None:
            raise ValueError('posx_fit and posy_fit must be set before calculating potential.')
        posx = 1e3*self.posx_fit  # convert [m] to [mm]
        posy = 1e3*self.posy_fit  # convert [m] to [mm]
        coefs = self.fit_coefs
        matrix = self.calc_2d_polyfit_matrix(degree, posx, posy)
        potential_fit = _np.reshape(_np.dot(matrix, coefs), (len(posx), len(posy)), order='C')
        self.matrix_poly = matrix
        self.potential_fit = potential_fit
        return potential_fit

    def calc_dy_operator(self):
        if self.matrix_poly is None:
            raise ValueError('Polynomial matrix has not been calculated yet.')
        matrix = self.matrix_poly
        deg = int((-1 + _np.sqrt(1+8*matrix.shape[1]))/2) - 1
        dy_operator = _np.zeros((matrix.shape[1], matrix.shape[1]))
        vec = _np.zeros((matrix.shape[1]))
        vec[0] = 1
        count = 0
        ref_idx = 0
        amp = 1
        for i in range(matrix.shape[1]-2):
            j = i + 1
            vec = amp*vec/_np.linalg.norm(vec)
            amp += 1
            dy_operator[:, j] = vec
            if (j - ref_idx) == deg + 1 - count:
                ref_idx = j
                count += 1
                amp = 1
                dy_operator[:, j] = 0
            vec = _np.roll(vec, 1)
        return dy_operator

    def calc_dx_operator(self):
        if self.matrix_poly is None:
            raise ValueError('Polynomial matrix has not been calculated yet.')
        matrix = self.matrix_poly
        deg = int((-1 + _np.sqrt(1+8*matrix.shape[1]))/2) - 1
        dx_operator = _np.zeros((matrix.shape[1], matrix.shape[1]))
        vec = _np.zeros((matrix.shape[1]))
        vec[0] = 1
        count = 0
        ref_idx = 0
        for i in range(matrix.shape[1]-deg-1):
            j = i + deg + 1
            if (i+1 - ref_idx) == deg + 1 - count:
                ref_idx = i
                vec = _np.roll(vec, 1)
                count += 1
                vec = (count+1)*vec/_np.linalg.norm(vec)
            dx_operator[:, j] = vec
            vec = _np.roll(vec, 1)
        return dx_operator

    def calc_dely(self):
        dy = self.calc_dy_operator()
        potential_fit = self.potential_fit
        coefs = self.fit_coefs
        matrix = self.matrix_poly
        coefs_dy = _np.dot(dy, coefs)
        dp_dy = _np.dot(matrix, coefs_dy)
        dp_dy = _np.reshape(dp_dy, potential_fit.shape, order='C')
        return dp_dy

    def calc_delx(self):
        dx = self.calc_dx_operator()
        potential_fit = self.potential_fit
        coefs = self.fit_coefs
        matrix = self.matrix_poly
        coefs_dx = _np.dot(dx, coefs)
        dp_dx = _np.dot(matrix, coefs_dx)
        dp_dx = _np.reshape(dp_dx, potential_fit.shape, order='C')
        return dp_dx

    def calc_kicks(self):
        brho = self.brho
        dp_dx = 1e3 * self.calc_delx()
        dp_dy = 1e3 * self.calc_dely()
        kicksx = -1/2 * dp_dx.T
        kicksy = -1/2 * dp_dy.T
        self.kickx = kicksx
        self.kicky = kicksy
        return kicksx/brho**2, kicksy/brho**2

    def plot_kicks(self, plane='x'):
        if self.kickx is None or self.kicky is None:
            raise ValueError('Kicks have not been calculated yet.')
        if plane.lower() == 'x':
            kick = self.kickx/self.brho**2
        elif plane.lower() == 'y':
            kick = self.kicky/self.brho**2
        else:
            raise ValueError('Invalid plane value.')
        X, Y = _np.meshgrid(self.posx_fit, self.posy_fit)
        fig = _plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(
            1e3*X,
            1e3*Y,
            1e6*kick,
            edgecolor='none',
            antialiased=True,
        )

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('kick [urad]')
        ax.set_title('Kick ' + plane.lower())
        ax.view_init(elev=30, azim=-30)
        _plt.tight_layout()
        _plt.show()
