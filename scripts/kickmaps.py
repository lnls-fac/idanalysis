#!/usr/bin/env python-sirius
"""IDKickmap class."""


import numpy as _np
import matplotlib.pyplot as _plt

import fieldmaptrack as _fmaptrack
from utils import create_deltadata as _create_deltadata


class IDKickMap:
    """."""

    DEFAULT_AUTHOR = '# Author: FAC fieldmaptrack.IDKickMap'

    def __init__(self, fname=None, author=None):
        """."""
        self.id_length = None  # [m]
        self.posx = None  # [m]
        self.posy = None  # [m]
        self.kickx = None  # [T².m²]
        self.kicky = None  # [T².m²]
        self.fposx = None  # [m]
        self.fposy = None  # [m]
        self.fmap_config = None
        self.author = author or IDKickMap.DEFAULT_AUTHOR
        self.brho, *_ = _fmaptrack.Beam.calc_brho(energy=3.0)
        # load
        if fname:
            self.load(fname)

    def load(self, fname):
        """."""
        self.fname = fname
        info = IDKickMap._load(self.fname)
        self.id_length = info['id_length']
        self.posx, self.posy = info['posx'], info['posy']
        self.kickx, self.kicky = info['kickx'], info['kicky']
        self.fposx, self.fposy = info['fposx'], info['fposy']

    def get_deltakickmap(self, idx):
        """."""
        configs = _create_deltadata()
        fname = configs.get_kickmap_filename(configs[idx])
        self.fname = fname
        self.load(fname)

    def fmap_calc_kickmap(
            self, fmap_fname, posx, posy, beam_energy=3.0, rk_s_step=0.2):
        """."""
        self.posx = _np.array(posx)  # [m]
        self.posy = _np.array(posy)  # [m]
        self.fmap_config = IDKickMap._create_fmap_config(
            fmap_fname, beam_energy=beam_energy, rk_s_step=rk_s_step)

        brho = self.fmap_config.beam.brho
        idlen = self.fmap_config.fmap.rz[-1] - self.fmap_config.fmap.rz[0]
        self.id_length = idlen/1e3
        self.kickx = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.kicky = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.fposx = _np.full((len(self.posy), len(self.posx)), _np.inf)
        self.fposy = _np.full((len(self.posy), len(self.posx)), _np.inf)
        for i, ryi in enumerate(self.posy):
            for j, rxi in enumerate(self.posx):
                self.fmap_config = IDKickMap._calc_trajectory(
                    self.fmap_config, init_rx=1e3*rxi, init_ry=1e3*ryi)
                pxf = self.fmap_config.traj.px[-1]
                pyf = self.fmap_config.traj.py[-1]
                rxf = self.fmap_config.traj.rx[-1]
                ryf = self.fmap_config.traj.ry[-1]
                stg = 'rx = {:.01f} mm, ry = {:.01f}: '.format(
                    rxi*1e3, ryi*1e3)
                stg += 'px = {:.01f} urad, py = {:.01f} urad'.format(
                    pxf*1e6, pyf*1e6)
                print(stg)
                self.kickx[i, j] = pxf * brho * brho
                self.kicky[i, j] = pyf * brho * brho
                self.fposx[i, j] = rxf / 1e3
                self.fposy[i, j] = ryf / 1e3

    @staticmethod
    def fmap_calc_trajectory(
            fmap_fname, init_rx, init_ry, init_px=0, init_py=0,
            beam_energy=3.0, rk_s_step=0.2):
        """."""
        fmap_config = IDKickMap._create_fmap_config(
                fmap_fname, beam_energy=beam_energy, rk_s_step=rk_s_step)
        fmap_config = IDKickMap._calc_trajectory(
            fmap_config, init_rx, init_ry, init_px, init_py)
        return fmap_config

    def __str__(self):
        """."""
        rst = ''
        # header
        rst += self.author
        rst += '\n# '
        rst += '\n# Total Length of Longitudinal Interval [m]'
        rst += '\n{}'.format(self.id_length)
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
                rst += '{:+11.4e} '.format(self.kickx[-i-1, j])

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
                rst += '{:+11.4e} '.format(self.kicky[-i-1, j])

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

    def plot_KsL_kickx(self, config_ind, grad, title):
        """."""
        _plt.clf()
        for idx in config_ind:
            self.get_deltakickmap(idx)
            posx, ksl = self.calc_KsL_kickx()
            _plt.plot(1e3*posx, ksl, color='C'+str(idx))
        _plt.hlines(
            [-0.1, 0.1],
            xmin=min(1e3*posx), xmax=max(1e3*posx),
            linestyles='--', label='skewquad spec.: 0.1 m⁻¹')
        _plt.xlabel('posx [mm]')
        _plt.ylabel('KsL (L * ' + grad + r' / B$\rho$ @ x=0) [m⁻¹]')
        _plt.grid()
        _plt.legend()
        _plt.title(title)
        title = title.replace('(', '_').replace(')', '').replace(' ', '')
        title += '.png'
        _plt.savefig(title)

    def plot_KsL_kicky(self, config_ind, grad, title):
        """."""
        _plt.clf()
        for idx in config_ind:
            self.get_deltakickmap(idx)
            posy, ksl = self.calc_KsL_kicky()
            _plt.plot(1e3*posy, ksl, color='C'+str(idx))

        _plt.hlines(
            [-0.1, 0.1],
            xmin=min(1e3*posy), xmax=max(1e3*posy),
            linestyles='--', label='skewquad spec.: 0.1 m⁻¹')
        _plt.xlabel('posy [mm]')
        _plt.ylabel('KsL (L * ' + grad + r' / B$\rho$ @ x=0) [m⁻¹]')
        _plt.grid()
        _plt.legend()
        _plt.title(title)
        title = title.replace('(', '_').replace(')', '').replace(' ', '')
        title += '.png'
        _plt.savefig(title)

    def plot_all(self):
        """."""
        titles = [
            'linH_kzero', 'linH_kmid', 'linH_kmax',
            'cirLH_kzero', 'cirLH_kmid', 'cirLH_kmax',
            'linV_kzero', 'linV_kmid', 'linV_kmax',
            ]

        for i, title in enumerate(titles):
            self.plot_KsL_kickx(
                i*10 + _np.arange(10), 'dBy/dy', title + ' (kickx)')
            self.plot_KsL_kicky(
                i*10 + _np.arange(10), 'dBx/dx', title + ' (kicky)')

    def plot_examples(self):
        """."""
        self.get_deltakickmap(idx=0)
        self.calc_KsL_kickx_at_x(ix=14, plot=True)
        self.calc_KsL_kicky_at_y(iy=8, plot=True)

    @staticmethod
    def _load(fname):
        """."""
        with open(fname) as fp:
            lines = fp.readlines()

        tables = []
        params = []
        for line in lines:
            line = line.strip()
            if line.startswith('START'):
                pass
            elif line.startswith('#'):
                pass
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
        return info

    @staticmethod
    def _create_fmap_config(fmap_fname, beam_energy, rk_s_step):
        config = _fmaptrack.common_analysis.Config()
        config.config_label = 'id-3gev'
        config.magnet_type = 'insertion-device'  # not necessary
        config.fmap_filename = fmap_fname
        config.fmap_extrapolation_flag = False
        config.not_raise_range_exceptions = False
        config.interactive_mode = False
        config.not_raise_range_exceptions = True

        transforms = dict()
        config.fmap = _fmaptrack.FieldMap(
            config.fmap_filename,
            transforms=transforms,
            not_raise_range_exceptions=config.not_raise_range_exceptions)

        config.traj_load_filename = None
        config.traj_is_reference_traj = True
        config.traj_init_rz = min(config.fmap.rz)

        config.traj_rk_s_step = rk_s_step
        config.traj_rk_length = None
        config.traj_rk_nrpts = None
        config.traj_force_midplane_flag = False
        config.interactive_mode = True

        config.beam_energy = beam_energy

        config.beam = _fmaptrack.Beam(energy=config.beam_energy)
        config.traj = _fmaptrack.Trajectory(
            beam=config.beam,
            fieldmap=config.fmap,
            not_raise_range_exceptions=config.not_raise_range_exceptions)
        config.traj_init_rz = min(config.fmap.rz)

        return config

    @staticmethod
    def _calc_trajectory(
            config, init_rx=0, init_ry=0, init_px=0, init_py=0):
        """."""
        config.traj_init_rx = init_rx
        config.traj_init_ry = init_ry
        config.traj_init_px = init_px
        config.traj_init_py = init_py

        # analysis = _fmaptrack.common_analysis.get_analysis_symbol(
        #     config.magnet_type)
        config = IDKickMap.trajectory_analysis(config)
        return config

    @staticmethod
    def trajectory_analysis(config):
        """Trajectory analysis."""
        if config.traj_load_filename is not None:
            # loads trajectory from file
            config.beam = _fmaptrack.Beam(energy=config.beam_energy)
            config.traj = _fmaptrack.Trajectory(
                beam=config.beam,
                fieldmap=config.fmap,
                not_raise_range_exceptions=config.not_raise_range_exceptions)
            config.traj.load(config.traj_load_filename)
            config.traj_init_rz = config.traj.rz[0]
        else:
            config.beam = _fmaptrack.Beam(energy=config.beam_energy)
            config = IDKickMap.calc_trajectory_fieldmaptrack(config)

        # prints basic information on the reference trajectory
        # ====================================================
        if not config.interactive_mode:
            print('--- trajectory (rz > {0} mm) ---'.format(
                config.traj_init_rz))
            print(config.traj)

        if not config.interactive_mode:
            # saves trajectory in file
            config.traj.save(filename='trajectory.txt')
            # saves field on trajectory in file
            config.traj.save_field(filename='field_on_trajectory.txt')
        return config

    @staticmethod
    def calc_trajectory_fieldmaptrack(config):
        """Calcs trajectory."""
        config.traj = _fmaptrack.Trajectory(
            beam=config.beam,
            fieldmap=config.fmap,
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
            init_px = config.traj_init_px * (_np.pi/180.0)
        else:
            config.traj_init_px = init_px = 0.0
        if hasattr(config, 'traj_init_py'):
            init_py = config.traj_init_py * (_np.pi/180.0)
        else:
            config.traj_init_py = init_py = 0.0
        init_pz = _np.sqrt(1.0 - init_px**2 - init_py**2)
        if config.traj_rk_s_step > 0.0:
            rk_min_rz = max(config.fmap.rz)
        else:
            rk_min_rz = min(config.fmap.rz)
        # rk_min_rz = config.fmap.rz[-1]
        config.traj.calc_trajectory(
            init_rx=init_rx, init_ry=init_ry, init_rz=init_rz,
            init_px=init_px, init_py=init_py, init_pz=init_pz,
            s_step=config.traj_rk_s_step,
            s_length=config.traj_rk_length,
            s_nrpts=config.traj_rk_nrpts,
            min_rz=rk_min_rz,
            force_midplane=config.traj_force_midplane_flag)
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


if __name__ == '__main__':
    idkickmap = IDKickMap()
    idkickmap.plot_examples()
