"""ID Trajectories."""

import pickle as _pickle
import numpy as _np
import matplotlib.pyplot as _plt

from fieldmaptrack.beam import Beam as _Beam
from fieldmaptrack.track import Trajectory as _Trajectory

class IDTrajectory(_Trajectory):
    """."""

    KICKS_POS_SABIA = [
        -784,  # from straight center [mm]
        +784,  # from straight center [mm]
        ]

    KICKS_DELTA = 0.5e-6  # [rad]

    def __init__(self, label, fieldmap, output_folder=None, energy=3):
        """."""
        self._label = label
        self._output_folder = '' if output_folder is None else output_folder
        beam=_Beam(energy=energy)
        _Trajectory.__init__(self, beam=beam, fieldmap=fieldmap)
    
        self._init_rz = min(min(fieldmap.rz), IDTrajectory.KICKS_POS_SABIA[0])
        self._min_rz = max(max(fieldmap.rz), IDTrajectory.KICKS_POS_SABIA[1])
        self._s_step = 0.1
        self._kicks = [
            IDTrajectory.KICKS_POS_SABIA, # rz location of kicks [mm]
            [0, 0],  # Horizontal up and down stream kicks [rad]
            [0, 0],  # Vertical up and down stream kicks [rad]
            ]

    @property
    def output_folder(self):
        """."""
        return self._output_folder

    @output_folder.setter
    def output_folder(self, value):
        """."""
        if value:
            value = value if value[-1] == '/' else value + '/'
        self._output_folder = value

    @property
    def init_rz(self):
        """."""
        return self._init_rz

    @init_rz.setter
    def init_rz(self, value):
        """."""
        self._init_rz = value

    @property
    def min_rz(self):
        """."""
        return self._min_rz

    @min_rz.setter
    def min_rz(self, value):
        """."""
        self._min_rz = value

    @property
    def s_step(self):
        """."""
        return self._s_step

    @s_step.setter
    def s_step(self, value):
        """."""
        self._s_step = value

    @property
    def kicks(self):
        """Kicks in trajectory.

        A list of the format:
        [[s1,s2],  # [mm]
         [kickX@s1, kickX@s2],  # [rad]
         [kickY@s1, kickY@s2],  # [rad]
        ]
        """
        return self._kicks

    @kicks.setter
    def kicks(self, value):
        """."""
        self._kicks = value

    @property
    def kickspos(self):
        """."""
        return self._kicks[0]

    @kickspos.setter
    def kickspos(self, value):
        """."""
        self._kicks[0] = value

    @property
    def hkicks(self):
        """."""
        return self._kicks[1]

    @hkicks.setter
    def hkicks(self, value):
        """."""
        self._kicks[1] = value

    @property
    def vkicks(self):
        """."""
        return self._kicks[2]

    @vkicks.setter
    def vkicks(self, value):
        """."""
        self._kicks[2] = value

    def calc_trajectory(self, s_step=None):
        """."""
        s_step = s_step or self._s_step

        super().calc_trajectory(
            init_rx=0.0, init_ry=0.0, init_rz=self._init_rz,
            init_px=0.0, init_py=0.0, init_pz=1.0,
            min_rz=self._min_rz, s_step=s_step, kicks=self._kicks)

    def correct_posang_init(self, s_step=None, plot=False, title=None):

        s_step = s_step or self._s_step

        self.hkicks = [0, 0]
        self.vkicks = [0, 0]
        spos = self.kickspos

        self.calc_trajectory(s_step=s_step)
        rz0 = _np.asarray(self.rz)

        # corrections based on
        #
        # k1 + k2 = (xf' - x0')
        # x2 + k1 * (s2 - s1) + k2 * (sf - s2) = 0

        # horizontal correction
        rx0, px0 = _np.asarray(self.rx), _np.asarray(self.px)
        sf = rz0[-1]
        p0, pf = px0[0], px0[-1]
        s1, s2 = spos
        _, r2 = _np.interp(spos, rz0, rx0)
        b = _np.array([p0 - pf, -r2])
        a = _np.array([[1, 1], [s2-s1, sf-s2]])
        kicks = _np.linalg.solve(a, b)
        self.hkicks = kicks

        # vertical correction
        ry0, py0 = _np.asarray(self.ry), _np.asarray(self.py)
        sf = rz0[-1]
        p0, pf = py0[0], py0[-1]
        s1, s2 = spos
        _, r2 = _np.interp(spos, rz0, ry0)
        b = _np.array([p0 - pf, -r2])
        a = _np.array([[1, 1], [s2-s1, sf-s2]])
        kicks = _np.linalg.solve(a, b)
        self.vkicks = kicks

        if plot:
            self.calc_trajectory(s_step=s_step)
            rz1 = _np.asarray(self.rz)
            rx1 = _np.asarray(self.rx)
            ry1 = _np.asarray(self.ry)
            print('kickx: {} urad'.format(1e6*self.hkicks))
            print('kicky: {} urad'.format(1e6*self.vkicks))
            _plt.plot(rz0, 1e3*rx0, 'o', color=[0.6,0.6,1.0], label='H uncorr')
            _plt.plot(rz0, 1e3*ry0, 'o', color=[1.0,0.6,0.6], label='V uncorr')
            _plt.plot(rz1, 1e3*rx1, 'o', color=[0,0,1.0], label='H corr')
            _plt.plot(rz1, 1e3*ry1, 'o', color=[1.0,0.0,0.0], label='V corr')
            _plt.xlabel('rz [mm]')
            _plt.ylabel('rx [um]')
            _plt.legend()
            _plt.title('Horizontal Trajectory')
            if title:
                _plt.title(title)
            _plt.grid()
            _plt.show()

    # --- soon-to-be-deprecated methods ---

    def calc_posang_residue(self, s_step=None):
        """."""
        self.calc_trajectory(s_step=s_step)
        res = _np.array(
            [self.rx[-1], self.px[-1], self.ry[-1], self.py[-1]])
        return res

    def calc_posang_respm(self, s_step=None, save=False):
        """."""
        s_step = s_step or self._s_step
        kicks = self._kicks

        delta_kick = IDTrajectory.KICKS_DELTA

        respm = _np.zeros((4, 4))
        for i in range(2):
            
            # horizontal
            kick0 = kicks[1][i]
            kicks[1][i] = kick0 + delta_kick/2
            r1 = self.calc_posang_residue(s_step=s_step)
            kicks[1][i] = kick0 - delta_kick/2
            r2 = self.calc_posang_residue(s_step=s_step)
            respm[:, 0+i] = (r1 - r2) / delta_kick
            kicks[1][i] = kick0

            # vertical
            kick0 = kicks[2][i]
            kicks[2][i] = kick0 + delta_kick/2
            r1 = self.calc_posang_residue(s_step=s_step)
            kicks[2][i] = kick0 - delta_kick/2
            r2 = self.calc_posang_residue(s_step=s_step)
            respm[:, 2+i] = (r1 - r2) / delta_kick
            kicks[2][i] = kick0

        if save:
            self.save_posang_respm(respm, s_step)
        
        return respm

    def correct_posang(self, s_step=None, respm=None):
        """."""
        if respm is None:
            # get posang respm from file 
            respm, s_step_ = self.load_posang_respm()
            if respm is None:
                # note: s_step in respm might be different
                respm = self.calc_posang_respm(s_step=s_step)
                self.save_posang_respm(s_step=s_step)
    
        # inverse response matrix
        umat, smat, vmat = _np.linalg.svd(respm, full_matrices=False)
        # print(smat)
        ismat = 1/smat
        ismat = _np.diag(ismat)
        invmat = -1 * _np.dot(_np.dot(vmat.T, ismat), umat.T)

        # calc kicks
        res0 = self.calc_posang_residue(s_step=s_step)
        dkicks = _np.dot(invmat, res0.flatten())

        # accumulate kick corrections   
        self._kicks[1] += dkicks[:2]
        self._kicks[2] += dkicks[2:]

        resf = self.calc_posang_residue(s_step=s_step)


        return resf, dkicks, res0

    def save_posang_respm(self, respm, s_step):
        """."""
        fname = self.output_folder + \
            'traj_posang_respm_' + self._label + '.pickle'
        data = dict(s_step=s_step, respm=respm)
        _pickle.dump(data, open(fname, 'wb'))

    def load_posang_respm(self):
        """."""
        fname = self.output_folder + \
            'traj_posang_respm_' + self._label + '.pickle'
        data = _pickle.load(open(fname, 'rb'))
        return data['respm'], data['s_step']

    def save_posang(self, s_step=None, label=None):
        """."""
        s_step = s_step or self._s_step
        fname = self._get_posang_filename(label)
        data = dict(
            s_step=s_step,
            kicks=self.kicks,
            rx=self.rx, ry=self.ry, rz=self.rz,
            px=self.px, py=self.py, pz=self.pz,
            )
        _pickle.dump(data, open(fname, 'wb'))

    def load_posang(self, label=None):
        """."""
        if label is None:
            label = ''
        fname = self._get_posang_filename(label)
        data = _pickle.load(open(fname, 'rb'))
        self._s_step = data['s_step']
        self.kicks = data['kicks']
        self.rx, self.px = data['rx'], data['px']
        self.ry, self.py = data['ry'], data['py']
        self.rz, self.pz = data['rz'], data['pz']

    def fit_parabola(self, plot=True, liminf=-500, limsup=+500):
        """."""
        sel = (self.rz > liminf) & (self.rz < limsup)
        rz = self.rz
        
        r_ = self.rx
        p = _np.polyfit(rz[sel], r_[sel], 2)
        fit = _np.polyval(p, rz[sel])
        idx = _np.argmax(_np.abs(fit))
        fit[idx]
        # fit(idx)
        # ry = self.ry
        _plt.plot(rz, r_)
        _plt.plot(rz[sel], fit)
        _plt.show()

    # val_mean = np.polyfit(traj1.rz[sel], (val_sel - val_fit)**2, 0)
    # amplitude = np.sqrt(2*val_mean[0])
    # val_max = val_fit[idx]

    def _get_posang_filename(self, label):
        if label in (None, ''):
            label == ''
        else:
            label = label if label[-1] == '_' else label + '_'
    
        fname = self._output_folder + \
            'traj_posang_' + label + self._label + '.pickle'
        
        return fname 
