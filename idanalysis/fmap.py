#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as _plt
from fieldmaptrack import FieldMap as _FieldMap
from fieldmaptrack import Beam as _Beam
from fieldmaptrack import Trajectory as _Trajectory
from scipy import integrate as _integrate

from . import utils as _utils

class EPUOnAxisFieldMap(_FieldMap):
    """."""
    class CONFIGS:
        """."""
        HP_G22P0 = 'HP_G22P0'
        HP_G25P7 = 'HP_G25P7'
        HP_G29P3 = 'HP_G29P3'
        HP_G40P9 = 'HP_G40P9'
        VP_G22P0_P = 'VP_G22P0_P'

    _DATA_PATH = (
            'ids-data/Ondulador UVV/Estrutura Magnética/Medidas e Resultados Ondulador/'
            'Operacao Ondulador  - Medidas Oficiais/')

    _CONFIG_FNAMES = {
        # NOTE: order is important! Bx, Bz, By (or -Bx, By, Bz in sirius coordsys)
        'HP_G22P0': [
            'Campo Vertical/Gap 22.0/Map0802_Bx_Gap22_Fase0.0_01_X=0_Real.dat',
            'Campo Vertical/Gap 22.0/Map0802_Bz_Gap22_Fase0.0_01_X=0_Real.dat',
            'Campo Vertical/Gap 22.0/Map1202_By_Gap22_Fase0.0_01_X=0_Real.dat',
            ],
        'HP_G25P7': [
            'Campo Vertical/Gap 25.7/Map1902_Bx_Gap25.7_Fase0.0_01_Real.dat',
            'Campo Vertical/Gap 25.7/Map1902_Bz_Gap25.7_Fase0.0_01_Real.dat',
            'Campo Vertical/Gap 25.7/Map1902_By_Gap25.7_Fase0.0_01_Real.dat',
            ],
        'HP_G29P3': [
            'Campo Vertical/Gap 29.3/Map1902_Bx_Gap29.3_Fase0.0_01_Real.dat',
            'Campo Vertical/Gap 29.3/Map1902_Bz_Gap29.3_Fase0.0_01_Real.dat',
            'Campo Vertical/Gap 29.3/Map1902_By_Gap29.3_Fase0.0_01_Real.dat',
            ],
        'HP_G40P9': [
            'Campo Vertical/Gap 40.9/Map1902_Bx_Gap40.9_Fase0.0_01_Real.dat',
            'Campo Vertical/Gap 40.9/Map1902_Bz_Gap40.9_Fase0.0_01_Real.dat',
            'Campo Vertical/Gap 40.9/Map1902_By_Gap40.9_Fase0.0_01_Real.dat',
            ],
        'VP_G22P0_P': [
            'Campo Horizontal/Gap 22.0/Fase 25.0/Map0902_Bx_Gap22_Fase25.0_01_X=0_Real.dat',
            'Campo Horizontal/Gap 22.0/Fase 25.0/Map0902_Bz_Gap22_Fase25.0_01_X=0_Real.dat',
            'Campo Horizontal/Gap 22.0/Fase 25.0/Map1302_By_Gap22.0_Fase25.0_01_X=0_Real.dat',
            ],
        }

    def __init__(self, folder,
        config, stepx=1.0, nrptsx=3, stepy=1.0, nrptsy=3, centralize=True,
        **kwargs):
        self._folder = folder
        self._config = config
        self._stepx = stepx
        self._nrptsx = nrptsx
        self._stepy = stepy
        self._nrptsy = nrptsy
        self._centralize = centralize
        content = self._get_fmap_content()
        super().__init__(content=content, **kwargs)

    @property
    def folder(self):
        return self._folder

    @staticmethod
    def _read_field_component(file_name):
        my_file = open(file_name)
        data_col1 = []
        data_col2 = []
        col1_values = []
        col2_values = []
        for line in my_file:
            list_data = line.split('\t') #returns a list
            if list_data[0] !='\n':
                try:
                    data_col1.append(float(list_data[0]))
                    data_col2.append(float(list_data[1]))
                    col1_values.append(float(list_data[0]))
                    col2_values.append(float(list_data[1]))
                except ValueError:
                    data_col1.append((list_data[0]))
                    data_col2.append((list_data[1]))
        my_file.close()
        rz = _np.array(col1_values)
        bf = _np.array(col2_values)
        return rz, bf

    def _read_field(self, config):
        fnames = EPUOnAxisFieldMap._CONFIG_FNAMES[config]
        fname = self.folder + \
            EPUOnAxisFieldMap._DATA_PATH + fnames[0]
        rz1, bx = EPUOnAxisFieldMap._read_field_component(fname)
        fname = self.folder + \
            EPUOnAxisFieldMap._DATA_PATH + fnames[1]
        rz2, by = EPUOnAxisFieldMap._read_field_component(fname)
        fname = self.folder + \
            EPUOnAxisFieldMap._DATA_PATH + fnames[2]
        rz3, bz = EPUOnAxisFieldMap._read_field_component(fname)

        # convert from UVX to SIRIUS coordinate systems and units.
        rz = (rz1 + rz2 + rz3)/3
        bx = -1*bx/10e3
        by = +1*by/10e3
        bz = +1*bz/10e3

        return rz, bx, by, bz

    def create_fieldmap_content(self, rz, bx, by, bz):
        """."""
        if self._centralize:
            self._rz_avg = _utils.calc_rz_of_field_center(rz, bx, by, bz)
            rz -= self._rz_avg

        # create date vectors
        limx = self._stepx * (self._nrptsx - 1)/2
        limy = self._stepy * (self._nrptsy - 1)/2
        rx = _np.linspace(-limx, +limx, self._nrptsx)
        ry = _np.linspace(-limy, +limy, self._nrptsy)
        x_col1 = _np.ones(len(rx)*len(ry)*len(rz))
        y_col2 = _np.ones(len(rx)*len(ry)*len(rz))
        z_col3 = _np.ones(len(rx)*len(ry)*len(rz))
        bx_col4 = _np.ones(len(rx)*len(ry)*len(rz))
        by_col5 = _np.ones(len(rx)*len(ry)*len(rz))
        bz_col6 = _np.ones(len(rx)*len(ry)*len(rz))
        line = 0
        for i in _np.arange(len(rz)):
            for j in _np.arange(len(ry)):
                for k in _np.arange(len(rx)):
                    x_col1[line] = rx[k]
                    y_col2[line] = ry[j]
                    z_col3[line] = rz[i]
                    bx_col4[line] = bx[i]
                    by_col5[line] = by[i]
                    bz_col6[line] = bz[i]
                    line+=1

        # create fieldmap file content
        content = ''
        content += 'X[mm]\tY[mm]\tZ[mm]\tBx[T]\tBy[T]\tBz[T]\n'
        content += '----------------------------------------------------------------------------------------------------------------------------------------------------------------\n'
        lfmt = '{:.1f}\t{:.1f}\t{:.1f}\t{:.5e}\t{:.5e}\t{:.5e}\n'
        for i in range(x_col1.size):
            content += lfmt.format(
                x_col1[i],y_col2[i],z_col3[i],bx_col4[i],by_col5[i],bz_col6[i])

        return content

    def _get_fmap_content(self):
        rz, bx, by, bz = self._read_field(self._config)
        content = self.create_fieldmap_content(rz, bx, by, bz)
        return content


class WigglerFieldmap(_FieldMap):
    """."""

    _DATA_PATH = (
        'wiggler-2T-STI/original-measurements/')

    def __init__(self, folder):
        self._folder = folder
        self.filename = None
        self.zmin = -419
        self.zmax = 3678
        self.zvalues = _np.arange(self.zmin,self.zmax+1,1)
        self.ymax = 12
        self.ystep = 1
        self.yvalues = None
        self.xvalues = [-45,-35,-25,-20,-16,-12,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,12,16,20,25,35]
        self.generate_fieldmap()

    def readfield(self):
        end_flag = 0
        with open(self.filename, encoding='utf-8', errors='ignore') as my_file:
            data_col1 = []
            data_col2 = []
            col1_values = []
            col2_values = []
            for i,line in enumerate(my_file):
                if i >= 24:
                    list_data = line.split('\t') #returns a list
                    if 'M' not in list_data[0]:
                        if end_flag == 0:
                            try:
                                data_col1.append(float(list_data[3]))
                                data_col2.append(float(list_data[5]))
                            except ValueError:
                                data_col1.append((list_data[3]))
                                data_col2.append((list_data[5]))
                    else:
                        end_flag = 1
        my_file.close()
        z = _np.array(data_col1)
        B = _np.array(data_col2)
        return z,B

    def readfile_axis(self, i):
        x = self.xvalues[i]
        folder = self._folder + WigglerFieldmap._DATA_PATH + \
            'gap 059.60 mm/ponta hall/mapeamento/'
        fieldname = "Map2701_X=" + str(x) + ".dat"
        self.filename = folder + fieldname
        rz_file,By_file = self.readfield()
        By_file = By_file/10e3

        By = []
        for z in self.zvalues:
            difference_array = _np.absolute(rz_file - z)
            index = difference_array.argmin()
            By.append(By_file[index])

        return By,fieldname

    def generate_fieldmap(self):
        By_dict = dict()
        for i,x in enumerate(self.xvalues):
            By,_ = self.readfile_axis(i)
            By_dict[x] = By

        self.yvalues = _np.arange(-self.ymax,self.ymax+self.ystep,self.ystep)

        x_col1 = _np.ones(len(self.xvalues)*len(self.yvalues)*len(self.zvalues))
        y_col2 = _np.ones(len(self.xvalues)*len(self.yvalues)*len(self.zvalues))
        z_col3 = _np.ones(len(self.xvalues)*len(self.yvalues)*len(self.zvalues))
        b_col4 = _np.ones(len(self.xvalues)*len(self.yvalues)*len(self.zvalues))
        b_col5 = _np.ones(len(self.xvalues)*len(self.yvalues)*len(self.zvalues))
        line = 0
        for i,z in enumerate(self.zvalues):
            for y in self.yvalues:
                for x in self.xvalues:
                    x_col1[line] = x
                    y_col2[line] = y
                    z_col3[line] = z
                    b_col4[line] = By_dict[x][i]
                    b_col5[line] = 0
                    line+=1

        my_file = open("Fieldmap.fld","w") #w=writing
        my_file.write('X[mm]\tY[mm]\tZ[mm]\tBx[T]\tBy[T]\tBz[T]\n')
        my_file.write('----------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        for i in range(x_col1.size):
            my_file.write("{:.1f}\t{:.1f}\t{:.1f}\t{:.5e}\t{:.5e}\t{:.5e}\n".format(x_col1[i],y_col2[i],z_col3[i],b_col5[i],b_col4[i],b_col5[i]))
        my_file.close()


class FieldmapOnAxisAnalysis:

    def __init__(self,
                 fieldmap=None,
                 filename=None,
                 beam_energy=3.0
                 ):
        """."""
        self.filename = filename
        self.fieldmap = fieldmap
        self.rz = None
        self.bx = None
        self.by = None
        self.bz = None
        self.field = None
        self.beam = _Beam(beam_energy)
        self.traj = None
        self.s_step = 0.5

    def calc_K(self, period, B, zmin, zmax, rz=None):
        begin_idx = _np.where(rz==zmin)
        end_idx = _np.where(rz==zmax)
        # length = end_idx[0][0] - begin_idx[0][0]
        k = 2*_np.pi/(period)
        a11, a12, a21, a22 = [0, ]*4
        b1, b2 = 0, 0
        for i in _np.arange(begin_idx[0][0], end_idx[0][0]+1, 1):
            b1 = B[i]*_np.cos(k*rz[i]) + b1
            b2 = B[i]*_np.sin(k*rz[i]) + b2
            a11 = (_np.cos(k*rz[i]))**2 + a11
            a12 = _np.cos(k*rz[i])*_np.sin(k*rz[i]) + a12
            a22 = (_np.sin(k*rz[i]))**2 + a22
        a21 = a12
        M = _np.array([[a11,a12],[a21,a22]])
        det = _np.linalg.det(M)
        M_inv = _np.array([[a22,-a12],[-a21,a11]])/det
        B_matriz = _np.array([[b1],[b2]])
        Res = _np.dot(M_inv,B_matriz)
        a = Res[0][0]
        b = Res[1][0]
        B0 = _np.sqrt(a**2+b**2)
        K = 93.36*B0*(period*1e-3)
        return K, B0

    def _calc_fxy(self,
                 s_step=None, max_rz=4000, plot_flag=True,
                 ipos=[0,0,0], iang=[0,0,1], end=None):
        init_rx, init_ry, init_rz = ipos
        init_px, init_py, init_pz = iang
        if self.s_step == None:
            self.s_step = _np.average(_np.diff(self.rz))/2
        self.traj.calc_trajectory(
            s_step=self.s_step, min_rz=max_rz,
            init_rx=init_rx, init_ry=init_ry, init_rz=init_rz,
            init_px=init_px, init_py=init_py, init_pz=init_pz)
        z = self.traj.rz.copy()
        x = self.traj.rx.copy()
        px= self.traj.px.copy()
        y = self.traj.ry.copy()
        py= self.traj.py.copy()
        dpx = _np.diff(px)
        if end == None:
            idx_end = _np.where(_np.abs(dpx) < 1e-7)
            final_idx = []
            cond = int(max_rz/(2*self.s_step))
            for i in _np.arange(_np.shape(idx_end)[1]):
                if idx_end[0][i] > cond:
                    final_idx.append(idx_end[0][i])
            final = final_idx[0]
        else:
            final = end
        if plot_flag == True:
            _plt.plot(z,x, color='g', label='step = {:.1f}mm'.format(self.s_step))
            _plt.xlabel('Z coordinate (mm)')
            _plt.ylabel('X coordinate (mm)')
            _plt.legend()
            _plt.show()

            _plt.plot(z,y, color='g', label='step = {:.1f}mm'.format(self.s_step))
            _plt.xlabel('Z coordinate (mm)')
            _plt.ylabel('Y coordinate (mm)')
            _plt.legend()
            _plt.show()
        return x[final], y[final], px[-1], py[-1], z[final], final

    @staticmethod
    def calc_first_integral(B, rz=None):
        dz = 1e-3*(rz[-1] - rz[0]) / (len(rz) -1)
        integral = _integrate.cumtrapz(y=B, dx=dz)
        return integral

    @staticmethod
    def calc_second_integral(I1, rz=None):
        dz = 1e-3*(rz[-1] - rz[0]) / (len(rz) -1)
        integral2 = _integrate.cumtrapz(y=I1, dx=dz)
        return integral2

    def load_fieldmap(self):
        field = _FieldMap(self.fieldmap)
        beam = self.beam
        self.traj = _Trajectory(beam=beam, fieldmap=field)
        # Get field components on axis
        idx_x0 = int((len(field.rx)-1)/2)
        idx_y0 = int((len(field.ry)-1)/2)
        self.rz = field.rz
        self.Bx = field.bx[idx_y0][idx_x0][:]
        self.By = field.by[idx_y0][idx_x0][:]
        self.Bz = field.bz[idx_y0][idx_x0][:]
        self.field = field


    def run(self):
        """."""
        x, y, px, py, z_end, idx_end = \
            self._calc_fxy(s_step=self.s_step, max_rz=4000, plot_flag=False,
                ipos=[0,0,0], iang=[0,0,1])

        print("Final y position: {:.2f} um".format(1e3*y))
        print("Final y angle   : {:.2f} urad".format(1e6*py))
        print("Final x position: {:.2f} um".format(1e3*x))
        print("Final x angle   : {:.2f} urad".format(1e6*px))

        by_integral1 = self.calc_first_integral(B=self.By,rz=self.rz)
        print("By first integral (Tm)  : {:.3e}".format(by_integral1[-1]))

        by_integral2 = self.calc_second_integral(I1=by_integral1,rz=self.rz)
        print("By second integral (Tm2): {:.3e}".format(by_integral2[-1]))

        bx_integral1 = self.calc_first_integral(B=self.Bx,rz=self.rz)
        print("Bx first integral (Tm)  : {:.3e}".format(bx_integral1[-1]))

        bx_integral2 = self.calc_second_integral(I1=bx_integral1,rz=self.rz)
        print("Bx second integral (Tm2): {:.3e}".format(bx_integral2[-1]))

        Ky, Bymax = self.calc_K(period=50,B=self.By,zmin=500,zmax=2500,rz=self.rz)
        print("K value for By          : ", Ky)
        print("Field amplitude for By  : ", Bymax)

        Kx, Bxmax = self.calc_K(period=50,B=self.Bx,zmin=500,zmax=2500,rz=self.rz)
        print("K value for Bx          : ", Kx)
        print("Field amplitude for Bx  : ", Bxmax)

        my_file = open(self.filename,"w") #w=writing
        my_file.write('x[um]\tpx[rad]\ty[mm]\tpy[urad]\tIx[Tm]\t        Iy[Tm]\t        I2x[Tm2]\tI2y[Tm2]\tBx[T]\tBy[T]\tKx\tKy\n')
        my_file.write("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t       {:.3e}\t{:.3e}\t{:.3e}\t{:.3e}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(1e3*x,1e6*px,1e3*y,1e6*py,bx_integral1[-1],by_integral1[-1],bx_integral2[-1],by_integral2[-1],Kx,Bxmax,Ky,Bymax))
        my_file.close()
