"""IDs configurations."""

import numpy as _np
import matplotlib.pyplot as _plt

import pyaccel
from fieldmaptrack.fieldmap import FieldMap as _FieldMap


class EPUData:
    """EPU fieldmap and kickmap data access class."""

    DEFAULT_GAP = 22  # [mm]
    DEFAULT_PERIOD = 50  # [mm]

    KMAP_FNAME_EXT = '.txt'  # // '.kck'

    FOLDER_EPU = (
        'MatlabMiddleLayer/Release/lnls/fac_scripts/'
        'sirius/insertion_devices/id_modelling/')

    FOLDER_EPU_MAPS = ( FOLDER_EPU + 'EPU50/')

    EPU_CONFIGS = (
        'EPU50_CP_kicktable',
        'EPU50_HP_kicktable',
        'EPU50_VP_kicktable',
        )

    def __init__(self, folder, configs=None):
        """."""
        self._folder = folder
        if configs is None:
            configs = EPUData.EPU_CONFIGS
        self._configs = configs
        self._header = dict()
        self._output_folder = ''

    @property
    def output_folder(self):
        """."""
        return self._output_folder

    @output_folder.setter
    def output_folder(self, value):
        """."""
        value = value if value[-1] == '/' else value + '/'
        self._output_folder = value

    def get_header(self, config):
        """."""
        return self._header[config]

    def get_fieldmap(self, config):
        """Return Trajectory object for a given config."""
        fname = self.get_fieldmap_filename(config)
        fmap = _FieldMap(fname=fname)
        label = self.get_config_label(config)
        return fmap, label

    def get_gap(self, config, norm=None):
        """Return EPU gap [mm]."""
        if config not in self._header:
            self._read_fieldmap_header(config)
        data = self._header[config]
        if norm is None:
            return data['gap']
        else:
            return data['gap']/norm

    def get_config_label(self, config):
        """."""
        config = config.replace('_kicktable','')
        gap = self.get_gap(config)
        fstr = config + '_gap={:.3f}'
        return fstr.format(gap)

    def get_kickmap_filename(self, config):
        """Return kickmap filename of a config."""
        return self._folder + config + EPUData.KMAP_FNAME_EXT

    def get_fieldmap_filename(self, config):
        """Return fieldmap filename of a config."""
        return self._folder + config + '.fld'

    def get_period_length(self, config=None):
        if config is None:
            config = self._configs[0]
        if config not in self._header:
            self._read_fieldmap_header(config)
        return self._header[config]['period_length']

    def get_dCSD(self, config, norm=None):
        """Return dCSD [mm] position of ID configuration."""
        if config not in self._header:
            self._read_fieldmap_header(config)
        data = self._header[config]
        if norm is None:
            return data['posCSD']/(self.get_period_length()/2)
        else:
            return data['posCSD']/norm

    def get_dCSE(self, config, norm=None):
        """Return dCSE [mm] position of ID configuration."""
        if config not in self._header:
            self._read_fieldmap_header(config)
        data = self._header[config]
        if norm is None:
            return data['posCSE']/(self.get_period_length()/2)
        else:
            return data['posCSE']/norm

    def get_dCID(self, config, norm=None):
        """Return dCID [mm] position of ID configuration."""
        if config not in self._header:
            self._read_fieldmap_header(config)
        data = self._header[config]
        if norm is None:
            return data['posCID']/(self.get_period_length()/2)
        else:
            return data['posCID']/norm

    def get_dCIE(self, config, norm=None):
        """Return dCIE [mm] position of ID configuration."""
        if config not in self._header:
            self._read_fieldmap_header(config)
        data = self._header[config]
        if norm is None:
            return data['posCIE']/(self.get_period_length()/2)
        else:
            return data['posCIE']/norm

    def get_config_names(self):
        names = []
        for config in self:
            names.append(config)
        return names

    # --- analysis ---

    def __getitem__(self, idx):
        """."""
        return self._configs[idx]

    def __len__(self):
        """."""
        return len(self._configs)

    def __iter__(self):
        """."""
        self._idx = 0
        return self

    def __next__(self):
        """."""
        if self._idx < len(self._configs):
            config = self._configs[self._idx]
            self._idx += 1
            return config
        else:
            raise StopIteration

    def _read_fieldmap_header(self, config):
        """Read fieldmap file header data."""
        data = dict()
        fname = self.get_fieldmap_filename(config)
        try:
            with open(fname, 'r') as fp:
                while True:
                    line = fp.readline()
                    if 'X[mm]' in line:
                        break
                    words = line.split()
                    if len(words) == 2:
                        wd = words[0]
                        wd = wd.replace(':', '')
                        wd = wd.replace('[mm]', '')
                        try:
                            data[wd] = float(words[1])
                        except ValueError:
                            data[wd] = words[1]
        except FileNotFoundError:
            data = dict(
                gap=EPUData.DEFAULT_GAP,
                period_length=EPUData.DEFAULT_PERIOD,
                )
        self._header[config] = data

    def _read_kickmap(self, config):
        """."""
        fname = self.get_kickmap_filename(config)
        return pyaccel.elements.Kicktable(filename=fname)
