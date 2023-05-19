from .deltadata import DeltaData
from .epudata import EPUData
from .trajectory import IDTrajectory
from .kickmaps import IDKickMap
import os as _os

# This has to be defined before using the library.
FOLDER_BASE = None

with open(_os.path.join(__path__[0], 'VERSION'), 'r') as _f:
    __version__ = _f.read().strip()
