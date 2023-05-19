#!/usr/bin/env python-sirius

from idanalysis import IDTrajectory

from idanalysis.utils import create_deltadata
from idanalysis.utils import get_config_names


def plot_corrected_trajectory(deltadata, index):    
    """Correct RK traj with FF corrs and plot results."""

    names = get_config_names(deltadata)
    config_name = names[index]
    fmap, label = deltadata.get_fieldmap(config_name)
    label = label.replace('_dGH=+0.0000', '')
    label = label.replace('_dCP=+0.0000', '')
    traj = IDTrajectory(label=label, fieldmap=fmap)
    title = 'RK Traj in ID\n' + label + '_Rand' + str(index+1)
    print(title.replace('\n', ' - '))
    traj.correct_posang_init(5, plot=True, title=title)


if __name__ == "__main__":
    deltadata = create_deltadata()
    plot_corrected_trajectory(deltadata, index=0)




