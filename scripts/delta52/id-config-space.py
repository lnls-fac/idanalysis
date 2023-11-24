#!/usr/bin/env python-sirius

import matplotlib.pyplot as plt


def id_delta():
    """."""

    title = 'DELTA dGV Configuration Space'
    suptitle = '[CSD = dGV; CSE = dGV + dP; CID = dP; CIE = 0]'

    # available config space
    color = 'gray'
    plt.plot([+0, +1], [-1, -1], '--', color=color)
    plt.plot([+1, +1], [-1, +0], '--', color=color)
    plt.plot([+1, +0], [+0, +1], '--', color=color)
    plt.plot([+0, -1], [+1, +1], '--', color=color)
    plt.plot([-1, -1], [+1, +0], '--', color=color)
    plt.plot([-1, +0], [+0, -1], '--', color=color)

    # no light scan
    color = 'black'
    pts = [+0, +0], [+1, -1]
    plt.plot(*pts, color=color, alpha=0.5, linewidth=3, label='no light')

    # linear vertical polarization energy scan
    color = 'red'
    pts = [+0, +1], [-1, -1]
    plt.plot(*pts, color=color, alpha=0.5, linewidth=3, label='linear vertical')

    # linear horizontal polarization energy scan
    color = 'blue'
    pts = [+0, +1], [+0, +0]
    plt.plot(*pts, color=color, alpha=0.5, linewidth=3, label='linear horizontal')

    # circular positive polarization energy scan
    color = 'green'
    pts = [+0, -1], [+1/2, +1/2]
    plt.plot(*pts, color=color, alpha=0.5, linewidth=3, label='circular positive')

    # circular negative polarization energy scan
    color = 'magenta'
    pts = [+0, 1], [-1/2, -1/2]
    plt.plot(*pts, color=color, alpha=0.5, linewidth=3, label='circular negative')

    # selected configurations
    pts =list()
    # pts += [('black', 'V_0', 0, -1), ('red', 'V_1', 1/2, -1), ('red', 'V_2', 1, -1)]
    # pts += [('red', 'V_1', 1/2, -1), ('red', 'V_2', 1, -1)]
    # pts += [('black', 'H_0', 0, 0), ('blue', 'H_1', 1/2, 0), ('blue', 'H_2', 1, 0)]
    # pts += [('black', '\\bar{C}_0', 0, -1/2), ('magenta', '\\bar{C}_1', 1/2, -1/2), ('magenta', '\\bar{C}_2', 1, -1/2)]
    # pts += [('magenta', '\\bar{C}_1', 1/2, -1/2), ('magenta', '\\bar{C}_2', 1, -1/2)]
    for point in pts:
        color, letter, *point = point
        plt.plot(*point, 'o', color=color)
        text = r'$' + letter + r'$'
        plt.annotate(
            text, point, xytext=(5, 5), textcoords='offset pixels',
            color=color)

    ax = plt.axes()
    ticklabels = ['-1', '-3/4', '-1/2', '-1/4', '0', '+1/4', '+1/2', '+3/4', '+1']
    ax.set_xticks([eval(tick) for tick in ticklabels])
    ax.set_xticklabels(ticklabels)
    ax.set_yticks([eval(tick) for tick in ticklabels])
    ax.set_yticklabels(ticklabels)
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.xlabel(r'dGV / ($\lambda/2$)')
    plt.ylabel(r'dP / ($\lambda/2$)')

    plt.suptitle(title)
    plt.title(suptitle)
    plt.grid()
    plt.legend()
    # plt.tight_layout()
    plt.show()



id_delta()

