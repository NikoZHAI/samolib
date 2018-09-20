import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import sys, os
sys.path.append('/home/niko/Documents/SPr-GAN/00_Projects/MOPRISM/77_Playground/GA/samolib/')

# Matplotlib
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def compare_zdt(i):

    ref = np.genfromtxt('pareto_Kursawe.txt', dtype='f8')

    path_mop = str(i) + '/'
    path_gom = str(i+1) + '/'

    # Test surrogate results
    f_mop = np.fromfile(path_mop + 'fs.dat', dtype='f8')
    f_mop = f_mop.reshape(int(f_mop.shape[0]/2), 2)
    f_gom = np.fromfile(path_gom + 'fs.dat', dtype='f8')
    f_gom = f_gom.reshape(int(f_gom.shape[0]/2), 2)

    hv_mop = np.fromfile(path_mop + 'hv_cov.dat', dtype='f8')
    hv_gom = np.fromfile(path_gom + 'hv_cov.dat', dtype='f8')
    hv_ind_mop = np.fromfile(path_mop + 'hv_ind.dat', dtype=int)
    hv_ind_gom = np.fromfile(path_gom + 'hv_ind.dat', dtype=int)


    fig, (ax, ax_metric) = plt.subplots(1, 2, figsize=(10,4.5), dpi=100)

    # Ranges
    ax.set_xlim((-20, -14))
    ax.set_ylim((-12, 0.5))

    if ax.get_yscale() == 'symlog':
        ax.set_yticks([0, 1, 5, 10, 50, 100], minor=False)
        ax.set_yticklabels([0, 1, 5, 10, 50, 100])

    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$', labelpad=0)
    ax.set(title="Non-dominated Solutions - KURS")

    ax.scatter(ref[:, 0], ref[:, 1], s=2.5, c='black', label="PS", zorder=99)
    ax.scatter(f_mop[:, 0], f_mop[:, 1], c='orangered', s=12., label="MOPRISM")
    ax.scatter(f_gom[:, 0], f_gom[:, 1], c='royalblue', s=20., label="GOMORS", marker='x')

    # Plot legend
    lgnd = ax.legend(loc='upper right', numpoints=1, fontsize=10)

    # change the marker size manually for both lines
    lgnd.legendHandles[0]._sizes = [12]
    lgnd.legendHandles[1]._sizes = [12]
    lgnd.legendHandles[2]._sizes = [20]

    ax.grid(True, ls=':', which='both')

    # Add problem label
    #ax.text(0.2, 0.2, 'ZDT-4',
    #        ha='left', va='bottom', size=14, bbox={'boxstyle': 'round',
    #                                               'ec': (0, 0, 0),
    #                                               'fc': (1, 1, 1),},
    #        transform=ax.transAxes
    #)


    # ================================ Metrics ====================================
    # Title
    ax_metric.set_title('Hypervolume Convergence')

    # Ranges
    ax_metric.set_xlim((100, 1000))
    ax_metric.set_ylim((0., 1.))

    # Lables
    ax_metric.set_ylabel(r'Hypervolume \ Coverage', labelpad=None)
    ax_metric.set_xlabel('Number of Evaluations (N)')

    # Grid
    ax_metric.grid(True, ls=':')

    # Plot
    hvplt_mop = ax_metric.plot(hv_ind_mop, hv_mop, label='MOPRISM', color='orangered',
                               linewidth=2.0)
    hvplt_gom = ax_metric.plot(hv_ind_gom, hv_gom, label='GOMORS', color='royalblue',
                               linewidth=2.0)

    # Ticks
    ax_metric.set_yticklabels(ax_metric.get_yticks().round(2))

    ax_metric.legend(loc='best', fontsize=10)

    fig.tight_layout()
    plt.show()
    fig.savefig('compare_' + 'kursawe' + '.png', dpi=300, format='png')

    return None

if __name__ == '__main__':

    for i in range(1):
        compare_zdt(i)
