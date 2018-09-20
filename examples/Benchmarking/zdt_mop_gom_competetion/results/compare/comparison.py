import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.ticker as ticker

import sys, os
sys.path.append('/home/niko/Documents/SPr-GAN/00_Projects/MOPRISM'
                '/77_Playground/GA/samolib/')

from benchmarks import (ZDT1, ZDT2, ZDT3, ZDT4, ZDT6)


# Matplotlib
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def compare(i):

    problems = (ZDT1, ZDT2, ZDT3, ZDT4, ZDT6)
    problem = problems[i]()
    ref = problem.solutions()

    path_raw = 'zdt1-6_moead_mop_raw/' # Path to the new tests with MOPRISM-raw
    path_old = 'zdt1-6_moead/'         # Path to the previous tests with MOPRISM-G and GOMORS
    path_mop = path_old + str(i) + '/'
    path_gom = path_old + str(i+5) + '/'
    path_mop_raw = path_raw + str(i) + '/'
    path_mop_20  = path_raw + str(i+5) + '/'
    path_mop_10  = path_raw + str(i+10) + '/'

    # Test surrogate results
    f_mop = np.fromfile(path_mop + 'fs.dat', dtype='f8')
    f_mop = f_mop.reshape(int(f_mop.shape[0]/2), 2)
    f_gom = np.fromfile(path_gom + 'fs.dat', dtype='f8')
    f_gom = f_gom.reshape(int(f_gom.shape[0]/2), 2)
    f_mop_raw = np.fromfile(path_mop_raw + 'fs.dat', dtype='f8')
    f_mop_raw = f_mop_raw.reshape(int(f_mop_raw.shape[0]/2), 2)
    f_mop_20 = np.fromfile(path_mop_20 + 'fs.dat', dtype='f8')
    f_mop_20 = f_mop_20.reshape(int(f_mop_20.shape[0]/2), 2)
    f_mop_10 = np.fromfile(path_mop_10 + 'fs.dat', dtype='f8')
    f_mop_10 = f_mop_10.reshape(int(f_mop_10.shape[0]/2), 2)

    hv_mop = np.fromfile(path_mop + 'hv_cov.dat', dtype='f8')
    hv_gom = np.fromfile(path_gom + 'hv_cov.dat', dtype='f8')
    hv_mop_raw = np.fromfile(path_mop_raw + 'hv_cov.dat', dtype='f8')
    hv_mop_20 = np.fromfile(path_mop_20 + 'hv_cov.dat', dtype='f8')
    hv_mop_10 = np.fromfile(path_mop_10 + 'hv_cov.dat', dtype='f8')
    hv_ind_mop = np.fromfile(path_mop + 'hv_ind.dat', dtype=int)
    hv_ind_gom = np.fromfile(path_gom + 'hv_ind.dat', dtype=int)
    hv_ind_mop_raw = np.fromfile(path_mop_raw + 'hv_ind.dat', dtype=int)
    hv_ind_mop_20 = np.fromfile(path_mop_20 + 'hv_ind.dat', dtype=int)
    hv_ind_mop_10 = np.fromfile(path_mop_10 + 'hv_ind.dat', dtype=int)

    # plot
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
    ax.scatter(f_mop[:, 0], f_mop[:, 1], c='orangered', s=12., label="MOPRISM-g")
    ax.scatter(f_gom[:, 0], f_gom[:, 1], c='royalblue', s=20., label="GOMORS", marker='x')
    ax.scatter(f_mop_raw[:, 0], f_mop_raw[:, 1], c='forestgreen', s=12., label="MOPRISM-raw", marker='^')
    ax.scatter(f_mop_20[:, 0], f_mop_20[:, 1], c='crimson', s=12., label="MOPRISM-20", marker='s')
    ax.scatter(f_mop_10[:, 0], f_mop_10[:, 1], c='brown', s=12., label="MOPRISM-10", marker='*')

    # Plot legend
    lgnd = ax.legend(loc='upper right', numpoints=1, fontsize=10)

    # change the marker size manually for both lines
    lgnd.legendHandles[0]._sizes = [12]
    lgnd.legendHandles[1]._sizes = [12]
    lgnd.legendHandles[2]._sizes = [20]
    lgnd.legendHandles[3]._sizes = [12]
    lgnd.legendHandles[4]._sizes = [12]
    lgnd.legendHandles[5]._sizes = [12]

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
    hvplt_mop = ax_metric.plot(hv_ind_mop, hv_mop, label='MOPRISM-g', color='orangered',
                               linewidth=2.0)
    hvplt_gom = ax_metric.plot(hv_ind_gom, hv_gom, label='GOMORS', color='royalblue',
                               linewidth=2.0)
    hvplt_mop_raw = ax_metric.plot(hv_ind_mop_raw, hv_mop_raw, label='MOPRISM-raw', color='forestgreen',
                               linewidth=2.0)
    hvplt_mop_20 = ax_metric.plot(hv_ind_mop_20, hv_mop_20, label='MOPRISM-20', color='crimson',
                               linewidth=2.0)
    hvplt_mop_10 = ax_metric.plot(hv_ind_mop_10, hv_mop_10, label='MOPRISM-10', color='brown',
                               linewidth=2.0)
    # Ticks
    ax_metric.set_yticklabels(ax_metric.get_yticks().round(2))

    ax_metric.legend(loc='best', fontsize=10)

    fig.tight_layout()
    plt.show()
    fig.savefig('compare_' + 'kursawe' + '.png', dpi=300, format='png')

    return None


def compare_gridplot(i):

    problems = (ZDT1, ZDT2, ZDT3, ZDT4, ZDT6)
    problem = problems[i]()
    ref = problem.solutions()

    path_raw = 'zdt1-6_moead_mop_raw/' # Path to the new tests with MOPRISM-raw
    path_old = 'zdt1-6_moead/'         # Path to the previous tests with MOPRISM-G and GOMORS
    path_mop = path_old + str(i) + '/'
    path_gom = path_old + str(i+5) + '/'
    path_mop_raw = path_raw + str(i) + '/'
    path_mop_20  = path_raw + str(i+5) + '/'
    path_mop_10  = path_raw + str(i+10) + '/'

    # Test surrogate results
    f_mop = np.fromfile(path_mop + 'fs.dat', dtype='f8')
    f_mop = f_mop.reshape(int(f_mop.shape[0]/2), 2)
    f_gom = np.fromfile(path_gom + 'fs.dat', dtype='f8')
    f_gom = f_gom.reshape(int(f_gom.shape[0]/2), 2)
    f_mop_raw = np.fromfile(path_mop_raw + 'fs.dat', dtype='f8')
    f_mop_raw = f_mop_raw.reshape(int(f_mop_raw.shape[0]/2), 2)
    f_mop_20 = np.fromfile(path_mop_20 + 'fs.dat', dtype='f8')
    f_mop_20 = f_mop_20.reshape(int(f_mop_20.shape[0]/2), 2)
    f_mop_10 = np.fromfile(path_mop_10 + 'fs.dat', dtype='f8')
    f_mop_10 = f_mop_10.reshape(int(f_mop_10.shape[0]/2), 2)

    hv_mop = np.fromfile(path_mop + 'hv_cov.dat', dtype='f8')
    hv_gom = np.fromfile(path_gom + 'hv_cov.dat', dtype='f8')
    hv_mop_raw = np.fromfile(path_mop_raw + 'hv_cov.dat', dtype='f8')
    hv_mop_20 = np.fromfile(path_mop_20 + 'hv_cov.dat', dtype='f8')
    hv_mop_10 = np.fromfile(path_mop_10 + 'hv_cov.dat', dtype='f8')
    hv_ind_mop = np.fromfile(path_mop + 'hv_ind.dat', dtype=int)
    hv_ind_gom = np.fromfile(path_gom + 'hv_ind.dat', dtype=int)
    hv_ind_mop_raw = np.fromfile(path_mop_raw + 'hv_ind.dat', dtype=int)
    hv_ind_mop_20 = np.fromfile(path_mop_20 + 'hv_ind.dat', dtype=int)
    hv_ind_mop_10 = np.fromfile(path_mop_10 + 'hv_ind.dat', dtype=int)

    # plot
    fig = plt.figure(figsize=(10,4.5), dpi=100)
    gs1 = GridSpec(1, 2)
    gs0 = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs1[0], wspace=0.025, hspace=0.025)
    ax_metric = plt.subplot(gs1[0, 1])
    ax_metric.set_facecolor((0.97, 0.97, 0.97)) # Background color
    axs = [plt.subplot(gs0[i_, j]) for i_ in range(2) for j in range(2)]
    fs = [f_gom, f_mop_raw, f_mop_20, f_mop_10]
    labels = ['GOMORS', 'MOPRISM-raw', 'MOPRISM-20', 'MOPRISM-10']
    colors = ['#ca0020', '#f4a582', 'forestgreen', '#92c5de', '#0571b0']

    for ax, f, l, c, i_ in zip(axs, fs, labels, colors[1:], range(2*2)):

        # Background color
        ax.set_facecolor((0.97, 0.97, 0.97))

        # Ranges
        if problem.__class__ is ZDT6:
            ax.set_xlim((0.2, 1.))
            if i_ < 2:
                ax.set_yscale('symlog', basey=10, subsy=np.arange(2, 10))
        elif problem.__class__ is ZDT4:
            ax.set_xlim((0., 1.))
            ax.set_yscale('symlog', basey=10, subsy=np.arange(2, 10))
        elif problem.__class__ is ZDT3:
            ax.set_xlim((0., 1.))
        else:
            ax.set_xlim((0., 1.))
            ax.set_ylim((0., 1.))

        if ax.get_yscale() == 'symlog':
            ax.set_yticks([0, 1, 5, 10, 50, 100], minor=False)
            ax.set_yticklabels([0, 1, 5, 10, 50, 100])

        if i_ < 2:
            ax.get_shared_x_axes().join(ax, axs[i_+2])
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r'$f_1$')

            if i_==2:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4,
                                                              integer=False,
                                                              prune='upper'))
                if not(ax.get_yscale() == 'symlog'):
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5,
                                                                  integer=False,
                                                                  prune='upper'))

        ax.grid(True, ls=':', which='both')

        if i_ == 1 or i_ == 3:
            ax.get_shared_y_axes().join(ax, axs[i_-1])
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'$f_2$', labelpad=0)

        #ax.set(title="Non-dominated Solutions - KURS")
        if problem.__class__ is ZDT3:
            ax.scatter(ref[:, 0], ref[:, 1], s=0.1, c='black', label="PS",
                       zorder=100)
        else:
            ax.plot(ref[:, 0], ref[:, 1], c='black', label="PS", zorder=100,
                    linewidth=0.8)
        ax.scatter(f_mop[:, 0], f_mop[:, 1], c=colors[0], s=10,
                   label='MOPRISM-g')
        ax.scatter(f[:, 0], f[:, 1], c=c, s=10., label=l, marker='x')

        # Plot legend
        lgnd = ax.legend(loc='best', numpoints=1, fontsize=7, facecolor='white')

        # change the marker size manually for both lines
        lgnd.legendHandles[0]._sizes = [12]
        lgnd.legendHandles[1]._sizes = [12]
        lgnd.legendHandles[2]._sizes = [20]


    # ================================ Metrics ====================================
    # Title
    ax_metric.set_title('Hypervolume Convergence')

    # Ranges
    ax_metric.set_xlim((100, 2000))
    ax_metric.set_xticks([100] + list(range(250, 2001, 250)), minor=False)
    ax_metric.set_xticklabels([100] + list(range(250, 2001, 250)))
    if problem.__class__ is ZDT4 or problem.__class__ is ZDT6:
        pass
    else:
        ax_metric.set_ylim((0., 1.))

    # Lables
    ax_metric.set_ylabel(r'Hypervolume \ Coverage', labelpad=None)
    ax_metric.set_xlabel('Number of Evaluations (N)')

    # Grid
    ax_metric.grid(True, ls=':')

    # Plot
    hvplt_mop = ax_metric.plot(hv_ind_mop, hv_mop, label='MOPRISM-g', color=colors[0],
                               linewidth=1.5)
    hvplt_gom = ax_metric.plot(hv_ind_gom, hv_gom, label='GOMORS', color=colors[1],
                               linewidth=1.5)
    hvplt_mop_raw = ax_metric.plot(hv_ind_mop_raw, hv_mop_raw, label='MOPRISM-raw', color=colors[2],
                                   linewidth=1.5)
    hvplt_mop_20 = ax_metric.plot(hv_ind_mop_20, hv_mop_20, label='MOPRISM-20', color=colors[3],
                                  linewidth=1.5)
    hvplt_mop_10 = ax_metric.plot(hv_ind_mop_10, hv_mop_10, label='MOPRISM-10', color=colors[4],
                                  linewidth=1.5)
    # Ticks
    ax_metric.set_yticklabels(ax_metric.get_yticks().round(2))

    ax_metric.legend(loc='best', fontsize=10, facecolor='white')

    fig.tight_layout()
    plt.show()
    if i == 4: i += 1
    fig.savefig('zdt' + str(i+1) + '.png', dpi=300, format='png')
    fig.clf()

    return None


if __name__ == '__main__':

    for i in range(5):
        #compare(i)
        compare_gridplot(i)
