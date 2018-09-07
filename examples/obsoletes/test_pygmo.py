#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 23:38:52 2018

@author: niko
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pygmo import *
from test_pyswmm.swmm_problem import CircularPipeProblem


BOUNDS = [[1., 4.]] * 11
PROBLEM = CircularPipeProblem(inpfile='test_pyswmm/network11pipes.inp', dim=11, n_objs=2,
                              bounds=BOUNDS, swmm_lib_path='test_pyswmm/swmm_api')

udp = PROBLEM
pop = population(prob=udp, size=100)
algo = algorithm(moead(gen=500))

try:
    for i in range(1):
        pop = algo.evolve(pop)
except ValueError as e:
    err = e

#hv = hypervolume(pop)
#hv.compute(ref_point = [1., 1., 1.])
#from matplotlib import pyplot as plt
#udp.plot(pop)


theo = udp.solutions()


# ================================Visualization================================
import matplotlib.pyplot as plt
from matplotlib import rc

# Matplotlib
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)



fig, (ax, ax_metric) = plt.subplots(1, 2, figsize=(10,4), dpi=100)

# Test surrogate results
final_arc = pop.get_f()
x = final_arc[:, 0]
y = final_arc[:, 1]


#eq1  = r'$\displaystyle f_1 = x_1$'
#eq2  = r'$\displaystyle f_2 = g(x)h(f_1(x),g(x))$'
#eq3 = r'$\displaystyle g(x)=1+\sum_{i=2}^{30} x_i$'
#eq4 = r'$\displaystyle h(f_1(x),g(x))=1-\sqrt{\frac{f_1(x)}{g(x)}}$'
#eq5 = r'$x_i\in\left[0,1\right]$'
#ax.text(0.68, 1.03, eq1+'\n'+eq2+'\n'+eq3+'\n'+eq4+'\n'+eq5,
#        fontsize=12, size=12,
#        ha="left", va="top",
#        bbox=dict(boxstyle="square", fc=(1., 1., 1., 0.8), ec='#a5a5a5'))

ax.set_xlabel(r'$F_1$')
ax.set_ylabel(r'$F_2$', labelpad=0)
ax.set(title="Non-Dominated Front")

#ax.set_ylim((0., 1.))
#ax.set_xlim((0., 1.))

ax.scatter(theo[:,0], theo[:,1], c='orangered', s=1.5,
           label="Analytical (Li et Zhang 2009)")
ax.scatter(x, y, c='royalblue', s=2.5, label="RBFN Assisted NSGA-II")

# Plot legend.
lgnd = ax.legend(loc='lower left', numpoints=1, fontsize=9)

# change the marker size manually for both lines
lgnd.legendHandles[0]._sizes = [10]
lgnd.legendHandles[1]._sizes = [10]
ax.grid(True, ls=':')

ax.text(ax.get_xlim()[1]-0.1, 0.9, 'LZ-5',
        ha='right', va='top', size=14,
        bbox={'boxstyle': 'round',
              'ec': (0, 0, 0),
              'fc': (1, 1, 1),
              }
        )


# ================================ Metrics ====================================
ax_hypervol = ax_metric.twinx()

# Calculate hypervolume coverage
hypervol_cov = [hv/pop.hypervol_ana for hv in pop.hypervol]

# Title
ax_metric.set_title('Hypervolume Metrics')

# Ranges
ax_metric.set_xlim((0, pop.episode))
ax_metric.set_ylim((0., 1.))
ax_hypervol.set_ylim((0., 1.))

# Lables
ax_metric.set_ylabel('Uncovered \ Hypervolume', color='royalblue', labelpad=-1)
ax_hypervol.set_ylabel('Hypervolume \ Coverage (\%)', color='orangered')
ax_metric.set_xlabel('Episode (N)')

# Ticks
ax_metric.set_yticklabels(ax_metric.get_yticks().round(1),
                          color='royalblue')
ax_hypervol.set_yticklabels(ax_hypervol.get_yticks().round(1),
                            color='orangered')

# Grid
ax_metric.grid(True, ls=':')

# Plot
metric = ax_metric.plot(pop.hypervol_diff, color='royalblue',
                        label='Uncovered Hypervolume', linewidth=2.0)
hypervol = ax_hypervol.plot(hypervol_cov,
                            label='Hypervolume Coverage (\%)',
                            color='orangered',
                            linewidth=2.0)

legends = metric + hypervol
labs = [l.get_label() for l in legends]

ax_metric.legend(legends, labs, loc='center right', fontsize=9)

plt.show()


#=========================== Benchmark surrogate =============================
n_sampled_arch = pop.sampled_archive.__len__()
print('Total evals: %i, sampled archive size: %i, front size: %i' % (
        pop.problem.n_evals, n_sampled_arch, pop.true_front.__len__()))