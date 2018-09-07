#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:52:22 2018

@author: niko
"""
import os, sys
#import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ea import Population
from benchmarks import LZ6
from _utils import gaussian_mutator, random_crossover
from _utils import nsga_crossover, multiroutine_crossover

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import rc

# Matplotlib
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


# LZ6
PM_FUN=None
PROBLEM = LZ6(6)
MINIMIZE = True
DIMENSION = 8
POP_SIZE = 64
N_OBJS = 2
MAX_GENERATION = 1000
STOPPING_RULE = 'max_generation'
MUTATION_RATE = 0.1
MUTATION_U = 0.
MUTATION_ST = 0.2
REF = [1., 1., 1.]


pop = Population(problem=PROBLEM, dim=DIMENSION, size=POP_SIZE, n_objs=N_OBJS,
             max_generation=MAX_GENERATION, reference=REF, minimize=MINIMIZE,
             stopping_rule=STOPPING_RULE)

pop.crossover_fun = nsga_crossover
pop.crossover_fun = multiroutine_crossover(routines=[nsga_crossover,
                                                     random_crossover],
                                           ns=[1.5, 1.])
pop.mutation_fun = gaussian_mutator
pop.selection_fun = pop.compute_front
pop.verbose = True

pop.evolve(metric=True)

theo = PROBLEM.solutions()

# ================================Visualization================================

fig = plt.figure(figsize=(10,4), dpi=100)
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax_metric = fig.add_subplot(1, 2, 2)

# Test surrogate results
final_arc = pop.front
x = []
y = []
z = []
for f in final_arc:
    x.append(f.fitness[0])
    y.append(f.fitness[1])
    z.append(f.fitness[2])


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
#ax.set_zlable(r'$F_3$', labelpad=0)
ax.set(title="Non-Dominated Front")

ax.set_ylim((0., 1.))
ax.set_xlim((0., 1.))
ax.set_zlim((0., 1.))

ax.scatter(theo[:,0], theo[:,1], theo[:,2], c='orangered', s=1.5,
           label="Analytical (Li et Zhang 2009)")
ax.scatter(x, y, z, c='royalblue', s=2.0, label="RBFN Assisted NSGA-II")

# Plot legend.
lgnd = ax.legend(loc='lower left', numpoints=1, fontsize=9)

# change the marker size manually for both lines
lgnd.legendHandles[0]._sizes = [10]
lgnd.legendHandles[1]._sizes = [10]
ax.grid(True, ls=':')

ax.text(ax.get_xlim()[1]-0.1, 0.9, 0.9, 'LZ-6',
        ha='right', va='top', size=14,
        bbox={'boxstyle': 'round',
              'ec': (0, 0, 0),
              'fc': (1, 1, 1),
              }
        )

ax.invert_yaxis()

# ================================ Metrics ====================================
ax_hypervol = ax_metric.twinx()

# Calculate hypervolume coverage
hypervol_cov = [hv/pop.hypervol_ana for hv in pop.hypervol]

# Title
ax_metric.set_title('Hypervolume Metrics')

# Ranges
ax_metric.set_xlim((0, pop.generation))
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