#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:16:30 2018

@author: niko

"""

import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from multioutput import MultiOutputRegressor
from benchmarks import load_theo, kursawe
from _utils import gaussian_mutator, nsga_crossover, find_duplication
from premade import GOMORS
from surrogates import heuristic_optimizer
import matplotlib.pyplot as plt
from matplotlib import rc

from sklearn.gaussian_process.kernels import RBF, WhiteKernel as W,\
                                             ConstantKernel as C

# Matplotlib
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


theo = load_theo('pareto_Kursawe.txt').as_matrix()


# Kursawe
PM_FUN = kursawe
MINIMIZE = True
DIMENSION = 3
N_OBJS = 2
POP_SIZE = 64
MAX_GENERATION = 40
MAX_EPISODE = 60
STOPPING_RULE = 'max_eval'
MUTATION_RATE = 0.1
MUTATION_U = 0.
MUTATION_ST = 0.3
REF = [-14., 0.1]

pop = GOMORS(dim=DIMENSION, size=POP_SIZE, n_objs=N_OBJS, fitness_fun=PM_FUN,
             max_generation=MAX_GENERATION, max_episode=MAX_EPISODE,
             reference=REF, minimize=MINIMIZE, stopping_rule=STOPPING_RULE)
pop.mutation_fun = gaussian_mutator
pop.bounds.append([-5., 5.])
pop.crossover_fun = nsga_crossover
pop.mutaton_rate = MUTATION_RATE

# Parametrization
params_ea = {'u': MUTATION_U,
             'st': MUTATION_ST,
             'trial_method': 'lhs',
             'trial_criterion': 'cm'}

kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e1)) + \
         W(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-5))

params_surrogate = \
    {'kernel': kernel,
     'alpha': 0.,
     'optimizer': heuristic_optimizer,
     'normalize_y': True,
     }

# ===============================Initialization================================

pop.config_surrogate(typ='kriging', params=params_surrogate, n_process=2)
pop.config_gap_opt(at='least_crowded', radius=0.5, size=POP_SIZE,
                   max_generation=2*MAX_GENERATION, selection_fun=None,
                   mutation_fun=None, mutation_rate=None,
                   crossover_fun=None, trial_method='lhs',
                   trial_criterion='cm', u=0., st=0.12)
pop.config_sampling(methods='default', sizes='default', rate='default',
                    candidates='default')

pop.run(params_ea=params_ea, params_surrogate=params_surrogate, theo=theo)

# ================================Visualization================================
fig, (ax, ax_metric) = plt.subplots(1, 2, figsize=(10,4), dpi=100)

# Test surrogate results
final_arc = pop.true_front
x = []
y = []
for f in final_arc:
    x.append(f.fitness[0])
    y.append(f.fitness[1])


eq1 = r'$\displaystyle f_1=\sum_{i=1}^2' \
      r'\left[-10exp\left(-0.2\sqrt{(x_i^2+x_{i+1}^2)}\right)\right]$'
eq2 = r'$\displaystyle f_2=\sum_{i=1}^3' \
      r'\left[|x_i|^{0.8}+5sin(x_i^3)\right], x_i\in\left[-5,5\right]$'
#ax.text(-20., -12., eq1+'\n'+eq2, fontsize=12, size=12,
#        ha="left", va="bottom",
#        bbox=dict(boxstyle="square", fc=(1., 1., 1., 0.8), ec='#a5a5a5'))
ax.set_xlabel(r'$F_1$')
ax.set_ylabel(r'$F_2$', labelpad=0)
ax.set(title="Non-Dominated Front")

ax.set_ylim((-12, 0.5))

ax.scatter(theo[:,0], theo[:, 1], c='orangered', s=1.5,
           label="Analytical (F. Kursawe 1991)")
ax.scatter(x, y, c='royalblue', s=2.0, label="ANN Assisted NSGA-II")

# Plot legend.
lgnd = ax.legend(loc='lower left', numpoints=1, fontsize=9)

# change the marker size manually for both lines
lgnd.legendHandles[0]._sizes = [10]
lgnd.legendHandles[1]._sizes = [10]
ax.grid(True, ls=':')

ax.text(ax.get_xlim()[1]-0.2, -0.2, 'Kursawe',
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
ax_metric.set_ylim((0., 10.))
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
y_hat = pop.surrogate.predict(np.array([i.gene for i in pop.sampled_archive]))
y = np.array([i.fitness for i in pop.sampled_archive])

n_sampled_arch = pop.sampled_archive.__len__()
dups = find_duplication(pop.sampled_archive)
n_dups = dups.__len__()
print('Total evals: %i, sampled archive size: %i, duplicated: %i, front size: %i' % (
        n_sampled_arch - n_dups, n_sampled_arch, n_dups, pop.true_front.__len__()))

