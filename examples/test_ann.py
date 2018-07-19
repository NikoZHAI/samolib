#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 18:53:40 2018

@author: niko
"""

import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from multioutput import MultiOutputRegressor
from benchmarks import kursawe
from _utils import gaussian_mutator, nsga_crossover
from premade import GOMORS

b = np.random.uniform(-5., 5., (20000, 3))
y = np.array([kursawe(bb) for bb in b])

# Kursawe
PM_FUN = kursawe
MINIMIZE = True
DIMENSION = 3
N_OBJS = 2
POP_SIZE = 64
MAX_GENERATION = 40
MAX_EPISODE = 60
MUTATION_RATE = 0.1
MUTATION_U = 0.
MUTATION_ST = 0.7
REF = [-14., 0.1]

pop = GOMORS(dim=DIMENSION, size=POP_SIZE, n_objs=N_OBJS, fitness_fun=PM_FUN,
             max_generation=MAX_GENERATION, max_episode=MAX_EPISODE,
             reference=REF, minimize=MINIMIZE)
pop.mutation_fun = gaussian_mutator
pop.bounds.append([-5., 5.])
pop.crossover_fun = nsga_crossover
pop.mutaton_rate = MUTATION_RATE

# Parametrization
params_ea = {'u': MUTATION_U,
             'st': MUTATION_ST,
             'trial_method': 'lhs',
             'trial_criterion': 'cm'}

params_surrogate = \
    {'hidden_layer_sizes': (4, 8),
     'activation': 'relu',
     'solver': 'adam',
     'early_stopping': False,
     'batch_size': 32,
     'warm_start': True,
     'beta_1': 0.9,
     'beta_2': 0.999,
     'epsilon': 1e-12,
     'alpha': 1e-8,
     'learning_rate': 'constant',
     'learning_rate_init': 0.0005,
     'max_iter': 500,
     'verbose': False,
     'no_improvement_tol': 500,
     }

# ===============================Initialization================================

pop.config_surrogate(typ='ANN', params=params_surrogate)
pop.config_gap_opt(at='least_crowded', radius=0.5, size=POP_SIZE,
                   max_generation=2*MAX_GENERATION, selection_fun=None,
                   mutation_fun=None, mutation_rate=None,
                   crossover_fun=None, trial_method='lhs',
                   trial_criterion='cm', u=0., st=0.12)
pop.config_sampling(methods='default', sizes='default', rate='default',
                    candidates='default')

pop.surrogate.fit(b, y)

pop.surrogate.benchmark(bounds=np.repeat(pop.bounds, 3, axis=0), dim=3, obj_fun=kursawe)
