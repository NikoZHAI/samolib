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
from surrogates import heuristic_optimizer

from sklearn.gaussian_process.kernels import RBF, WhiteKernel as W,\
                                             ConstantKernel as C

b = np.random.uniform(-5., 5., (500, 3))
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

kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e1)) + \
         W(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-5))

params_surrogate = \
    {'kernel': kernel,
     'alpha': 0.,
     'optimizer': heuristic_optimizer,
     'normalize_y': True,
     }

# ===============================Initialization================================

pop.config_surrogate(typ='kriging', params=params_surrogate)
pop.config_gap_opt(at='least_crowded', radius=0.5, size=POP_SIZE,
                   max_generation=2*MAX_GENERATION, selection_fun=None,
                   mutation_fun=None, mutation_rate=None,
                   crossover_fun=None, trial_method='lhs',
                   trial_criterion='cm', u=0., st=0.12)
pop.config_sampling(methods='default', sizes='default', rate='default',
                    candidates='default')

m1 = pop.surrogate.models[0]
m2 = pop.surrogate.models[1]

m1.fit(b, y[:, 0].reshape((-1, 1)))
m2.fit(b, y[:, 1].reshape((-1, 1)))

pop.surrogate.benchmark(bounds=np.repeat(pop.bounds, 3, axis=0), dim=3, obj_fun=kursawe)