#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 18:34:38 2018

@author: niko
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from benchmarks import Shakespears
from _utils import gaussian_mutator, random_crossover, rend_k_elites

from ea import Population


pop = Population(problem=Shakespears(), size=500, stopping_rule='max_eval',
                 max_eval=50*1000, dtype=int, mixinteger=True, mutation_rate=0.12,
                 mutation_fun=gaussian_mutator, crossover_fun=random_crossover)

pop.selection_fun = rend_k_elites(pop=pop, k=2)

pop.evolve()

