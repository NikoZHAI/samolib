#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:22:07 2018

@author: niko
"""

import numpy as np
import copy as cp
from pyDOE import lhs
from _utils import rend_k_elites, random_crossover, calc_cwd_dist
from _utils import calc_hypervol, sort_by_fitness
from _utils import Cache, _no_fit_fun, construct_problem

class Individual(object):

    def __init__(self, dim=3, bounds=[[-5., 5.]], name='Undefined',
                 fitness=np.array([np.inf]), trial_method='random',
                 gene=None, **kwargs):
        self.dimension = dim
        self._gene = 'vector of {} elements (parameters)...'.format(dim)
        self.name = name
        self._dominated = False
        self.bounds = np.add(bounds, [0., 1e-12])
        self.fitness = fitness
        self.acceptance = 0.
        self.trial_method = trial_method

        # Generate bounds
        if self.bounds.__len__() == 1 and self.bounds.shape.__len__() == 2:
            self.bounds = np.repeat(self.bounds, self.dimension, axis=0)
        elif dim == self.bounds.shape[0]:
            pass
        elif dim != self.bounds.shape[0]:
            raise NotImplementedError("Boundary and gene dimension "
                                      "does not match...")
        else:
            raise NotImplementedError("Problems with no boundary or "
                                      "single boundary "
                                      "are not implemented yet...")

        # Make trials
        if trial_method == 'random':
            self.gene = np.array([np.random.uniform(*b) for b in self.bounds])

        elif trial_method == 'lhs':
            self.gene = np.array([g * (b[1] - b[0]) + b[0] \
                                  for g, b in zip(gene, self.bounds)])
        elif trial_method == None:
            self.gene = gene
        else:
            raise NotImplementedError('%s trial design method is not '
                                      'implemented yet' % (self.trial_method))
        return None

    def calc_fitness(self, fun):
        self.fitness = fun.__call__(self.gene)
        return self.fitness

    def mutate(self, routine, rate, **kwargs):
        doomed = np.random.sample(self.dimension).__lt__(rate)
        if doomed.any():
            self.gene = routine(self.gene, self.bounds, doomed, **kwargs)
        return self.gene

    def get_fitness(self, obj):
        return self.fitness.__getitem__(obj)

    def update_dist(self, right, left, obj):
        self._dist += right.fitness[obj] - left.fitness[obj]
        return self._dist

    def to_dict(self):
        return self._to_dict(level=-1, d={})

    def _to_dict(self, level=-1, d={}):
        if level == -1:
            d[self.gene[level]] = self.fitness
        else:
            d = {self.gene[level]: d}
            if level == -self.dimension:
                return d
        return self._to_dict(level-1, d)

    def __lt__(self, other):
        _better = self.fitness.__lt__(other.fitness)
        _crossed = self.fitness.__eq__(other.fitness)
        return _better.all() or (_crossed.any() and _better.any())

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return other.__lt__(self)

    def __ge__(self, other):
        return other.__lt__(self) or self.__eq__(other)

    def __eq__(self, other):
        return self.fitness.__eq__(other.fitness).all()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.fitness) + " => " + str(self.gene)


class Population(object):

    def __init__(self, problem=None, dim=3, n_objs=2, size=32,
                 fitness_fun=_no_fit_fun, reference=[],
                 selection_fun=rend_k_elites,
                 mutation_fun=None, mutation_rate=0.1,
                 crossover_fun=random_crossover,
                 stopping_rule='max_generation',
                 bounds=[], max_generation=200, max_eval=8000,
                 minimize=True, *args, **kwargs):

        if problem is None:
            problem = construct_problem(dim=dim, n_objs=n_objs,
                                        fun=fitness_fun, bounds=bounds)

        self.problem = problem
        self.dim = problem.dim
        self.n_objs = problem.n_objs
        self.fitness_fun = problem.obj_fun
        self.bounds = problem.bounds

        self.size = size
        self.gene_len = self.dim
        self.generation = 0
        self.verbose = False
        self.stopping_rule = stopping_rule
        self.max_generation = max_generation
        self.max_eval = max_eval
        self.reference = reference
        self.minimize = True

        self.global_pop = []
        self.elites = []
        self.front = []
        self.cache = Cache()
        self.hypervol = []
        self.hypervol_diff = []
        self.hypervol_cov = []

        self.selection_fun = selection_fun
        self.mutation_fun = mutation_fun
        self.mutation_rate = mutation_rate
        self.crossover_fun = crossover_fun

        self.n_process = 1
        return None

    def generate_init(self, trial_method='random',
                      trial_criterion='cm', **kwargs):

        if self.generation == 0:
            self.episode = 0

            local = self.trial(method=trial_method, criterion=trial_criterion)
            for i in local:
                i.calc_fitness(fun=self.fitness_fun)

            self.global_pop = cp.deepcopy(local)
            self.elites = []
            self.front = []
            self.bounds = local[0].bounds

        else:
            # Reset generation for a new round of surrogate assisted search
            self.generation = 0

        return None

    def trial(self, method='random', **kwargs):
        if method == 'random':
            local = [Individual(dim=self.dim, bounds=self.bounds, \
                     trial_method=method, **kwargs) for _i in range(self.size)]
        elif method == 'lhs':
            normalized_trials = self.trial_lhs(criterion='cm')
            local = [Individual(dim=self.dim, bounds=self.bounds, \
                                trial_method=method, gene=g,       \
                                **kwargs) for g in normalized_trials]
        else:
            raise NotImplementedError('%s trial design method is not '
                                      'implemented yet' % (method))
        return local

    def trial_lhs(self, criterion='cm'):
        return lhs(n=self.dim, samples=self.size, criterion=criterion)

    def crossover(self, **kwargs):
        """ Perform crossover in self.front (fake front in surrogate EA)
        """
        self._crossover(parent_pop=self.front,
                        offspring_pop=self.global_pop,
                        mutate=True, calc_fitness=True, **kwargs)
        return None

    def _crossover(self, parent_pop, offspring_pop, mutate=True,
                   calc_fitness=True, **kwargs):

        _params = {'elites': parent_pop, 'gene_len': self.gene_len}

        if 'multiroutine' in self.crossover_fun.__name__:
            arch = self.crossover_fun(_pop_size=self.size, params=_params,
                                      **kwargs)
        else:
            arch = self.crossover_fun(elites=parent_pop, pop_size=self.size,
                                      gene_len=self.gene_len, **kwargs)

        self._post_crossover(pop=offspring_pop, archive=arch, mutate=mutate,
                             calc_fitness=calc_fitness, **kwargs)

        return offspring_pop

    def _post_crossover(self, pop, archive, mutate, calc_fitness, **kwargs):
        if mutate and calc_fitness:
            for i, _gene in zip(pop, archive):
                i.gene = np.array(_gene)
                i.mutate(self.mutation_fun, self.mutation_rate, **kwargs)
                i.calc_fitness(fun=self.fitness_fun)

        elif mutate:
            for i, _gene in zip(pop, archive):
                i.gene = np.array(_gene)
                i.mutate(self.mutation_fun, self.mutation_rate, **kwargs)

        elif calc_fitness:
            for i, _gene in zip(pop, archive):
                i.gene = np.array(_gene)
                i.calc_fitness(fun=self.fitness_fun)

        else:
            for i, _gene in zip(pop, archive):
                i.gene = np.array(_gene)

        return None

    def select(self, rank=None, **kwargs):
        self.elites = self.compute_front(**kwargs)
        return None

    def cache(self, individual, **kwargs):
        self.cached.append(individual)
        return None

    def evolve(self, early_stopping=False, **kwargs):
        self.generate_init(**kwargs)

        while self.generation <= self.max_generation:
            self.select(**kwargs)
            self.update_front(**kwargs)
            self.crossover(**kwargs)
            if self.verbose:
                print(self.generation)
            self.generation += 1
        return None

    def update_front(self, **kwargs):
        # Update current front
        updates = []
        if self.generation == 0:
            self.front = self.elites.copy()
            return None
        else:
            current_front = self.front

        for f in self.elites:
            dominated = np.less_equal(current_front, f)
            if dominated.any():
                continue
            else:
                dominating = np.less(f, current_front)
                updates.append(cp.copy(f))
                to_remove = np.where(dominating)[0]
                i_pop = 0
                for r in to_remove:
                    current_front.pop(r-i_pop)
                    i_pop += 1

        self.front.extend(updates)
        return None

    def compute_front(self, pop=None, **kwargs):
        local = self.global_pop if pop is None else pop
        front = []

        # Compute front of the current generation
        for i in local:
            if np.less(local, i).any() or np.equal(front, i).any():
                continue
            else:
                front.append(cp.copy(i))
        return front

    def render_features(self, pop='global'):
        if pop == 'global':
            _pop = self.global_pop
        elif pop == 'front':
            _pop = self.front
        elif pop.__class__.__name__ == 'list':
            _pop = pop
        else:
            raise NotImplementedError('Render features (genes) from sets other'
                                      ' than global_pop, front, or true_front '
                                      'not supported...')

        return np.array([i.gene for i in _pop])

    def render_targets(self,  pop='global'):
        if pop == 'global':
            _pop = self.global_pop
        elif pop == 'front':
            _pop = self.front
        elif pop.__class__.__name__ == 'list':
            _pop = pop
        else:
            raise NotImplementedError('Render targets (fitnesses) from sets '
                                      'other than global_pop, front, or '
                                      'true_front not supported...')

        return np.array([i.fitness for i in _pop])

    def find_least_crowded(self, candidates='front', **kwargs):
        """ Find the least crowded solution in a set of non-dominated
            solutions (PM-evaluated by default)
        """
        try:
            _candidates = self.render_pop(candidates=candidates)
        finally:
            pass

        calc_cwd_dist(pop=_candidates, **kwargs)

        _init = True
        for i in _candidates:
            if i._on_edge:
                continue

            if _init:
                _least_crowded = i
                _init = False

            if _least_crowded._dist < i._dist: _least_crowded = i

            self._least_crowded = cp.deepcopy(i)

        return self._least_crowded

    def hypervol_metric(self, front, ref, analytical=False, minimize=True):
        """ Calculate hypervolume metrics of the given front with an explicit
            reference point
        """
        # Sort the individuals in the current true front on one axis
        sort_by_fitness(tosort=front, obj=0, reverse=minimize)

        # Extract fitness from the true front individuals to form front_matrix
        front_matrix = np.array([[f for f in i.fitness] for i in front])

        # Calculate the current hypervolume given the reference
        self.hypervol.append(calc_hypervol(ref, front_matrix))

        if  analytical is False:
            self.hypervol_ana = 0.
            return None

        elif self.hypervol.__len__() == 1:
            # Sorting order of the analytical Pareto optimals, ascending if
            # maximization problem and vice versa
            order_ = -1 if minimize else 1

            # Sort the analytical Pareto Optimals
            _a = analytical[analytical[:, 0].argsort()[::order_]]

            # Calculate the hypervolume between the reference and analyticals
            self.hypervol_ana = calc_hypervol(ref, _a)

        else:
            pass

        # If analytical solutions given, calculate hv difference and coverage
        hv_diff = self.hypervol_ana - self.hypervol[-1]
        self.hypervol_diff.append(hv_diff)
        self.hypervol_cov.append(self.hypervol[-1]/self.hypervol_ana)

        return None

    def _render_pop_by_name(self, name='global'):
        """ Render a pop ( or a frontier or a set of solutions) given its name
        """
        if not(name.__class__ is str):
            raise TypeError('name should be a "str"... ')

        # Renders self.global_pop
        global_list = ['global', 'population', 'pop']

        # Renders self.front
        front_list = ['front', 'hat']

        _s = name.lower()

        if any(_pattern in _s for _pattern in global_list):
            return self.global_pop
        elif any(_pattern in _s for _pattern in front_list):
            return self.front
        else:
            raise ValueError('Name of the population %s not found....' % name)

        return None

    def render_pop(self, candidates='global'):
        if candidates.__class__ is list:
            return candidates
        elif candidates.__class__ is str:
            return self._render_pop_by_name(name=candidates)
        else:
            raise ValueError('"candidates" must be a string indicating the '
                             'desired candidate set OR a list of candidates %'
                             ' is not supported...' % candidates)
        return None

    def stop(self):
        if self.stopping_rule == 'max_generation':
            return self.max_generation_termination()
        elif self.stopping_rule == 'max_eval':
            return self.max_eval_termination()
        else:
            raise ValueError("Unknown stopping_rule: %s" % self.stopping_rule)
        return True

    def max_eval_termination(self):
        terminate = False
        if self.problem.n_evals >= self.max_eval: terminate = True
        return terminate

    def max_generation_termination(self):
        terminate = False
        if self.max_generation <= self.generation: terminate = True
        return terminate

