#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 21 09:42:07 2018

@author: niko
"""

import numpy as np
import copy as cp
from surrogates import NeuralNet, Kriging, RBFN
from sklearn.svm import NuSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from _utils import calc_cwd_dist, remove_duplication
from multiprocessing.pool import Pool


class SurrogateEAMixin(object):
    """ Mixins for Surrogate Assisted EAs
    """
    def __init__(self, max_episode=60, embedded_ea=None, *args, **kwargs):
        self.true_front = []
        self.episode = 0
        self.max_episode = max_episode
        self.embedded_ea = embedded_ea
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

        # Renders self.true_front
        true_front_list = ['pm', 'true']

        _s = name.lower()

        if any(_pattern in _s for _pattern in global_list):
            return self.global_pop
        elif any(_pattern in _s for _pattern in front_list):
            return self.front
        elif any(_pattern in _s for _pattern in true_front_list):
            return self.true_front
        else:
            raise ValueError('Name of the population % not found....' % name )

        return None

    def generate_init(self, trial_method='random',
                      trial_criterion='cm', **kwargs):

        if self.generation == 0:
            self.episode = 0

            local = self.trial(method=trial_method, criterion=trial_criterion)

            for i in local:
                if self.cache.find(i, overwrite=True) is None:
                    i.calc_fitness(fun=self.fitness_fun)

            self.global_pop = local
            self.elites = []
            self.front = []
            self.true_front = []
            self.bounds = local[0].bounds

            self.generation = 1

        return None

    def recalc_fitness_with(self, fun):
        # Re-calculate fitness in population
        for i in self.global_pop:
            i.calc_fitness(fun)

        # Re-calculate fitness in current front
        for i in self.front:
            i.calc_fitness(fun)

        # Re-evaluate the pareto front with PM results
        self.front = self.compute_front(on=self.front)

        return None

    def update_true_front(self, front=None):
        """Use the surrogate front to update the current true Pareto front.
           The inaccuracy of the surrogate (overfitting or underfitting) may
           result in fake Pareto optimals which override the true ones. In
           order to save those true values, we build this so-called true_front
           archive to cache the PM's true Pareto optimal.
        """

        front = self.front if front is None else front
        self.true_front.extend(front)
        self.true_front = self.compute_front(pop=self.true_front)
        return None

    def crossover(self, **kwargs):
        """ Perform crossover in self.front (fake front in surrogate EA)
        """
        self._crossover(parent_pop=self.front,
                        offspring_pop=self.global_pop,
                        mutate=True, calc_fitness=False, **kwargs)
        return None

    def crossover_in_true_front(self, **kwargs):

        self._crossover(parent_pop=self.true_front,
                        offspring_pop=self.global_pop,
                        mutate=True, calc_fitness=False, **kwargs)
        return None

    def render_features(self, pop='global'):
        if pop == 'global':
            _pop = self.global_pop
        elif pop == 'front':
            _pop = self.front
        elif pop == 'true_front':
            _pop = self.true_front
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
        elif pop == 'true_front':
            _pop = self.true_front
        elif pop.__class__.__name__ == 'list':
            _pop = pop
        else:
            raise NotImplementedError('Render targets (fitnesses) from sets '
                                      'other than global_pop, front, or '
                                      'true_front not supported...')

        return np.array([i.fitness for i in _pop])

    def expensive_eval(self, candidates, verbose=True):
        """ Expensively evaluate the candidates' fitnesses
        """
        if self.n_process <= 1:
            new_candidates = [i for i in candidates if self._expensive_eval(i)
                              is not None]
        else:
            # parallel
            no_dup = remove_duplication(candidates)
            new_candidates = self._expensive_eval_parallel(no_dup)

        if self.verbose and verbose:
            print("New expensively evaluations: %s" % len(new_candidates))

        return new_candidates

    def _expensive_eval(self, i):

        if self.cache.find(i, overwrite=True) is None:
            i.calc_fitness(fun=self.fitness_fun)

            self.cache.save(i)

            if hasattr(self, 'sampled_archive'): self.sampled_archive.append(i)
            return i

        return None

    def _expensive_eval_parallel(self, candidates):
        pool = Pool(self.n_process)
        rs = [pool.apply_async(self._expensive_eval, i) for i in candidates]
        candidates = [r.get() for r in rs]
        pool.close()
        pool.join()
        return candidates

    def cheap_eval(self, candidates='global'):
        """ Evaluate the candidates with the surrogate model
        """
        if candidates == 'global':
            candidates = self.global_pop
        else:
            pass

        [i.calc_fitness(fun=self.surrogate.render_fitness) for i in candidates]
        return None

    def train_surrogate(self, samples=None):
        """ Train the surrogate(s)
        """
        if samples == []: return self.surrogate

        # Retraining of the surrogate
        if self.surrogate._warm_start and samples is not None:
            X = self.render_features(pop=samples)
            y = self.render_targets(pop=samples)
        else:
            X = self.render_features(pop=self.sampled_archive)
            y = self.render_targets(pop=self.sampled_archive)
        self.surrogate.fit(X, y)

        return self.surrogate

    def _progressive_revolution(self):
        """ Progressive revolution
        """
        new_pop = self._crossover(parent_pop=self.true_front,
                                  offspring_pop=cp.deepcopy(self.global_pop),
                                  mutate=True, calc_fitness=False)

        new_front = self.expensive_eval(new_pop)

        self.update_true_front(front=new_front)

        if self.surrogate._warm_start:
            self.train_surrogate(samples=new_pop)
        else:
            self.train_surrogate(samples=self.sampled_archive)

        return None

    def progressive_revolution(self, n_no_improvement, tol):
        """ Performance a progressive revolution if no improvement observed
        """
        deadlock = False
        if len(self.hypervol) < n_no_improvement or n_no_improvement < 2:
            return deadlock

        diff = np.diff(self.hypervol[-n_no_improvement:])
        ratio = np.divide(diff, self.hypervol[-n_no_improvement:-1] + \
                          np.finfo(float).tiny).__abs__()

        if np.less(ratio, tol).all():
            deadlock = True

            if self.verbose:
                print('')
                print('No major improvement was observed, performing '
                      'a progressive revolution...')

            self._progressive_revolution()

        return deadlock

    def find_least_crowded(self, candidates='default', **kwargs):
        """ Find the least crowded solution in a set of non-dominated
            solutions (PM-evaluated by default)
        """
        if candidates in ['default', 'pm', 'PM', 'Pm', 'true_front']:
            _candidates = self.true_front
        elif candidates in ['local', 'front']:
            _candidates = self.front
        else:
            raise ValueError('Can not find the least crowded solution on "%"'
                             % candidates)

        calc_cwd_dist(pop=_candidates, **kwargs)

        _init = True
        for i in _candidates:
            if _init:
                _least_crowded = i
                _init = False

#            if i._on_edge:
#                continue

            if _least_crowded._dist < i._dist: _least_crowded = i

        self._least_crowded = cp.deepcopy(_least_crowded)

        return self._least_crowded

    def config_surrogate(self, typ='ANN', params={}, premade=None,
                         n_models=None, n_process=1, X_scaler=None,
                         y_scaler=None, warm_start=False, **kwargs):
        """ Configurate and initialize a surrogate
        """
        if premade is not None:
            self.surrogate = premade
            return None

        _t = typ.lower()

        if _t in ['ann', 'mlp', 'neural_network', 'neural-network']:
            self.surrogate = NeuralNet(n_objs=self.n_objs, params=params,
                                       n_models=n_models, n_process=n_process,
                                       X_scaler=X_scaler, y_scaler=y_scaler,
                                       warm_start=warm_start)
        elif _t in ['svm', 'svr']:
            self.surrogate = SVR(**params)
        elif _t in ['nusvm', 'nusvr', 'nu-svm', 'nu-svr', 'nu_svm', 'nu_svr']:
            self.surrogate = NuSVR(**params)
        elif _t[:3] == 'rbf':
            self.surrogate = RBFN(n_objs=self.n_objs, params=params,
                                  n_models=n_models, n_process=n_process,
                                  X_scaler=X_scaler, y_scaler=y_scaler,
                                  warm_start=warm_start)
        elif 'tree' in _t:
            self.surrogate = DecisionTreeRegressor(**params)
        elif 'kriging' in _t:
            self.surrogate = Kriging(n_objs=self.n_objs, params=params,
                                     n_models=n_models, n_process=n_process,
                                     X_scaler=X_scaler, y_scaler=y_scaler,
                                     warm_start=warm_start)
        else:
            raise NotImplementedError('Surrogate type % not supported...' %typ)

        return None

    def max_episode_termination(self):
        terminate = False
        if self.max_episode <= self.episode: terminate = True
        return terminate

    def stop(self):
        if self.stopping_rule == 'max_generation':
            return self.max_generation_termination()
        elif self.stopping_rule == 'max_eval':
            return self.max_eval_termination()
        elif self.stopping_rule == 'max_episode':
            return self.max_episode_termination()
        else:
            raise ValueError("Unknown stopping_rule: %s" % self.stopping_rule)
        return True

    def config_embedded_ea(self, **kwargs):
        if self.embedded_ea is None:
            pass
        elif hasattr(self.embedded_ea, "_external_moea"):
            self.embedded_ea_ = \
            self.embedded_ea(problem=self.problem, surrogate=self.surrogate,
                             size=self.size, generation=self.max_generation,
                             **kwargs)
        else:
            raise ValueError("Unknown embedded EA")

        return None

    def evolve_surrogate(self, **params_ea):
        if self.embedded_ea is None:
            self._naive_ea(**params_ea)
        elif hasattr(self.embedded_ea, "_external_moea"):
            self._evolve_embedded_ea(**params_ea)
        else:
            raise ValueError("Unknown embedded EA")

        return None

    def _naive_ea(self, **params_ea):

        while not self.max_generation_termination():
            self.crossover_in_true_front(**params_ea)
            self.cheap_eval(candidates='global')
            self.select(**params_ea)
            self.update_front(**params_ea)

            self.generation += 1

        return None

    def _evolve_embedded_ea(self, **params_ea):

        # Apply crossover in the true front for surrogate-assisted optimization
        # self.crossover_in_true_front(**params_ea)

        # Evolutionary computation over the surrogate
        # Prevent neglect of the first population
        if self.episode < 1:
            self.embedded_ea_.load_external_pop_xf(pop=self.global_pop)
        else:
            self.embedded_ea_.load_external_pop_x(pop=self.global_pop)
        self.embedded_ea_.evolve()
        self.embedded_ea_.export_internal_pop(pop=self.global_pop)

        # Select and update non-dominated solutions
        self.select(**params_ea)
        self.update_front(**params_ea)
        self.generation = self.max_generation

        return None



