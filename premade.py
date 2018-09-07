#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:26:06 2018

@author: niko
"""

import numpy as np
import copy as cp
from ea import Population
from surrogate_ea import SurrogateEAMixin
from local_search import LocalSearchMixin
from sampling import SamplingMixin
from _utils import _no_fit_fun, _no_selection_fun, random_crossover, \
                   gaussian_mutator


def MetaFramework(cls):
    """ Mixed-in class decorator. """
    classinit = cls.__dict__.get('__init__')  # Possibly None.

    # define an __init__ function for the class
    def __init__(self, *args, **kwargs):
        # call the __init__ functions of all the bases
        for base in cls.__bases__:
            base.__init__(self, *args, **kwargs)
        # also call any __init__ function that was in the class
        if classinit:
            classinit(self, *args, **kwargs)

    # make the local function the class's __init__
    setattr(cls, '__init__', __init__)
    return cls


@MetaFramework
class GOMORS(SamplingMixin, LocalSearchMixin, SurrogateEAMixin, Population):
    """ GOMORS framework with customizable surrogate
    """

    def __init__(self, problem=None, dim=None, n_objs=None, size=32,
                 fitness_fun=_no_fit_fun, reference=[],
                 selection_fun=_no_selection_fun,
                 mutation_fun=gaussian_mutator,
                 mutation_rate=0.1,
                 crossover_fun=random_crossover,
                 bounds=[], max_generation=25,
                 stopping_rule='max_eval', max_episode=25,
                 max_eval=400, revolution=True,
                 no_improvement_step_tol=3,
                 improvement_tol=1e-2,
                 verbose=True, *args, **kwargs):

        self.max_eval = max_eval

        self.selection_fun = self.compute_front

        self.revolution = revolution
        self.no_improvement_step_tol = no_improvement_step_tol
        self.improvement_tol = improvement_tol

        self.verbose = verbose

        # Adaptive sampling default configurations: samplers, sizes,
        # sample rates, and populations to sample from.
        self.default_samplers = [self.random_sampling,
                                 self.hv_improvement_sampling,
                                 self.max_mini_x_dist_sampling,
                                 self.max_mini_y_dist_sampling,
                                 self.hv_improvement_sampling,
                                 ]
        self.default_sample_sizes = [1]*5
        self.default_sample_rates = [0.1] + [99.] * 4
        self.default_sample_cadidates = ['decision_space'] + ['pm_hat'] * 3 + \
                                        ['gap']

        return None

    def config_gap_opt(self, at='least_crowded', radius=0.1, size= None,
                       max_generation=None, selection_fun=None,
                       mutation_fun=None, mutation_rate=None,
                       crossover_fun=None, trial_method='lhs',
                       trial_criterion='cm', u=0., st=0.2, **kwargs):
        """ Gap optimization step of GOMORS (Akhtar et Shoemaker, 2016)
        """

        # Construct configuration dict
        self._gap_opt_config = dict(radius=radius, size=size,
                                    max_generation=max_generation,
                                    selection_fun=selection_fun,
                                    mutation_fun=mutation_fun,
                                    mutation_rate=mutation_rate,
                                    crossover_fun=crossover_fun,
                                    trial_method=trial_method,
                                    trial_criterion=trial_criterion,
                                    u=u, st=st)
        self._gap_opt_config.update(kwargs)
        return self._gap_opt_config

    def gap_opt(self, at):
        """ Perform a 'Gap Optimization' proposed in Akhtar et Shoemaker 2016
        """
        # Find the least crowded solution in the Pm as the start point
        # in local search
        if at in ['default', 'least_crowded']:
            _loc = self.find_least_crowded(candidates='Pm')
        else:
            _loc = at

        # Perform a local search
        self.local_search_front = self.local_search(at=_loc,
                                                    **self._gap_opt_config)
        return self.local_search_front

    def config_sampling(self, methods='default', sizes='default',
                        sample_rates='default', candidates='default',
                        **kwargs):
        """ Sample individuals for expensive evaluations

        Parameters
        ----------
        methods : {str, list}, default 'default'
                  Sampling strategies, if 'default', applies the 5 sampling
                  rules discussed in Akhtar et Shoemaker 2016. If list
                  , should be ['random', 'hv_imporvement', 'max_mini_x_dist'
                  , 'max_mini_y_dist', 'gap_hv_improvement'] or its subsets.

        sizes : {str, int, list}, default 'default'
                Sampling sizes, if 'default', one candidate will be sampled
                via each sampling method, equivalent to 1 or [1]*n_methods. If
                integer n, n candidates will be sampled via each sampling
                method, equivalent to [n]*n_methods. If list, k samples
                will be sampled via sampling method at corresponding position.

        sample_rates : {str, list, float}, default 'default'
                      Sampling rate, the probability that a candidate is
                      sampled via a method.
                      If 'default', the corresponding sampling
                      rate of each sampling methode will be {'random': 0.1,
                      'all_the_rest': 1.0}. Can be assigned by a list of
                      numerical values. Note that sample_rate >= 1 means always
                      sample, sample_rate <= 0 means never sample through a
                      particular method.

        candidates : {str, list}, default 'default'
                     The population where the sampling rules should be applied.
                     If default, equivalent to ['Pm', 'Pm_hat', 'Pm_hat',
                     'Pm_hat', 'Pm_gap'].

        **kwargs : Optional

        Returns
        -------
        self._adaptive_sampling_config : iterable
                  Iterable of adaptive sampling workflow
        """
        # Generate a sequence of sampling methods
        if methods == 'default':
            _methods = self.default_samplers
        elif methods.__class__ is list:
            _methods = self.render_sampling_methods(method_list=methods)
        else:
            raise ValueError('Argument method should either be a list or '
                             '"default"... "%" is not supported' % methods)

        # Generate a sequence of sampling sizes
        if sizes == 'default':
            _sizes = self.default_sample_sizes
        elif type(sizes) is int and sizes >= 0:
            _sizes = [sizes] * _methods.__len__()
        elif sizes.__class__ in [list, tuple]:
            if np.greater_equal(sizes, 0).all():
                _sizes = sizes
            else:
                raise ValueError('Sample size must be greater than zero...')
        else:
            raise ValueError('Argument "sizes" must be str, int or list...')

        # Generate a sequence of sample rates
        if sample_rates == 'default':
            _sample_rates = self.default_sample_rates
        elif type(sizes) is int and sizes >= 0:
            _sample_rates = [sample_rates] * _methods.__len__()
        elif sizes.__class__ in [list, tuple]:
            _sample_rates = sample_rates
        else:
            raise ValueError('Argument "sample_rates" must be str, float '
                             'or list...')

        # Generate a sequence of candidate population
        if candidates == 'default':
            _candidates = self.default_sample_cadidates
        elif candidates.__class__ is list:
            _candidates = candidates
        else:
            raise ValueError('Argument "candidates" must be str, or list...')

        _len = len(_methods)

        for _e in [_sizes, _sample_rates, _candidates]:
            if len(_e) != _len:
                raise RuntimeError('The number of sampling methods doesn\'t '
                                   'match the number of one or more element of'
                                   ' the followings: sampling sizes, rates, or'
                                   ' sampling candidates...')

        self._adaptive_sampling_config = zip(_methods, _sizes, _sample_rates,
                                             _candidates)
        return self._adaptive_sampling_config

    def sample_for_expensive_evals(self, config=None):
        """ Sample points for expensive evaluations
        """
        config = self._adaptive_sampling_config if config == None else config

        candidates = []
        for _m, _s, _r, _p in cp.deepcopy(config):
            candidates += _m(size=_s, sample_rate=_r, candidates=_p)

        # Remove dups
        candidates = [c for c in candidates if self.cache.find(c) is None]

        if self.verbose:
            print("Newly sampled points: %s" % len(candidates))

        return candidates


    def run(self, params_ea=None, params_surrogate=None, theo=None):

        # ========================== Initialization =========================

        # Generation of first population
        self.generate_init(**params_ea)
        self.select(**params_ea)
        self.update_front(**params_ea)
        self.update_true_front()
        self.sampled_archive.extend(cp.deepcopy(self.global_pop))
        self.cache.save(self.sampled_archive)

        # Initialize and train the surrogate
        self.surrogate.fit(self.render_features(pop=self.sampled_archive),
                           self.render_targets(pop=self.sampled_archive))

        # Configure the embedded EA
        self.config_embedded_ea(**params_ea)

        # =========================Meta Modelling==========================

        while not self.stop():

            # Optional Crossover to formulate new population
            if self.episode > 1: self.crossover_in_true_front()

            # Evolutional computation on the surrogate
            self.evolve_surrogate(**params_ea)

            print("Episode: %s, Total expensive evaluations: %s, "
                  "True front size: %s, "
                  "Surrogate Front size: %s" %
                  (self.episode, self.problem.n_evals, len(self.true_front),
                   self.front.__len__()))

            # Gap optimization
            self.gap_opt(at='least_crowded')

            # Re-evaluate the surrogate-sampled individuals using the PM
            newly_sampled = self.sample_for_expensive_evals()
            candidates = self.expensive_eval(candidates=newly_sampled)
            new_front = self.compute_front(pop=candidates)
            self.update_true_front(front=new_front)

            # Retraining of the surrogate
            self.train_surrogate(samples=candidates)

            # Calculate hypervolume metrics
            self.hypervol_metric(self.true_front, ref=self.reference,
                                 analytical=theo)

            # Detect no improvement in hypervolume (a deadlock)
            if self.revolution:
                if self.progressive_revolution(self.no_improvement_step_tol, 1e-2):
                    # Calculate hypervolume metrics
                    self.hypervol.pop()
                    self.hypervol_pos.pop()
                    self.hypervol_index.pop()
                    self.hypervol_metric(self.true_front, ref=self.reference,
                                         analytical=theo)
                if self.stop(): return self

            # Reset the surogate's generation counter
            self.generation = 1
            self.episode += 1
            self.front = [] #self.true_front.copy()

        return self

