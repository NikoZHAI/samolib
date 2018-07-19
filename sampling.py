#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 21 09:42:07 2018

@author: niko
"""

import numpy as np
import copy as cp
from ea import Individual
from _utils import calc_hypervol, sort_by_fitness

class SamplingMixin(object):
    """ Mixins for adaptive sampling
    """
    def __init__(self, *args, **kwargs):
        self.sampled_archive = []
        return None

    def _render_sampling_method_by_name(self, name):
        """ Render corresponding sampling methods given a method name

        Parameters
        ----------
        name : str
            Method name

        Returns
        -------
        method : callable
            Sampling method
        """
        _s = name.lower()
        if 'random' in _s:
            return self.random_sampling
        elif 'mini_x_dist' in _s:
            return self.max_mini_x_dist_sampling
        elif 'mini_y_dist' in _s:
            return self.max_mini_y_dist_sampling
        elif 'hv_improvement' in _s:
            return self.hv_improvement_sampling
        else:
            raise ValueError('Sampling method % not found or not'
                             ' implemented yet...' % name)
        return None

    def render_sampling_methods(self, method_list):
        """ Render corresponding sampling methods given a list

        Parameters
        ----------
        method_list : {list of str or callables}
            Method names or implemented functions

        Returns
        -------
        methods : a list of callables
            Sampling methods
        """
        methods = []
        for _m in method_list:
            if hasattr(_m, '__call__'):
                methods.append(_m)
            elif type(_m) is str:
                methods.append(self._render_sampling_method(name=_m))
            else:
                raise ValueError('Sampling method % not found or not'
                                     ' implemented yet...' % _m)
        return methods

    def random_sampling(self, size=1, sample_rate=0.1,
                        candidates='decision_space', **kwargs):
        """ Random sampling in the descision space or the surrogate's front
        """
        if np.random.uniform() < sample_rate:
            if candidates in ['decision_space', 'domain']:
                return [Individual(dim=self.dim, bounds=self.bounds, \
                        trial_method='random', **kwargs) for i in range(size)]
            elif candidates.__class__ is str:
                return cp.deepcopy(np.random.choice(self.render_pop(candidates),
                                                    size=size).tolist())
            elif candidates.__class__.__name__ == 'list':
                return cp.deepcopy(np.random.choice(candidates, size=size))
            else:
                raise NotImplementedError('Random sample from out of the '
                                          'decision space or the surrogate\'s'
                                          ' front is not supported...')
        else:
            return []

    def hv_improvement_sampling(self, size=1, sample_rate=99.,
                                candidates='Pm_hat', **kwargs):
        """ Sampling by maximizing hypervolume improvement (exhaustive search)
        """
        if np.random.uniform() < sample_rate:
            _candidates = self.render_pop(candidates=candidates)
        else:
            return []

        if _candidates.__len__() <= size:
            return _candidates

        _previous = self.true_front

        # Sort the individuals in the current true front on one axis
        sort_by_fitness(tosort=_previous, obj=0, reverse=self.minimize)

        # Extract fitness from the true front individuals to form front_matrix
        front_matrix = np.array([[f for f in i.fitness] for i in _previous])

        # Calculate the current hypervolume given the reference
        hv_init = calc_hypervol(self.reference, front_matrix)

        hvs = []
        for f in _candidates:
            _ = [f]
            _.extend(_previous)

            sort_by_fitness(tosort=_, obj=0, reverse=self.minimize)

            front_matrix = np.array([[f for f in i.fitness] for i in _])

            hvs.append(calc_hypervol(self.reference, front_matrix))

        diffs = np.subtract(hvs, hv_init)
        inds = np.argpartition(diffs, -size)[-size:]

        return [_candidates[i] for i in inds]

    def max_mini_x_dist_sampling(self, size=1, sample_rate=99.,
                                 candidates='front', **kwargs):
        """ Sampling by maximizing minimum domaine Euclidean distance
        """
        if np.random.uniform() < sample_rate:
            _candidates = self.render_pop(candidates=candidates)
        else:
            return []

        if _candidates.__len__() <= size:
            return _candidates

        cand_xs = self.render_features(pop=_candidates)
        sampled_xs = self.render_features(pop=self.sampled_archive)

        inds = self._sample_max_mini_dist(size=size, candidates=cand_xs,
                                          front=sampled_xs)
        return [_candidates[i] for i in inds]

    def max_mini_y_dist_sampling(self, size=1, sample_rate=99.,
                                 candidates='front', **kwargs):
        """ Sampling by maximizing minimum domaine Euclidean distance
        """
        if np.random.uniform() < sample_rate:
            _candidates = self.render_pop(candidates=candidates)
        else:
            return []

        if _candidates.__len__() <= size:
            return _candidates

        cand_ys = self.render_targets(pop=_candidates)
        sampled_ys = self.render_targets(pop=self.sampled_archive)

        inds = self._sample_max_mini_dist(size=size, candidates=cand_ys,
                                          front=sampled_ys)
        return [_candidates[i] for i in inds]

    def _sample_max_mini_dist(self, size, candidates, front):
        dists = [np.subtract(front, c).__pow__(2).sum(axis=1).min() \
                 for c in candidates]

        return np.argpartition(dists, -size)[-size:]

