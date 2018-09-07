#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:22:07 2018

@author: niko
"""

import warnings

import numpy as np
from scipy.special import binom
import copy as cp
from deap.tools._hypervolume.hv import hypervolume
from operator import methodcaller, attrgetter
from functools import partial
from sklearn.base import BaseEstimator, TransformerMixin


def _no_fit_fun(*args, **kwargs):
    raise NoFitnessFunError()
    return None


def _no_selection_fun(*args, **kwargs):
    raise NoSelectionFunError()
    return None


def construct_problem(dim, n_objs, fun, bounds):
    p = UnconstrainedProblem(dim=dim, n_objs=n_objs, fun=fun)
    bounds = np.array(bounds)

    # Generate bounds
    if bounds.ndim == 1 and bounds.__len__() == 2:
        bounds = np.repeat(bounds, dim, axis=0)
    elif dim == bounds.shape[0]:
        pass
    elif dim != bounds.shape[0]:
        raise NotImplementedError("Boundary and gene dimension "
                                  "does not match...")
    else:
        raise NotImplementedError("Problems with no boundary or "
                                  "single boundary "
                                  "are not implemented yet...")

    p.bounds = bounds
    return p


def rend_k_elites(pop, k=1, **kwargs):
    if pop.n_objs > 1:
        _fun = pop.compute_front
    else:
        _fun = partial(_rend_k_elites_so, pop=pop, k=k)

    return _fun


def _rend_k_elites_so(pop, k=1):
    elites = sort_by_fitness(tosort=pop.global_pop, obj=0, reverse=False)[:k]

    return elites.copy()

def reject_acceptance(elites, gene_len, **kwargs):
    p = np.random.uniform()
    picked = np.random.choice(elites)
    rejected = []
    while (not picked._on_edge) and (p > 0):
        if picked in rejected:
            picked = np.random.choice(elites)
            continue
        else:
            p -= picked.acceptance
            picked = np.random.choice(elites)
            rejected.append(picked)
    return picked


def calc_nsga_acceptance(pop, **kwargs):
    """ Calculate the NSGA-II acceptance of each individual in pop

        pop should be assigned by a pop_list[region] fashion.
    """
    total_dist = calc_cwd_dist(pop, **kwargs)

    if pop.__len__() < 5:
        [i.__setattr__('acceptance', 1./pop.__len__()) for i in pop]
        return None

    for i in pop:
        i.acceptance = np.divide(i._dist, total_dist)

    return None


def calc_cwd_dist(pop, **kwargs):
    """ Calculate the crowding distance of each individual in pop

        pop should be assigned by a pop_list[region] fashion.
    """
    if pop.__len__() <= 2:
        for i in pop:
            i.__setattr__('_dist', 0.)
            i.__setattr__('_on_edge', True)
        return None

    for i in pop:
        i._dist = 0.
        i._on_edge = False

    for j in range(pop[0].fitness.shape[0]):
        sort_by_fitness(tosort=pop, obj=j)
        pop[0]._on_edge = True
        pop[-1]._on_edge = True
        pop[0]._dist += (pop[1].fitness[j] - pop[0].fitness[j]) * 2.
        pop[-1]._dist += (pop[-1].fitness[j] - pop[-2].fitness[j]) * 2.
        [pop[i].update_dist(pop[i+1], pop[i-1], j) \
                            for i in range(1, pop.__len__()-1)]
    total_dist = np.sum([i._dist for i in pop])

    return total_dist


def random_crossover(elites, gene_len, pop_size, **kwargs):
    return [np.array([np.random.choice(elites).gene[i] \
                      for i in range(gene_len)]) for _i in range(pop_size)]


def random_nsga_crossover(elites, gene_len, pop_size, **kwargs):
    calc_nsga_acceptance(elites)
    return [np.array([reject_acceptance(elites, gene_len, **kwargs).gene[i] \
                      for i in range(gene_len)]) for _i in range(pop_size)]


def clone_crossover(elites, gene_len, pop_size, **kwargs):
    return [np.random.choice(elites).gene.copy() for _i in range(pop_size)]


def nsga_crossover(elites, gene_len, pop_size, **kwargs):
    calc_nsga_acceptance(elites)
    if len(elites) <= pop_size:
        arch = elites.copy()
        arch.extend(np.random.choice(elites, pop_size-len(arch),
                                     [e.acceptance for e in elites]))
    else:
        arch = [e for e in elites if e._on_edge]
        elites.sort(key=attrgetter('acceptance'), reverse=True)

        arch.extend(np.random.choice(elites, pop_size-len(arch),
                                     [e.acceptance for e in elites]))

    return [[np.random.choice(arch).gene[g] for g in range(gene_len)] \
            for i in range(pop_size)]


def multiroutine_crossover(routines=None, ns=None, params=None,
                           _pop_size=None, **kwargs):
    if hasattr(routines, '__iter__') and hasattr(ns, '__iter__'):
        pass
    elif hasattr(routines, '__iter__') and not hasattr(ns, '__iter__'):
        ns = [1.] * len(routines)
    else:
        raise ValueError("Multi-routine crossover is mal-configured...")

    _fun = partial(_multiroutine_crossover, routines=routines, ns=ns,
                   params=params, _pop_size=_pop_size, **kwargs)
    _fun.__name__ = "multiroutine_crossover"
    return _fun


def _multiroutine_crossover(routines, ns, params, _pop_size, **kwargs):
    new_pop = []
    params.update(kwargs)
    ns = np.rint(np.array(ns)*_pop_size/np.sum(ns)).astype(int)
    if ns.sum() != _pop_size:
        raise ValueError("The sum of the multiroutine crossover "
                         "parameter ns %i does not match pop size %i" \
                         % (ns.sum(), _pop_size))

    for r, n in zip(routines, ns):
        new_pop += r(pop_size=n, **params)

    return new_pop


def gaussian_mutator(gene, bounds, doomed, u=0., st=0.2, **kwargs):

    for i in range(gene.shape[0]):

        if not doomed[i]:
            continue
        else:
            b = bounds[i]
            gene[i] += np.random.normal(u, np.multiply(b[1]-b[0], st))

        if gene[i] < b[0]: gene[i] = b[0]
        if gene[i] > b[1]: gene[i] = b[1]

    return gene


def sort_by_fitness(tosort, obj, reverse=False):
    tosort.sort(key=methodcaller('get_fitness', obj=obj), reverse=reverse)
    return tosort


def valid_moead_popsize(size, n_objs):

    # find the largest H resulting in a population smaller or equal to NP
    if n_objs == 2:
        H = size - 1
    elif n_objs == 3:
        H = int( 0.5 * (np.sqrt(8. * size + 1.) - 3.) )
    else:
        H = 1
        while binom(H + n_objs -1, n_objs - 1) <= size:
            H += 1
        H -= 1

    _size = binom(H + n_objs -1, n_objs - 1)

    if _size + np.finfo(float).tiny < size:
        _size = binom(H + n_objs, n_objs - 1)
        m = "The population size is not suitable for MOEAD's grid weights " \
            "generation. Instead, use %i as pop size in 'evolve_surrogate'."\
            % (_size)

        size = _size
        warnings.warn(m)

    return int(size)


def construct_moead_problem_with_surrogate(problem, surrogate):
    """ Construct a PyGMO problem with surrogate as its fitness function
    """
    new_prob = cp.deepcopy(problem)
    new_prob.fitness = surrogate.render_fitness

    return new_prob


def calc_hypervol(ref=[], front=[], minimize=True, **kwargs):
    """Calculate hypervolume metrics of a Pareto set, given a reference point.

    Parameters
    ----------
    ref : {1D-array-like}, shape (n_objectives, )
          The reference point.

    front : {array-like, sparse matrix}, shape (n_optimals, n_objectives)
            The Pareto optimals on which to calculate hypervolume.

    Returns
    -------
    hypevol : scalar, the calculated hypervolume between the reference point
              and the Pareto optimals.
    """

    # return distance if single objective
    if front.shape[1] == 1:
        return np.subtract(front.ravel() - ref).sum()
    elif front.shape[1] == 2:
        _fs = front[ front[:, 0].argsort()[::-1]]
        return _calc_hypervol(ref=ref, front=_fs, minimize=minimize, **kwargs)
    else:
        pass

    return hypervolume(front, ref)


def _calc_hypervol(ref=[], front=[], minimize=True, **kwargs):

    hypevol = np.insert(front[:-1, 0], 0, ref[0]) - front[:, 0]

    for i in range(1, front.shape[1]):

        if minimize:
            hypevol *= (ref[i] - front[:, i])
        else:
            hypevol = -hypevol
            hypevol *= (front[:, i] - ref[i])

    return hypevol.sum()


def remove_duplication(pop, **kwargs):
    ind_dup = []
    for i in range(len(pop)):
        if i in ind_dup: continue
        for j in range(len(pop)):
            if (pop[i] == pop[j] and i != j): ind_dup.append(j)

    return [pop[i] for i in range(len(pop)) if i in ind_dup]


def bounds_scale(X, bounds, scale=(-1., 1.)):
    """ Scale the features X with range in bounds into range in scale
    """

    if X.ndim == 1: X = X.reshape(1, -1)

    _X = (X - bounds[:,0]) * (scale[1] - scale[0])
    _X = _X / (bounds[:,1] - bounds[:,0]) + scale[0]

    if X.ndim == 1: return _X.ravel()

    return _X


class Cache(object):

    def __init__(self):
        self.cache = {}
        return None

    def save(self, to_save):
        if hasattr(to_save, '__iter__'):
            [self._save(i) for i in to_save]
        else:
            self._save(to_save)
        return None

    def _save(self, item):
        path = item.gene
        _dict = item.to_dict()
        _inter = self.cache
        for node in path:
            if _inter.__class__ is dict:
                if node in _inter.keys():
                    _inter = _inter[node]
                    _dict = _dict[node]
                else:
                    _inter.update(cp.deepcopy(_dict))
                    return None
            else:
                _inter = np.array(_dict)
        return None

    def find(self, item, overwrite=True):
        path = item.gene
        _inter = self.cache
        for node in path:
            if _inter.__class__ is dict:
                if node in _inter.keys():
                    _inter = _inter[node]
                else:
                    return None

        _fitness = np.array(_inter)
        if overwrite: item.fitness = _fitness

        return _fitness

    def __repr__(self):
        return str(self.cache)


class UnconstrainedProblem(object):
    def __init__(self, dim, n_objs, fun, bounds=None):
        self.dim = dim
        self.n_objs = n_objs
        if fun is not None: self.fun = fun
        self.n_evals = 0
        self.bounds = np.array(bounds)

    def obj_fun(self, X):
        self.n_evals += 1
        return self.fun(X)

    def fitness(self, x):
        """ Alias for self.fun
        """
        return self.obj_fun(x)

    def get_bounds(self):
        if self.bounds is None:
            raise ValueError("Problem bounds undefined...")
        return self.bounds[:, 0], self.bounds[:, 1]

    def get_nobj(self):
        return self.n_objs


class IdentityScaler(BaseEstimator, TransformerMixin):

    def __init__(self, copy=False):
        self.copy = copy
        return None

    def fit(self, X, y=None):
        return None

    def fit_transform(self, X, y=None):
        if self.copy: return X.copy()
        return X

    def transform(self, X, y='deprecated', copy=None):
        if self.copy: return X.copy()
        return X

    def partial_fit(self, X, y=None):
        return None

    def inverse_transform(self, X, copy=False):
        if self.copy or copy: return X.copy()
        return X


class BoundsScaler(BaseEstimator, TransformerMixin):

    def __init__(self, bounds, scale_to=(-1., 1.), copy=True):
        self.copy = copy
        self.bounds = bounds
        self.scale_to = scale_to

        if self.scale_to is None:
            raise ValueError("Missing parameter 'scale-to' in BoundsScaler")

        return None

    def fit(self, X, y=None):
        return None

    def fit_transform(self, X, y=None):
        if self.copy: return bounds_scale(X, bounds=self.bounds,
                                          scale=self.to_scale)
        X[:] = bounds_scale(X, bounds=self.bounds, scale=self.to_scale)[:]
        return X

    def transform(self, X, y='deprecated', copy=None):
        if self.copy: return bounds_scale(X, bounds=self.bounds,
                                          scale=self.to_scale)
        X[:, :] = bounds_scale(X, bounds=self.bounds,
                               scale=self.to_scale)[:, :]
        return X

    def partial_fit(self, X, y=None):
        return None

    def inverse_transform(self, X, copy=False):
        if X.ndim == 1: _X = X.reshape(1, -1)

        _X = (_X - self.scale_to[0]) / (self.scale_to[1] - self.scale_to[0])
        _X = _X * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:, 0]

        if X.ndim == 1: _X = _X.ravel()

        if self.copy or copy: return _X

        X[:] = _X[:]

        return X


class FactorScaler():

    def __init__(self, factor, copy=True):
        self.copy = copy
        self.factor = factor

        if self.factor is not None and abs(self.factor) > 0.:
            pass
        else:
            raise ValueError("factor must be a non-zero numerical value")

        return None

    def fit(self, X, y=None):
        return None

    def fit_transform(self, X, y=None):
        if self.copy: return (X * self.factor)
        X[:] = X[:] * self.factor
        return X

    def transform(self, X, y='deprecated', copy=None):
        if self.copy: return (X * self.factor)
        X[:] = X[:] * self.factor
        return X

    def partial_fit(self, X, y=None):
        return None

    def inverse_transform(self, X, copy=False):
        if self.copy: return (X / self.factor)
        X[:] = X[:] / self.factor
        return X


class NoFitnessFunError(RuntimeError):
    """Exception raised when no fitness function given.

        Attributes:
            message -- explanation of the error
    """

    def __init__(self, message="Aborted, no fitness function given..."):
        super().__init__(message)
        return None


class NoSelectionFunError(RuntimeError):
    """Exception raised if no explicit selection routine given (for dev).

        Attributes:
            message -- explanation of the error
    """

    def __init__(self, message="Aborted, no selection routine given..."):
        super().__init__(message)
        return None
