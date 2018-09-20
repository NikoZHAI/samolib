#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 21 09:42:07 2018

@author: niko
"""
from abc import abstractmethod
from multiprocessing import Pool
# import traceback, functools

import numpy as np
from multilayer_perceptron import MLPSurrogate
from ea import Population
from _utils import gaussian_mutator
from sklearn.svm import NuSVR, SVR
from sklearn.preprocessing import StandardScaler
#from sklearn.tree import DecisionTreeRegressor
from gpr import GaussianProcessRegressor
from pySOT.rbf import RBFInterpolant
import matplotlib.pyplot as plt


def nse(y, y_pred):
    '''
    ===
    NSE
    ===

    Coefficient of determination R²

    Parameters
    ----------
    y : array_like [n]
    Targets
    y_pred : array_like [n]
    Predictions

    Returns
    -------
    f : float
    NSE value
    '''
    m = np.nanmean(y)
    a = np.square(np.subtract(y, y_pred))
    b = np.square(np.subtract(y, m))
    if a.any() < 0.0:
        return(np.nan)
    f = 1.0 - (np.nansum(a)/np.nansum(b))
    return f


def heuristic_optimizer(obj_fun, init, bounds, size=64, max_generation=250):
    # GA hyperparameters
    dim = bounds.__len__()
    n_objs = 1

    pop = Population(dim=dim, size=size, n_objs=n_objs, fitness_fun=obj_fun,
                     mutation_fun=gaussian_mutator, minimize=True,
                     max_generation=max_generation, bounds=bounds)
    pop.evolve(u=0., st=0.2)
    return pop.elites[0].gene ,pop.elites[0].fitness


def gen_param_sets(n_models, params={}):
    """ Generate a list of parameter sets, one set per model
        Parameters
        ----------
        n_models: int
            Number of models

        params: dict
            Parameter sets. Each element could be a list of paramters, each
            parameter in the list is assigned to the corresponding model.
            Element can also be a single value. Models will share the same
            parameter in this case. A mix of the two above methods is allowed.
    """
    param_sets = [{}] * n_models

    for i in range(n_models):
        for k, e in params.items():
            if type(e) is list:
                if len(e) == n_models:
                    param_sets[i].update({k: e[i]})
                elif len(e) == 1:
                    param_sets[i].update({k: e[0]})
                else:
                    raise ValueError("The list of parameters must have the "
                                     "same length as the number of models, or "
                                     "all models use the same parameter if "
                                     "only one parameter passed..."
                                     "n_models = %i, n_params (%s): %i"
                                     % (n_models, len(e)))
            else:
                param_sets[i].update({k: e})
    return param_sets


#def trace_unhandled_exceptions(func):
#    @functools.wraps(func)
#    def wrapped_func(*args, **kwargs):
#        try:
#            return func(*args, **kwargs)
#        except:
#            print('Exception in ' + func.__name__)
#            traceback.print_exc()
#    return wrapped_func


# @trace_unhandled_exceptions
def fit_parallel(model, X, y):
    model.fit(X, y)
    return model


class BaseSurrogate(object):

    @abstractmethod
    def __init__(self, n_objs=2, n_models=None, params={}, n_process=1,
                 X_scaler=None, y_scaler=None, warm_start=False, **kwargs):
        self.n_objs = n_objs
        self.n_models = n_objs if n_models is None else n_models
        self.param_sets = gen_param_sets(n_models=self.n_models, params=params)
        self.n_process = n_process
        self._warm_start = warm_start
        self.X_scaler = X_scaler
        self.scale_X = True if X_scaler is not None else False
        self.y_scaler = y_scaler
        self.scale_y = True if y_scaler is not None else False

        return None

    def benchmark(self, problem=None, X=None, y=None, obj_fun=None, dim=None,
                  bounds=None, n_samples=100, random=None):

        # Init random module, Features X and targets y
        if random is None: random = np.random
        if problem is not None:
            bounds = problem.bounds
            dim = problem.dim
            obj_fun = problem.fun
        if bounds is not None: bounds = bounds

        if X is None:
            X = self._gen_benchmark_samples(dim, bounds, n_samples, random)
        if y is None:
            if obj_fun is None:
                raise ValueError("y or obj_fun must be given")
            else:
                y = np.array([obj_fun(x) for x in X])

        y_pred = self.predict(X)
        self._plot_benchmark(y, y_pred)
        return [nse(y[:,i], y_pred[:,i]) for i, m in enumerate(self.models)]

    def _plot_benchmark(self, y, y_pred):
        n_objs = y.shape[1]
        fig, axs = plt.subplots(nrows=n_objs, ncols=2)
        for i, ax_ in enumerate(axs):
            ax_[0].plot(y[:,i], label='test')
            ax_[0].plot(y_pred[:,i], label='pred')
            ax_[1].plot(y_pred[:,i] - y[:,i], c='orangered', linestyle='-',
                        linewidth=0.8)

            ax_[0].set_title('Objective %i' % (i+1), size='small')
            ax_[1].set_title('Error F%i' % (i+1), size='small')

            ax_[0].grid(True, ls=':')
            ax_[1].grid(True, ls=':')
        fig.tight_layout()
        plt.show()
        return None

    def _gen_benchmark_samples(self, dim, bounds, n_samples, random):

        return random.uniform(low=bounds[:,0], high=bounds[:,1],
                              size=(n_samples, dim))

    def fit(self, X, y):
        """ Train the surrogate with features X and targets y
        """
        if self._warm_start:
            if self.scale_X:
                self.X_scaler.partial_fit(X)
                X = self.X_scaler.transform(X)
            if self.scale_y:
                self.y_scaler.partial_fit(X)
                y = self.y_scaler.transform(X)
        else:
            if self.scale_X: X = self.X_scaler.fit_transform(X)
            if self.scale_y: y = self.y_scaler.fit_transform(y)

        return self._fit(X, y)


    def _fit(self, X, y):
        """ Train the surrogate with features X and targets y
        """
        if self.n_process <= 1:
            [m.fit(X, y[:, i]) for i, m in enumerate(self.models)]
        else:
            if self.n_process > self.n_models: self.n_process = self.n_models
            tasks = [(m, X, y[:, i]) for i, m in enumerate(self.models)]
            self._fit_parallel(tasks=tasks)
        return None

    def _fit_parallel(self, tasks):
        with Pool(self.n_process) as p:
            # Now we use apply() instead of apply_async() for thread safety in np.random
            self.models = [p.apply(fit_parallel, t) for t in tasks]
        return None

    def predict(self, X):
        """ Make predictions given features X
        """
        if self.scale_X: X = self.X_scaler.transform(X)
        if self.scale_y:
            return self.y_scaler.inverse_transform(self._predict(X))

        return self._predict(X)


    def _predict(self, X):
        """ Make predictions given features X
        """
        ys = [m.predict(X) for m in self.models]
        return np.column_stack(ys)

    def render_fitness(self, X):
        if X.ndim == 1:
            return self.predict(X.reshape(1, -1)).ravel()
        return self.predict(X)


class MixedFakeSurrogate(object):

    def __init__(self, problem, surrogate, objs_to_surrogate=[]):
        """ A fake surrogate to delegate a cheap objective function
        """
        if hasattr(problem, 'construct_fake_surrogate'):
            self.func = problem.construct_fake_surrogate(surrogate)
        else:
            raise NotImplementedError("User-defined problems must have a "
                                      "'construct_fake_surrogate' method to "
                                      "construct fake surrogates")
        self.surrogate = surrogate
        self.objs_to_surrogate = objs_to_surrogate
        self._warm_start = surrogate._warm_start
        return None

    def fit(self, X, y):
        if len(self.objs_to_surrogate):
            self.surrogate.fit(X, y[:, self.objs_to_surrogate])
        return None

    def predict(self, X):
        if X.ndim == 1:
            return self.func(X).T
        else:
            return np.apply_along_axis(self.func, 1, X)

    def render_fitness(self, X):
        if X.ndim == 1:
            return self.func(X)
        else:
            return self.predict(X)

    def benchmark(self, X=None, y=None, obj_fun=None, dim=None, bounds=None,
                  n_samples=100, random=None):

        # Init random module, Features X and targets y
        if random is None: random = np.random
        if bounds is None: bounds = self.bounds
        if X is None:
            X = self._gen_benchmark_samples(dim, bounds, n_samples, random)
        if y is None:
            if obj_fun is None:
                raise ValueError("y or obj_fun must be given")
            else:
                y = np.array([obj_fun(x) for x in X])

        y_pred = self.surrogate.predict(X)
        y_partial = np.atleast_2d(y[:, self.objs_to_surrogate])
        self._plot_benchmark(y_partial, y_pred)
        return [nse(y_partial[:,i], y_pred[:,i]) for i in \
                range(len(self.objs_to_surrogate))]

    def _plot_benchmark(self, y, y_pred):
        n_objs = y.shape[1]
        fig, axs = plt.subplots(nrows=n_objs, ncols=2)
        for i, ax_ in enumerate(axs):
            ax_[0].plot(y[:,i], label='test')
            ax_[0].plot(y_pred[:,i], label='pred')
            ax_[1].plot(y_pred[:,i] - y[:,i], c='orangered', linestyle='-',
                        linewidth=0.8)

            ax_[0].set_title('Objective %i' % (i+1), size='small')
            ax_[1].set_title('Error F%i' % (i+1), size='small')

            ax_[0].grid(True, ls=':')
            ax_[1].grid(True, ls=':')
        fig.tight_layout()
        plt.show()
        return None


class NeuralNet(BaseSurrogate):

    def __init__(self, n_objs=2, n_models=None, params={}, n_process=1,
                 X_scaler=None, y_scaler=None, warm_start=False):
        sup = super(NeuralNet, self)
        sup.__init__(n_objs=n_objs, n_models=n_models, params=params,
                     n_process=n_process, X_scaler=X_scaler, y_scaler=y_scaler,
                     warm_start=warm_start)

        self.models = [MLPSurrogate(**p) for p in self.param_sets]
        self._warm_start = np.all([m.warm_start for m in self.models])

        return None


class Kriging(BaseSurrogate):

    def __init__(self, n_objs=2, n_models=None, params={}, n_process=1,
                 X_scaler=None, y_scaler=None, warm_start=False):
        sup = super(Kriging, self)
        sup.__init__(n_objs=n_objs, n_models=n_models, params=params,
                     n_process=n_process, X_scaler=X_scaler, y_scaler=y_scaler,
                     warm_start=warm_start)

        self.models = [GaussianProcessRegressor(**p) for p in self.param_sets]

        return None

class RBFN(BaseSurrogate):

    def __init__(self, n_objs=2, n_models=None, params={}, n_process=1,
                 X_scaler=None, y_scaler=None, warm_start=True):
        sup = super(RBFN, self)
        sup.__init__(n_objs=n_objs, n_models=n_models, params=params,
                     n_process=n_process, X_scaler=X_scaler, y_scaler=y_scaler,
                     warm_start=warm_start)

        self.models = [RBFInterpolant(**p) for p in self.param_sets]
        [m.__setattr__('fit', m.add_point) for m in self.models]

        return None

    def _fit(self, X, y):
        """ Train the RBF network with features X and targets y
        """
        if X.ndim == 1:
            if self.n_process < 2:
                [m.fit(X, y[:, i]) for i, m in enumerate(self.models)]
            else:
                if self.n_process > self.n_models:
                    self.n_process = self.n_models
                tasks = [(m, X, y[:, i]) for i, m in enumerate(self.models)]
                self._fit_parallel(tasks=tasks)
            return None

        if self.n_process < 2:
            [[m.add_point(xx, yy[i]) for i, m in enumerate(self.models)] \
              for xx, yy in zip(X, y)]
        else:
            if self.n_process > self.n_models:
                self.n_process = self.n_models

            for xx, yy in zip(X, y):
                tasks = [(m, xx, yy[i]) for i, m in enumerate(self.models)]
                self._fit_parallel(tasks=tasks)

        return None

    def _predict(self, X):
        """ Make predictions given features X
        """
        ys = [m.evals(X) for m in self.models]
        return np.column_stack(ys)

    def render_fitness(self, X):
        if X.ndim == 1:
            ys = [m.eval(X) for m in self.models]
            return np.array(ys)
        return self.predict(X)


class SVM(BaseSurrogate):

    def __init__(self, n_objs=2, n_models=None, params={}, n_process=1,
                 X_scaler=None, y_scaler=None, warm_start=False):
        sup = super(SVM, self)
        sup.__init__(n_objs=n_objs, n_models=n_models, params=params,
                     n_process=n_process, X_scaler=X_scaler, y_scaler=y_scaler,
                     warm_start=warm_start)

        self.models = [SVR(**p) for p in self.param_sets]

        return None