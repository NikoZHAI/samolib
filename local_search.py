#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 21 09:42:07 2018

@author: niko
"""

import numpy as np
from ea import Population

class LocalSearchMixin(object):
    """ Mixins for local search
    """
    def __init__(self, *args, **kwargs):
        self.local_search_front = []
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

        # Renders self.local_front
        local_search_list = ['gap', 'local', 'trust']

        _s = name.lower()

        if any(_pattern in _s for _pattern in global_list):
            return self.global_pop
        elif any(_pattern in _s for _pattern in front_list):
            return self.front
        elif any(_pattern in _s for _pattern in true_front_list):
            return self.true_front
        elif any(_pattern in _s for _pattern in local_search_list):
            return self.local_search_front
        else:
            raise ValueError('Name of the population %s not found....' % name)

        return None

    def local_search(self, at, radius, size=None,
                     max_generation=None, selection_fun=None,
                     mutation_fun=None, mutation_rate=None,
                     crossover_fun=None, trial_method='lhs',
                     trial_criterion='cm', u=0., st=0.2, **kwargs):
        """ Apply MOEA local search at specific sub-regions
        """
        # Local Search should be used as a Mixin class, following variables
        # caome from objects of other classes
        dim = self.dim
        size = size or self.size
        selection_fun = selection_fun or self.selection_fun
        mutation_fun = mutation_fun or self.mutation_fun
        mutation_rate = mutation_rate or self.mutation_rate
        crossover_fun = crossover_fun or self.crossover_fun
        max_generation = max_generation or self.max_generation

        if hasattr(self, 'surrogate'):
            fitness_fun = self.surrogate.render_fitness
        else:
            fitness_fun = self.fitness_fun

        if hasattr(radius, '__iter__'):
            if not (radius.__len__() is self.gene_len):
                raise ValueError('If "radius" is an iterable object, its '
                                 'length (found %i) must be identical to the '
                                 'problem\'s dimentionality (found %i).'
                                 % (radius.__len__(), self.gene_len))
            elif radius.__class__.__name__ == 'ndarray':
                pass
            else:
                radius = np.array(radius)

        elif type(radius) not in [int, float]:
            raise TypeError('radius must be a digital number or a list of '
                            'digital numbers')
        else:
            pass

        if at.__class__.__name__ == 'Individual':
            _at = at.gene
        elif hasattr(at, '__iter__') and at.__len__() == self.gene_len:
            _at = at
        else:
            raise ValueError('"at" should either be paased an Individual or an'
                             ' array with the same length as an Individual\'s '
                             'gene.')

        # Generate local search population
        _pop = self._gen_local_search_pop(pioneer=at, r=radius, loc=_at,
                                          dim=dim, size=size,
                                          fitness_fun=fitness_fun,
                                          selection_fun=selection_fun,
                                          mutation_fun=mutation_fun,
                                          mutation_rate=mutation_rate,
                                          crossover_fun=crossover_fun,
                                          max_generation=max_generation)

        # Perform local search
        if hasattr(self, 'embedded_ea'):
            if hasattr(self.embedded_ea, '_external_moea'):
                return self._local_search_with_embedded_ea(pop=_pop, **kwargs)
            else:
                pass
        else:
            pass

        _pop.evolve(u=u, st=st, trial_method=trial_method,
                    trial_criterion=trial_criterion)

        for i in _pop.front:
            i.bounds = self.bounds

        return _pop.front


    def _gen_local_search_pop(self, pioneer, r, loc, dim, size, fitness_fun,
                              selection_fun, mutation_fun, mutation_rate,
                              crossover_fun, max_generation):

        # Generate local search bounds
        _up_bounds = np.add(loc, r)
        _lo_bounds = np.subtract(loc, r)
        bounds = [[_lb, _rb] for _lb, _rb in zip(_lo_bounds, _up_bounds)]

        # Restrict to the outter boundaries
        _i = 0

        for _ in np.greater(bounds, pioneer.bounds):
            if not _[0]: bounds[_i][0] = pioneer.bounds[_i][0]
            if _[1]: bounds[_i][1] = pioneer.bounds[_i][1]
            _i += 1

        # Instantiate a new population
        _copy = Population(dim=dim, size=size-1, n_objs=self.n_objs,
                           fitness_fun=fitness_fun,
                           selection_fun=selection_fun,
                           mutation_fun=mutation_fun,
                           mutation_rate=mutation_rate,
                           crossover_fun=crossover_fun,
                           max_generation=max_generation, bounds=bounds)

        # Add the pioneer into the population's front
        _copy.global_pop.append(pioneer)
        _copy.__setattr__('size', size)
        return _copy


    def _local_search_with_embedded_ea(self, pop, **kwargs):
        """ Local search based on an embedded EA
        """
        size, generation = int(pop.size), int(pop.max_generation)
        local_ea = self.embedded_ea(problem=pop.problem, size=size,
                                    generation=generation, **kwargs)
        local_ea.evolve()
        local_ea.export_internal_pop(pop=pop.global_pop)
        pop.select()

        for i in pop.elites:
            i.bounds = self.bounds

        return pop.elites.copy()
