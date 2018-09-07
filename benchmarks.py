#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 14:33:48 2018

@author: niko
"""

import numpy as np
import pandas as pd
from _utils import UnconstrainedProblem
import sys


def load_theo(path, n_objs=2):
    names = ['f'+str(i+1) for i in range(n_objs)]
    return pd.read_csv(filepath_or_buffer=path,
                       names=names,
                       delim_whitespace=True)


class Kursawe(UnconstrainedProblem):
    def __init__(self):
        super(Kursawe, self).__init__(dim=3, n_objs=2, fun=kursawe)
        self.bounds = np.repeat([[-5., 5.]], self.dim, axis=0)
        return None


class ZDT_SOLUTION_MIXIN(object):

    def solutions(self, resolution=1000):
        xs = np.zeros((resolution+1, self.dim))
        xs[:, 0] = np.linspace(0., 1., resolution+1)

        return np.apply_along_axis(self.fun, 1, xs)


class ZDT1(UnconstrainedProblem, ZDT_SOLUTION_MIXIN):
    def __init__(self):
        super(ZDT1, self).__init__(dim=30, n_objs=2, fun=zdt1)
        self.bounds = np.repeat([[0., 1.]], self.dim, axis=0)
        return None


class ZDT2(UnconstrainedProblem, ZDT_SOLUTION_MIXIN):
    def __init__(self):
        super(ZDT2, self).__init__(dim=30, n_objs=2, fun=zdt2)
        self.bounds = np.repeat([[0., 1.]], self.dim, axis=0)
        return None


class ZDT3(UnconstrainedProblem):
    def __init__(self):
        super(ZDT3, self).__init__(dim=30, n_objs=2, fun=zdt3)
        self.bounds = np.repeat([[0., 1.]], self.dim, axis=0)
        return None

    def solutions(self, resolution=1000):
        x1 = np.linspace(0., 1., resolution+1)
        xs = np.zeros((resolution+1, self.dim))
        xs[:, 0] = x1

        res = np.apply_along_axis(self.fun, 1, xs)

        to_save = []
        for i, row in enumerate(res):
            dominated = False
            for j, _row in enumerate(res):
                if i != j and np.greater_equal(row, _row).all():
                    dominated = True
                    break
                else:
                    continue
            if not dominated: to_save.append(i)

        return res[to_save]


class ZDT4(UnconstrainedProblem, ZDT_SOLUTION_MIXIN):
    def __init__(self):
        super(ZDT4, self).__init__(dim=10, n_objs=2, fun=zdt4)
        self.bounds = np.repeat([[0., 1.], [-5., 5.]], [1, self.dim-1], axis=0)
        return None


class ZDT6(UnconstrainedProblem, ZDT_SOLUTION_MIXIN):
    def __init__(self, dim=10):
        super(ZDT6, self).__init__(dim=dim, n_objs=2, fun=zdt6)
        self.bounds = np.repeat([[0., 1.]], self.dim, axis=0)
        return None


class LZ1(UnconstrainedProblem):
    def __init__(self, dim=3):
        if dim < 3:
            raise ValueError("Dimension of problem LZ1 must be greater than 2")
        super(LZ1, self).__init__(dim=dim, n_objs=2, fun=lz1)
        self.bounds = np.repeat([[0., 1.]], self.dim, axis=0)
        return None

    def solutions(self, resolution=1000):
        """ Return analytical solutions of the LZ function No.1
        """
        x1s = np.linspace(0., 1., resolution+1).reshape(-1, 1)
        js = np.arange(1, self.dim, dtype=int) + 1
        xs = np.insert(_solution_lz1(x1s, self.dim, js), [0], x1s, axis=1)

        return np.apply_along_axis(self.fun, 1, xs)


class LZ2(UnconstrainedProblem):
    def __init__(self, dim=3):
        if dim < 3:
            raise ValueError("Dimension of problem LZ2 must be greater than 2")
        super(LZ2, self).__init__(dim=dim, n_objs=2, fun=lz2)
        self.bounds = np.repeat([[0., 1.], [-1., 1.]], [1, self.dim-1], axis=0)
        return None

    def solutions(self, resolution=1000):
        """ Return analytical solutions of the LZ function No.2
        """
        x1s = np.linspace(0., 1., resolution+1).reshape(-1, 1)
        js = np.arange(1, self.dim, dtype=int) + 1
        xs = np.insert(_solution_lz2(x1s, self.dim, js), [0], x1s, axis=1)

        return np.apply_along_axis(self.fun, 1, xs)


class LZ3(UnconstrainedProblem):
    def __init__(self, dim=3):
        if dim < 3:
            raise ValueError("Dimension of problem LZ3 must be greater than 2")
        super(LZ3, self).__init__(dim=dim, n_objs=2, fun=lz3)
        self.bounds = np.repeat([[0., 1.], [-1., 1.]], [1, self.dim-1], axis=0)
        return None

    def solutions(self, resolution=1000):
        """ Return analytical solutions of the LZ function No.3
        """
        x1s = np.linspace(0., 1., resolution+1).reshape(-1, 1)
        js = np.arange(self.dim, dtype=int) + 1
        j1s = js[2::2]
        j2s = js[1::2]

        x_odd = np.insert(_solution_lz3_odd(x1s, self.dim, j1s), [0],
                          x1s, axis=1)
        x_even = _solution_lz3_even(x1s, self.dim, j2s)

        xs = np.insert(x_odd, np.arange(len(j2s), dtype=int)+1, x_even, axis=1)

        return np.apply_along_axis(self.fun, 1, xs)


class LZ4(UnconstrainedProblem):
    def __init__(self, dim=3):
        if dim < 3:
            raise ValueError("Dimension of problem LZ4 must be greater than 2")
        super(LZ4, self).__init__(dim=dim, n_objs=2, fun=lz4)
        self.bounds = np.repeat([[0., 1.], [-1., 1.]], [1, self.dim-1], axis=0)
        return None

    def solutions(self, resolution=1000):
        """ Return analytical solutions of the LZ function No.4
        """
        x1s = np.linspace(0., 1., resolution+1).reshape(-1, 1)
        js = np.arange(self.dim, dtype=int) + 1
        j1s = js[2::2]
        j2s = js[1::2]

        x_odd = np.insert(_solution_lz4_odd(x1s, self.dim, j1s), [0],
                          x1s, axis=1)
        x_even = _solution_lz4_even(x1s, self.dim, j2s)

        xs = np.insert(x_odd, np.arange(len(j2s), dtype=int)+1, x_even, axis=1)

        return np.apply_along_axis(self.fun, 1, xs)


class LZ5(UnconstrainedProblem):
    def __init__(self, dim=3):
        if dim < 3:
            raise ValueError("Dimension of problem LZ5 must be greater than 2")
        super(LZ5, self).__init__(dim=dim, n_objs=2, fun=lz5)
        self.bounds = np.repeat([[0., 1.], [-1., 1.]], [1, self.dim-1], axis=0)
        return None

    def solutions(self, resolution=1000):
        """ Return analytical solutions of the LZ function No.5
        """
        x1s = np.linspace(0., 1., resolution+1).reshape(-1, 1)
        js = np.arange(self.dim, dtype=int) + 1
        j1s = js[2::2]
        j2s = js[1::2]

        x_odd = np.insert(_solution_lz5_odd(x1s, self.dim, j1s), [0],
                          x1s, axis=1)
        x_even = _solution_lz5_even(x1s, self.dim, j2s)

        xs = np.insert(x_odd, np.arange(len(j2s), dtype=int)+1, x_even, axis=1)

        return np.apply_along_axis(self.fun, 1, xs)


class LZ6(UnconstrainedProblem):
    def __init__(self, dim=3):
        if dim < 3:
            raise ValueError("Dimension of problem LZ6 must be greater than 2")
        super(LZ6, self).__init__(dim=dim, n_objs=3, fun=lz6)
        self.bounds = np.repeat([[0., 1.], [-2., 2.]], [2, self.dim-2], axis=0)
        return None

    def solutions(self, resolution=50):
        """ Return analytical solutions of the LZ function No.5
        """
        _r = resolution + 1
        x1s = np.linspace(0., 1., _r)
        x2s = np.linspace(0., 1., _r)
        js = np.arange(2, self.dim, dtype=int) + 1

        x1x2_mesh = np.meshgrid(x1s, x2s)

        x1x2s = np.ravel(x1x2_mesh, order='F').reshape((_r**2, 2))

        x3_to_ns = _solution_lz6(x1x2s[:, 0].reshape(-1, 1),
                                 x1x2s[:, 1].reshape(-1, 1), self.dim, js)

        xs = np.insert(x3_to_ns, [0, 0], x1x2s, axis=1)

        return np.apply_along_axis(self.fun, 1, xs)


class LZ7(UnconstrainedProblem):
    def __init__(self, dim=3):
        if dim < 3:
            raise ValueError("Dimension of problem LZ7 must be greater than 2")
        super(LZ7, self).__init__(dim=dim, n_objs=2, fun=lz7)
        self.bounds = np.repeat([[0., 1.]], self.dim, axis=0)
        return None

    def solutions(self, resolution=1000):
        """ Return analytical solutions of the LZ function No.7
        """
        x1s = np.linspace(0., 1., resolution+1).reshape(-1, 1)
        js = np.arange(1, self.dim, dtype=int) + 1
        xs = np.insert(_solution_lz1(x1s, self.dim, js), [0], x1s, axis=1)

        return np.apply_along_axis(self.fun, 1, xs)


class LZ8(UnconstrainedProblem):
    def __init__(self, dim=3):
        if dim < 3:
            raise ValueError("Dimension of problem LZ8 must be greater than 2")
        super(LZ8, self).__init__(dim=dim, n_objs=2, fun=lz8)
        self.bounds = np.repeat([[0., 1.]], self.dim, axis=0)
        return None

    def solutions(self, resolution=1000):
        """ Return analytical solutions of the LZ function No.8
        """
        x1s = np.linspace(0., 1., resolution+1).reshape(-1, 1)
        js = np.arange(1, self.dim, dtype=int) + 1
        xs = np.insert(_solution_lz1(x1s, self.dim, js), [0], x1s, axis=1)

        return np.apply_along_axis(self.fun, 1, xs)


class LZ9(UnconstrainedProblem):
    def __init__(self, dim=3):
        if dim < 3:
            raise ValueError("Dimension of problem LZ9 must be greater than 2")
        super(LZ9, self).__init__(dim=dim, n_objs=2, fun=lz9)
        self.bounds = np.repeat([[0., 1.], [-1., 1.]], [1, self.dim-1], axis=0)
        return None

    def solutions(self, resolution=1000):
        """ Return analytical solutions of the LZ function No.9
        """
        x1s = np.linspace(0., 1., resolution+1).reshape(-1, 1)
        js = np.arange(1, self.dim, dtype=int) + 1
        xs = np.insert(_solution_lz2(x1s, self.dim, js), [0], x1s, axis=1)

        return np.apply_along_axis(self.fun, 1, xs)


class Rastrigin(UnconstrainedProblem):
    def __init__(self, dim=3):
        super(Rastrigin, self).__init__(dim=dim, n_objs=1, fun=rastrigin)
        self.bounds = np.repeat([[-5.12, 5.12]], self.dim, axis=0)
        return None

    def solutions(self, *args, **kwargs):
        return 0.

    def optimal_x(self):
        return np.repeat([0.], self.dim)


class Ackley(UnconstrainedProblem):
    def __init__(self):
        super(Ackley, self).__init__(dim=2, n_objs=1, fun=ackley)
        self.bounds = np.repeat([[-5., 5.]], self.dim, axis=0)
        return None

    def solutions(self, *args, **kwargs):
        return 0.

    def optimal_x(self):
        return np.repeat([0.], self.dim)


class Levi13(UnconstrainedProblem):
    def __init__(self):
        super(Levi13, self).__init__(dim=2, n_objs=1, fun=levi13)
        self.bounds = np.repeat([[-10., 10.]], self.dim, axis=0)
        return None

    def solutions(self, *args, **kwargs):
        return 0.

    def optimal_x(self):
        return np.repeat([1.], self.dim)


class Shakespears(UnconstrainedProblem):
    __moto__ = "To be or not to be"

    def __init__(self, moto=None):
        if moto.__class__ is str:
            self.__moto__ = moto

        dim = len(self.__moto__)

        super(Shakespears, self).__init__(dim=dim, n_objs=1, fun=None)

        code = np.array([ord(c) for c in self.__moto__], dtype=int)

        self.bounds = np.repeat([[code.min(), code.max()]], self.dim, axis=0)

        self.__code__ = code
        return None

    def fun(self, x):
        if not len(x) == self.dim:
            raise RuntimeError("Length of the str mutated...")

        print(''.join(chr(int(c)) for c in x))
        match = np.equal(x, self.__code__)

        if match.all():
            sys.exit("Congrats! The famous moto found with %i evaluations!" \
                     % self.n_evals)

        return np.array([-(np.count_nonzero(match) / self.dim) ** 2.])

    def solutions(self):
        return self.__moto__


def kursawe(x):
    f1 = np.multiply(-10.0,
                     np.exp(np.multiply(-0.2, np.sqrt( \
                        np.add(x[:-1].__pow__(2), x[1:].__pow__(2)))))).sum()
    f2 = np.add(np.abs(x).__pow__(0.8),
                np.sin(x.__pow__(3)).__mul__(5.0)).sum()

    return np.array([f1, f2])


def zdt1(x):
    f1 = x[0]
    g = np.add(1.0, np.multiply(9./29., x[1:].sum()))
    f2 = np.multiply(g, (1. - np.sqrt(np.divide(f1, g))))

    return np.array([f1, f2])


def zdt2(x):
    f1 = x[0]
    g = np.add(1.0, np.multiply(9./29., x[1:].sum()))
    f2 = np.multiply(g, (1. - np.square(np.divide(f1, g))))

    return np.array([f1, f2])


def zdt3(x):
    f1 = x[0]
    g = np.add(1.0, np.multiply(9./29., x[1:].sum()))
    f2 = np.multiply(g, (1. - np.sqrt(np.divide(f1, g)) - np.multiply(
                     np.divide(f1, g), np.sin(f1*10.*np.pi))))

    return np.array([f1, f2])


def zdt4(x):
    f1 = x[0]
    g = np.add(91., np.subtract(x[1:].__pow__(2.),
                                10. * np.cos(4. * np.pi * x[1:])).sum())
    f2 = np.multiply(g, (1. - np.sqrt(np.divide(f1, g))))

    return np.array([f1, f2])


def zdt6(x):
    f1 = 1. - np.exp(-4. * x[0]) * np.power(np.sin(6.* np.pi * x[0]), 6)
    g = np.add(1.0, np.multiply(9.,
                                np.power(np.divide(x[1:].sum(), 9.), 0.25)))
    f2 = np.multiply(g, (1. - np.square(np.divide(f1, g))))

    return np.array([f1, f2])

def rastrigin(x):
    return 10. * len(x) + np.sum(np.power(x, 2) - 10. * np.cos(2*np.pi*x))

def ackley(x):
    e1 = -0.2 * np.sqrt(0.5 * np.power(x, 2.).sum())
    e2 = 0.5 * np.cos( 2 * np.pi * x ).sum()
    return -20. * np.exp(e1) - np.exp(e2) + np.e + 20.

def levi13(x):
    v0 = np.array([0, 1., 1.])
    v1 = np.append(np.power(np.sin(3 * np.pi * x), 2),
                   np.power(np.sin(2 * np.pi * x[1]), 2))
    v2 = np.concatenate(([1.], np.power(x - 1., 2)))
    return (v1 + v0).__mul__(v2).sum()

def _init_lz(x):
    # elememts with odd indices
    x_odd = x[2::2]
    # elements with even indices
    x_even = x[1::2]

    n = x.shape[-1]

    js = np.arange(n, dtype=int) + 1

    j1s = js[2::2]
    j2s = js[1::2]

    return x_odd, x_even, n, j1s, j2s

def lz1(x):
    x1s, x2s, n, j1s, j2s = _init_lz(x)
    s1 = _solution_lz1(x[0], n, j1s)
    s2 = _solution_lz1(x[0], n, j2s)

    f1 = x[0] + 2./np.linalg.norm(j1s) * \
         np.subtract(x1s, s1).__pow__(2.).sum()
    f2 = 1. - x[0]**.5 + 2./np.linalg.norm(j2s) * \
         np.subtract(x2s, s2).__pow__(2.).sum()

    return np.array([f1, f2])

def _solution_lz1(x1, n, js):
    return np.power(x1, 0.5 + 1.5 * (js - 2.) / (n - 2.))


def lz2(x):
    x1s, x2s, n, j1s, j2s = _init_lz(x)
    s1 = _solution_lz2(x[0], n, j1s)
    s2 = _solution_lz2(x[0], n, j2s)

    f1 = x[0] + 2./np.linalg.norm(j1s, 1) * \
         np.subtract(x1s, s1).__pow__(2.).sum()
    f2 = 1. - x[0]**.5 + 2./np.linalg.norm(j2s, 1) * \
         np.subtract(x2s, s2).__pow__(2.).sum()

    return np.array([f1, f2])

def _solution_lz2(x1, n, js):
    return np.sin( np.pi * (6.*x1 + js/n) )


def lz3(x):
    x1s, x2s, n, j1s, j2s = _init_lz(x)
    s1 = _solution_lz3_odd(x[0], n, j1s)
    s2 = _solution_lz3_even(x[0], n, j2s)

    f1 = x[0] + 2./np.linalg.norm(j1s) * \
         np.subtract(x1s, s1).__pow__(2.).sum()
    f2 = 1. - x[0]**.5 + 2./np.linalg.norm(j2s) * \
         np.subtract(x2s, s2).__pow__(2.).sum()

    return np.array([f1, f2])

def _solution_lz3_odd(x1, n, js):
    return 0.8 * x1 * np.cos( np.pi * (6.*x1 + js/n) )

def _solution_lz3_even(x1, n, js):
    return 0.8 * x1 * np.sin( np.pi * (6.*x1 + js/n) )


def lz4(x):
    x1s, x2s, n, j1s, j2s = _init_lz(x)
    s1 = _solution_lz4_odd(x[0], n, j1s)
    s2 = _solution_lz4_even(x[0], n, j2s)

    f1 = x[0] + 2./np.linalg.norm(j1s) * \
         np.subtract(x1s, s1).__pow__(2.).sum()
    f2 = 1. - x[0]**.5 + 2./np.linalg.norm(j2s) * \
         np.subtract(x2s, s2).__pow__(2.).sum()

    return np.array([f1, f2])

def _solution_lz4_odd(x1, n, js):
    return 0.8 * x1 * np.cos( np.pi * (6.*x1 + js/n)/3.0 )

def _solution_lz4_even(x1, n, js):
    return 0.8 * x1 * np.sin( np.pi * (6. * x1 + js / n) )


def lz5(x):
    x1s, x2s, n, j1s, j2s = _init_lz(x)
    s1 = _solution_lz5_odd(x[0], n, j1s)
    s2 = _solution_lz5_even(x[0], n, j2s)

    f1 = x[0] + 2./np.linalg.norm(j1s) * \
         np.subtract(x1s, s1).__pow__(2.).sum()
    f2 = 1. - x[0]**.5 + 2./np.linalg.norm(j2s) * \
         np.subtract(x2s, s2).__pow__(2.).sum()

    return np.array([f1, f2])

def _solution_lz5_odd(x1, n, js):
    factr = 0.3 * x1**2. * np.cos(np.pi * (24.*x1 + 4.*js/n)) + 0.6*x1
    return factr * np.cos( np.pi * (6.*x1 + js/n) )

def _solution_lz5_even(x1, n, js):
    factr = 0.3 * x1**2. * np.cos(np.pi * (24.*x1 + 4.*js/n)) + 0.6*x1
    return factr * np.sin( np.pi * (6.*x1 + js/n) )


def lz6(x):
    # elememts with indices that i-1 is a multiplication of 3
    x1s = x[3::3]
    # elememts with indices that i-2 is a multiplication of 3
    x2s = x[4::3]
    # elememts with indices that i is a multiplication of 3
    x3s = x[2::3]

    n = x.shape[-1]

    indices = np.arange(n, dtype=int) + 1
    j1s = indices[3::3]
    j2s = indices[4::3]
    j3s = indices[2::3]

    s1 = _solution_lz6(x[0], x[1], n, j1s)
    s2 = _solution_lz6(x[0], x[1], n, j2s)
    s3 = _solution_lz6(x[0], x[1], n, j3s)

    f1 = np.cos(0.5*x[:2]*np.pi).prod() + 2./np.linalg.norm(j1s) * \
         np.subtract(x1s, s1).__pow__(2.).sum()
    f2 = np.cos(0.5*x[0]*np.pi) * np.sin(0.5*x[1]*np.pi) + \
         2./np.linalg.norm(j2s) * np.subtract(x2s, s2).__pow__(2.).sum()
    f3 = np.sin(0.5*x[0]*np.pi) + 2./np.linalg.norm(j3s) * \
         np.subtract(x3s, s3).__pow__(2.).sum()

    return np.array([f1, f2, f3])

def _solution_lz6(x1, x2, n, js):
    return 2 * x2 * np.sin( np.pi * (2.*x1 + js/n) )


def lz7(x):
    x1s, x2s, n, j1s, j2s = _init_lz(x)
    y1 = x1s - _solution_lz1(x[0], n, j1s)
    y2 = x2s - _solution_lz1(x[0], n, j2s)

    t1 = 4.*np.power(y1, 2.) - np.cos(8.*np.pi*y1) + 1.
    t2 = 4.*np.power(y2, 2.) - np.cos(8.*np.pi*y2) + 1.

    f1 = x[0] + 2./np.linalg.norm(j1s) * t1.sum()
    f2 = 1. - x[0]**.5 + 2./np.linalg.norm(j2s) * t2.sum()

    return np.array([f1, f2])


def lz8(x):
    x1s, x2s, n, j1s, j2s = _init_lz(x)
    y1 = x1s - _solution_lz1(x[0], n, j1s)
    y2 = x2s - _solution_lz1(x[0], n, j2s)

    t1 = 4.*np.power(y1, 2.).sum() - 2. * np.cos(20.*np.pi*y1 / \
                                                 np.sqrt(j1s)).prod() + 2.
    t2 = 4.*np.power(y2, 2.).sum() - 2. * np.cos(20.*np.pi*y2 / \
                                                 np.sqrt(j2s)).prod() + 2.

    f1 = x[0] + 2./np.linalg.norm(j1s) * t1
    f2 = 1. - x[0]**.5 + 2./np.linalg.norm(j2s) * t2

    return np.array([f1, f2])


def lz9(x):
    x1s, x2s, n, j1s, j2s = _init_lz(x)
    s1 = _solution_lz2(x[0], n, j1s)
    s2 = _solution_lz2(x[0], n, j2s)

    f1 = x[0] + 2./np.linalg.norm(j1s) * \
         np.subtract(x1s, s1).__pow__(2.).sum()
    f2 = 1. - x[0]**2. + 2./np.linalg.norm(j2s) * \
         np.subtract(x2s, s2).__pow__(2.).sum()

    return np.array([f1, f2])
