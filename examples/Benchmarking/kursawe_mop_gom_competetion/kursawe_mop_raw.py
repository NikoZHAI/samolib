#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking MOPRISM-raw, MOPRISM-10 and MOPRISM-20 with Kursawe functions

In all of the tests, local search is switched off and the candidates are
directly sampled from the final generation of the MOEA in each episode.

All tests apply RBFs as surrogate.

@author: niko
"""

import sys, os
import numpy as np
import itertools
sys.path.append('/home/niko/Documents/SPr-GAN/00_Projects/'
                'MOPRISM/77_Playground/GA/samolib/')

from premade import MOPRISM   # Framework
from ea import MOEAD          # MOEA
from surrogates import RBFN   # Surrogate
from _utils import nsga_crossover, gaussian_mutator, random_crossover
from sklearn.preprocessing import StandardScaler
from benchmarks import Kursawe

# RBFN surrogate kernels
from pySOT.kernels import CubicKernel
from pySOT.tails import LinearTail

import matplotlib.pyplot as plt
from matplotlib import rc

# Matplotlib
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def run_sim(n, path, simtitle, PROBLEM, sampling_size, seed):

    POP_SIZE        = 100
    MAX_GENERATION  = 50
    MAX_EPISODE     = 100
    MAX_EVAL        = 1000
    STOPPING_RULE   = 'max_eval'
    MUTATION_RATE   = 0.1
    MUTATION_U      = 0.
    MUTATION_ST     = 0.2
    REF             = [1., 1.]
    MINIMIZE        = True
    VERBOSE         = True
    X_SCALER        = StandardScaler()

    # Set global numpy random_state
    np.random.seed(seed)

    theo = np.genfromtxt('pareto_Kursawe.txt', dtype='f8')

    # Instantiate a population
    pop = MOPRISM(size=POP_SIZE, problem=PROBLEM, max_generation=MAX_GENERATION,
                  max_episode=MAX_EPISODE, reference=REF, minimize=MINIMIZE,
                  stopping_rule=STOPPING_RULE, max_eval=MAX_EVAL,
                  mutation_rate=MUTATION_RATE, revolution=True,
                  embedded_ea=MOEAD, verbose=VERBOSE, no_improvement_step_tol=3,
                  local_search=False)

    pop.selection_fun = pop.compute_front
    pop.mutation_fun = gaussian_mutator
    pop.crossover_fun = random_crossover

    # Parametrization
    params_ea = {'u': MUTATION_U,
                 'st': MUTATION_ST,
                 'trial_method': 'lhs',
                 'trial_criterion': 'cm'}

    kernel = CubicKernel
    tail = LinearTail

    params_surrogate = \
        {'kernel': kernel,
         'tail': tail,
         'maxp': MAX_EVAL + POP_SIZE,
         'eta': 1e-8,
         }

    # ===============================Initialization============================
    pop.config_surrogate(typ='rbf', params=params_surrogate, n_process=1,
                         X_scaler=X_SCALER, warm_start=True)

    sampling_methods = [pop.lazy_sampling]
    sampling_rates = [1.]
    sampling_size = [sampling_size]
    candidates = ['global_pop']
    pop.config_sampling(methods=sampling_methods, sizes=sampling_size,
                        sample_rates=sampling_rates, candidates=candidates)


    pop.run(params_ea=params_ea,
            params_surrogate=params_surrogate,
            theo=theo)

    # ============================= Save Results ================================ #
    # path to save
    directory = path + simtitle + '/' + str(n) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    pop.render_features(pop.true_front).tofile(directory + 'xs.dat')
    pop.render_targets(pop.true_front).tofile(directory + 'fs.dat')
    np.array(pop.hypervol_diff).tofile(directory + 'hv_diff.dat')
    np.array(pop.hypervol_cov).tofile(directory + 'hv_cov.dat')
    np.array(pop.hypervol_index).tofile(directory + 'hv_ind.dat')

    # ================================Visualization============================== #
    # plot_res(pop=pop, ref=theo, directory=directory)

    return pop


def plot_res(pop, ref, directory):

    fig, (ax, ax_metric) = plt.subplots(1, 2, figsize=(10,4), dpi=100)

    # Test surrogate results
    final_arc = pop.true_front
    x = []
    y = []
    for f in final_arc:
        x.append(f.fitness[0])
        y.append(f.fitness[1])

    ax.set_xlabel(r'f_1')
    ax.set_ylabel(r'f_2', labelpad=0)
    ax.set(title="MOPRISM with embedded NSGA-II")

    #ax.set_ylim((0., 2500.))
    #
    ax.scatter(ref[:, 0], ref[:, 1], c='orangered', s=1.5,
               label="Reference")
    ax.scatter(x, y, c='royalblue', s=2.0, label="MOPRISM")

    # Plot legend.
    lgnd = ax.legend(loc='lower left', numpoints=1, fontsize=9)

    # change the marker size manually for both lines
    lgnd.legendHandles[0]._sizes = [10]
    lgnd.legendHandles[1]._sizes = [10]
    ax.grid(True, ls=':')

    # ================================ Metrics ====================================
    ax_hypervol = ax_metric.twinx()

    # Calculate hypervolume coverage
    hypervol_cov = [hv/pop.hypervol_ana for hv in pop.hypervol]

    # Title
    ax_metric.set_title('Hypervolume Metrics')

    # Ranges
    ax_metric.set_xlim((0, pop.episode))
    #ax_metric.set_ylim((0., 1.))
    #ax_hypervol.set_ylim((0., 1.))

    # Lables
    ax_metric.set_ylabel(r'Uncovered \ Hypervolume ($\times 10^4$)',
                         color='royalblue', labelpad=0.4)
    ax_hypervol.set_ylabel('Hypervolume \ Coverage (\%)', color='orangered')
    ax_metric.set_xlabel('Episode (N)')

    # Grid
    ax_metric.grid(True, ls=':')

    # Plot
    metric = ax_metric.plot(pop.hypervol_diff, color='royalblue',
                            label='Uncovered Hypervolume', linewidth=2.0)
    hypervol = ax_hypervol.plot(hypervol_cov,
                                label='Hypervolume Coverage (\%)',
                                color='orangered',
                                linewidth=2.0)

    # Ticks
    ax_metric.set_yticklabels((ax_metric.get_yticks() / 1e4).astype(int),
                              color='royalblue')
    ax_hypervol.set_yticklabels(ax_hypervol.get_yticks().round(1),
                                color='orangered')

    legends = metric + hypervol
    labs = [l.get_label() for l in legends]

    ax_metric.legend(legends, labs, loc='center right', fontsize=9)

    plt.show()
    fig.savefig(directory + 'res.png', dpi=300, format='png')

    return None

def main():

    PROBLEM    = [Kursawe]
    sample_size = [100, 0.2, 0.1]

    path_ = 'results/compare/'
    simtitle = 'kursawe_moead_mop_raw'
    directory = path_ + simtitle + '/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(directory+'seeds.txt'):
        seeds = np.random.randint(0, int(2**32-1), dtype='u4',
                                  size=len(PROBLEM)*len(sample_size))
        np.savetxt(directory+'seeds.txt', seeds, fmt='%i', delimiter=';')
    else:
        seeds = np.genfromtxt(directory + 'seeds.txt', dtype='u4')

    configs = list(itertools.product(sample_size, PROBLEM))

    for i in range(len(PROBLEM)*len(sample_size)):
        size, p = configs[i]
        s = seeds[i]

        pop = run_sim(i, path_, simtitle, p(), size, s)
        del pop
        # pops.append(pop)

    return 0


if __name__ == '__main__':
    status = main()
