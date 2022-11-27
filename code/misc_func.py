'''
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
'''
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from numba import njit

def parallel_run(fun, arg_list):
    p = Pool(processes = cpu_count())
    m = p.map(fun, arg_list)
    p.terminate()
    p.join() 
    return m

def figure():
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    fig.set_size_inches(w=14, h=5)

    ax.spines["top"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    
    ax.grid()
    return fig, ax

def rhoBlochrep(rho_list, timeList, ax = None, component = ['rx', 'ry', 'rz'], line = '-', label = ''):  
    rho_list = rho_list.copy()
    if not ax:
        fig, ax = figure()
    
    dict_sigma = {'rx':np.array([[0,1],[1,0]], dtype = np.complex128), 
                  'ry':np.array([[0,-1j],[1j,0]], dtype = np.complex128), 
                  'rz':np.array([[1,0],[0,-1]], dtype = np.complex128)}
    dict_color = {'rx':'b', 'ry':'g', 'rz':'r'}
    
    t_steps = len(timeList)
    t = np.linspace(timeList[0], timeList[-1], t_steps)
    for r_i in component:
        r_ = []
        for i in range(t_steps):
            r_.append(np.real(np.trace(rho_list[i].dot(dict_sigma[r_i]))))
        ax.plot(t, r_, line, label=f'{r_i} {label}', color=dict_color[r_i])

    ax.set_xlabel('time')
    ax.legend()

@njit
def rhoBlochrep_data(rho):
    pauli_components = np.zeros((3, np.shape(rho)[0]), dtype=np.float64)
    for n_it, rho_it in enumerate(rho):
        rho_it = np.ascontiguousarray(rho_it)
        pauli_components[0][n_it] += np.real(np.trace(rho_it.dot(np.ascontiguousarray(np.array([[0,1],[1,0]], dtype = np.complex128)))))
        pauli_components[1][n_it] += np.real(np.trace(rho_it.dot(np.ascontiguousarray(np.array([[0,-1j],[1j,0]], dtype = np.complex128)))))
        pauli_components[2][n_it] += np.real(np.trace(rho_it.dot(np.ascontiguousarray(np.array([[1,0],[0,-1]], dtype = np.complex128)))))
    return pauli_components

@njit
def numba_choice(population, weights, k):
    # Get cumulative weights
    wc = np.cumsum(weights)
    # Total of weights
    m = wc[-1]
    # Arrays of sample and sampled indices
    sample = np.empty(k, dtype=np.int32)
    sample_idx = np.full(k, -1, np.int32)
    # Sampling loop
    i = 0
    while i < k:
        # Pick random weight value
        r = m * np.random.rand()
        # Get corresponding index
        idx = np.searchsorted(wc, r, side='right')
        # Check index was not selected before
        # If not using Numba you can just do `np.isin(idx, sample_idx)`
        for j in range(i):
            if sample_idx[j] == idx:
                continue
        # Save sampled value and index
        sample[i] = population[idx]
        sample_idx[i] = population[idx]
        i += 1
    return sample