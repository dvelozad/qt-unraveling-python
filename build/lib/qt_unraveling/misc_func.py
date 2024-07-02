"""
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
"""
import numpy as np
import matplotlib.pyplot as plt
from multiprocess import Pool, cpu_count
from numba import njit
from tqdm import tqdm

## Pauli matrices
sigmax = np.array([[0,1],[1,0]], dtype = np.complex128)
sigmay = np.array([[0,-1j],[1j,0]], dtype = np.complex128)
sigmaz = np.array([[1,0],[0,-1]], dtype = np.complex128)

## Ladder operators
sigmap = 0.5*(sigmax + 1j*sigmay)
sigmam = 0.5*(sigmax - 1j*sigmay)

def parallel_run(fun, arg_list, tqdm_bar=False):
    if tqdm_bar:
        m = []
        with Pool(processes=cpu_count()) as p:
            with tqdm(total=np.shape(arg_list)[0], ncols=60) as pbar:
                for _ in p.imap(fun, arg_list):
                    m.append(_)
                    pbar.update()
    else:
        with Pool(processes=cpu_count()) as p:
            m = p.map(fun, arg_list)
    return m


def figure():
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    fig.set_size_inches(w=10, h=5)

    ax.spines["top"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    
    ax.grid()
    return fig, ax

def rhoBlochcomp_plot(rho_list, timeList, ax = None, component = ['rx', 'ry', 'rz'], line = '-', label = ''):  
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

#@njit
def rhoBlochcomp_data(rho):
    pauli_components = np.zeros((3, np.shape(rho)[0]), dtype=np.float64)
    for n_it, rho_it in enumerate(rho):
        rho_it = np.ascontiguousarray(rho_it)
        pauli_components[0][n_it] += np.real(np.trace(rho_it.dot(np.ascontiguousarray(np.array([[0,1],[1,0]], dtype = np.complex128)))))
        pauli_components[1][n_it] += np.real(np.trace(rho_it.dot(np.ascontiguousarray(np.array([[0,-1j],[1j,0]], dtype = np.complex128)))))
        pauli_components[2][n_it] += np.real(np.trace(rho_it.dot(np.ascontiguousarray(np.array([[1,0],[0,-1]], dtype = np.complex128)))))
    return pauli_components

def rhoBlochSphere(rho_list):
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection='3d')

    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color="black", alpha=0.2, shade=True, cmap='twilight_shifted_r')
    ax.plot_wireframe(x, y, z, color="black", alpha=0.1)

    ## Bloch sphere components
    for rho in rho_list:
        x_t, y_t, z_t = rhoBlochcomp_data(rho)
        ax.plot(x_t, y_t, z_t, linewidth=1)
        
    ax.set_aspect('auto')
    plt.tight_layout()

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