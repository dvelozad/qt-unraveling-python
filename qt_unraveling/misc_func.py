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
from numba import njit, prange, complex128, float64, int32
from tqdm import tqdm

## Pauli matrices
sigmax = np.array([[0,1],[1,0]], dtype = np.complex128)
sigmay = np.array([[0,-1j],[1j,0]], dtype = np.complex128)
sigmaz = np.array([[1,0],[0,-1]], dtype = np.complex128)

## Ladder operators
sigmap = 0.5*(sigmax + 1j*sigmay)
sigmam = 0.5*(sigmax - 1j*sigmay)

# Pre-compile these constants for numba functions
_SIGMAX = np.ascontiguousarray(np.array([[0,1],[1,0]], dtype=np.complex128))
_SIGMAY = np.ascontiguousarray(np.array([[0,-1j],[1j,0]], dtype=np.complex128))
_SIGMAZ = np.ascontiguousarray(np.array([[1,0],[0,-1]], dtype=np.complex128))

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
    
    # Handle length mismatch between rho_list and timeList
    t_steps = min(len(timeList), len(rho_list))
    if len(timeList) != len(rho_list):
        print(f"Warning: Length mismatch between timeList ({len(timeList)}) and rho_list ({len(rho_list)}). Using only the first {t_steps} points.")
    
    # Truncate time list if needed
    t = np.linspace(timeList[0], timeList[-1], t_steps) if t_steps < len(timeList) else timeList
    
    for r_i in component:
        r_ = []
        for i in range(t_steps):
            r_.append(np.real(np.trace(rho_list[i].dot(dict_sigma[r_i]))))
        ax.plot(t, r_, line, label=f'{r_i} {label}', color=dict_color[r_i])

    ax.set_xlabel('time')
    ax.legend()

@njit(float64[:,:](complex128[:,:,:]), fastmath=True, cache=True)
def rhoBlochcomp_data(rho):
    pauli_components = np.zeros((3, rho.shape[0]), dtype=np.float64)
    for n_it in range(rho.shape[0]):
        rho_it = np.ascontiguousarray(rho[n_it])
        pauli_components[0, n_it] = np.real(np.trace(rho_it.dot(_SIGMAX)))
        pauli_components[1, n_it] = np.real(np.trace(rho_it.dot(_SIGMAY)))
        pauli_components[2, n_it] = np.real(np.trace(rho_it.dot(_SIGMAZ)))
    return pauli_components

# Pure Python version for handling different types of inputs
def rhoBlochcomp_data_py(rho):
    """
    Pure Python version of rhoBlochcomp_data that can handle different array shapes
    and doesn't rely on Numba's strict typing.
    """
    # Convert lists to numpy arrays if needed
    if isinstance(rho, list):
        rho = np.array(rho)
    
    # Handle 3D arrays (time series of density matrices)
    if len(rho.shape) == 3:
        time_steps = rho.shape[0]
        pauli_components = np.zeros((3, time_steps), dtype=np.float64)
        
        for n_it in range(time_steps):
            rho_it = rho[n_it]
            pauli_components[0, n_it] = np.real(np.trace(np.dot(rho_it, sigmax)))
            pauli_components[1, n_it] = np.real(np.trace(np.dot(rho_it, sigmay)))
            pauli_components[2, n_it] = np.real(np.trace(np.dot(rho_it, sigmaz)))
    
    # Handle 2D arrays (single density matrix)
    elif len(rho.shape) == 2:
        pauli_components = np.zeros((3, 1), dtype=np.float64)
        pauli_components[0, 0] = np.real(np.trace(np.dot(rho, sigmax)))
        pauli_components[1, 0] = np.real(np.trace(np.dot(rho, sigmay)))
        pauli_components[2, 0] = np.real(np.trace(np.dot(rho, sigmaz)))
    
    else:
        raise ValueError(f"Input rho has unsupported shape: {rho.shape}")
        
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
        try:
            # Try the non-Numba version first which can handle different input types
            x_t, y_t, z_t = rhoBlochcomp_data_py(rho)
            ax.plot(x_t, y_t, z_t, linewidth=1)
        except Exception as e:
            print(f"Warning: Error calculating Bloch components: {e}")
            print(f"Input shape: {np.shape(rho)}")
        
    ax.set_aspect('auto')
    plt.tight_layout()

@njit(int32[:](int32[:], float64[:], int32), fastmath=True, cache=True)
def numba_choice(population, weights, k):
    # Get cumulative weights
    wc = np.cumsum(weights)
    # Total of weights
    m = wc[-1]
    # Arrays of sample and sampled indices
    sample = np.empty(k, dtype=np.int32)
    sample_idx = np.full(k, -1, dtype=np.int32)
    # Sampling loop
    i = 0
    while i < k:
        # Pick random weight value
        r = m * np.random.rand()
        # Get corresponding index
        idx = np.searchsorted(wc, r, side='right')
        
        # Check if idx was already selected
        already_selected = False
        for j in range(i):
            if sample_idx[j] == idx:
                already_selected = True
                break
                
        if not already_selected:
            # Save sampled value and index
            sample[i] = population[idx]
            sample_idx[i] = idx  # Store actual index, not the value
            i += 1
    return sample

# For parallel computation with numba
@njit(parallel=True, fastmath=True, cache=True)
def parallel_process(data, n):
    result = np.zeros((n, data.shape[1]), dtype=data.dtype)
    for i in prange(n):
        result[i] = process_single(data[i % data.shape[0]])
    return result

@njit(fastmath=True, cache=True)
def process_single(data_row):
    # Example processing function for a single data row
    return data_row