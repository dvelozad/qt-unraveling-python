'''
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
'''
import numpy as np
from numba import njit, jit, complex128, float64

@njit
def get_currens_measurement(stateRho, L_it):
    expected_value_quadtature_op = np.zeros(np.shape(L_it)[0], dtype=np.complex128)
    for n_L, L in enumerate(L_it):
        expected_value_quadtature_op[n_L] += np.trace(np.dot(L + np.conjugate(np.transpose(L)), stateRho))
    return expected_value_quadtature_op
