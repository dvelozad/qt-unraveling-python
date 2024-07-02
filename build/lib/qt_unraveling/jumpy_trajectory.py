'''
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
'''
import numpy as np
from numba import njit, objmode, float64, complex128, int64

import qt_unraveling.usual_operators as op
import qt_unraveling.misc_func as misc

#@njit(int64(complex128[:,:], complex128[:,:,:], float64, int64))
def dNRho(stateRho, measurement_op_list, dt, seed=0):
    weight = np.zeros(np.shape(measurement_op_list)[0], dtype=np.float64)
    M_index = np.zeros(np.shape(measurement_op_list)[0], dtype=np.int64)

    for mu, Mmu in enumerate(measurement_op_list):
        Mmu = np.ascontiguousarray(Mmu)
        Mmu_dag_Mmu = np.dot(np.conjugate(np.transpose(Mmu)), Mmu)
        weight[mu] += dt*np.real(np.trace(np.dot(stateRho, Mmu_dag_Mmu)))
        M_index[mu] += mu
    
    np.random.seed(seed)
    R = np.random.rand()
    if R < sum(weight): 
        jump_index = misc.numba_choice(M_index, weight, 1)[0]
    else:
        jump_index = np.shape(measurement_op_list)[0]
    return jump_index

#@njit(complex128[:,:,:](complex128[:,:], float64[:], complex128[:,:], complex128[:,:,:], complex128[:], complex128[:,:,:], complex128[:], int64))
def ortogonal_mixing(oMatrix, coherent_fields, L_it):
    new_ops = np.zeros(np.shape(L_it), dtype=np.complex128)
    for n_O, O in enumerate(oMatrix):
        new_ops[n_O] += coherent_fields[n_O]*np.eye(np.shape(L_it)[0])
        for n_L, L in enumerate(L_it):
            new_ops[n_O] += O[n_L]*L
    return new_ops

#@njit(complex128[:,:,:](complex128[:,:], float64[:], complex128[:,:], complex128[:,:,:], complex128[:], complex128[:,:,:], complex128[:], int64))
def coherent_field_mixing(coherent_fields, L_it):
    new_ops = np.zeros(np.shape(L_it), dtype=np.complex128)
    for n_L, L in enumerate(L_it):
        new_ops[n_L] += L + coherent_fields[n_L]*np.eye(np.shape(L)[0])
    return new_ops

#@njit(complex128[:,:,:](complex128[:,:], float64[:], complex128[:,:], complex128[:,:,:], complex128[:], complex128[:,:,:], complex128[:], int64))
def jumpRhoTrajectory_td(initialStateRho, timelist, drivingH, original_lindbladList, eta_diag, lindbladList, coherent_fields, seed):
    ## Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]

    ## Number of Lindblad operators
    num_lindblad_channels = np.shape(lindbladList(timelist[0],initialStateRho))[0]

    rho_trajectory = np.ascontiguousarray(np.zeros(np.shape(timelist) + np.shape(initialStateRho), dtype=np.complex128))
    rho_trajectory[0] += initialStateRho

    no_jump_term_1 = np.ascontiguousarray(np.zeros(np.shape(initialStateRho), dtype=np.complex128))
    no_jump_term_2 = np.ascontiguousarray(np.zeros(np.shape(initialStateRho), dtype=np.complex128))
    inefficient_term = np.ascontiguousarray(np.zeros(np.shape(initialStateRho), dtype=np.complex128))

    for n_it, it in enumerate(timelist[:-1]):
        L_it = lindbladList(it, rho_trajectory[n_it])
        original_L_it = original_lindbladList(it)

        for n_i, L_i in enumerate(original_L_it):
            inefficient_term += (1 - eta_diag[n_i])*op.D(L_i, rho_trajectory[n_it])*dt

        ## coherent_field mix
        #ortogonal_ops = ortogonal_mixing(oMatrix, coherent_fields, L_it)
        coherent_field_ops = coherent_field_mixing(coherent_fields, L_it)

        ## Jump index
        jump_index = dNRho(rho_trajectory[n_it], coherent_field_ops, dt, seed*timeSteps+n_it)

        ## Euler step
        if jump_index < num_lindblad_channels:
            rho_trajectory[n_it+1] += rho_trajectory[n_it] + op.G(coherent_field_ops[jump_index], rho_trajectory[n_it]) + inefficient_term

        elif jump_index == num_lindblad_channels:
            for n_r, L_r in enumerate(L_it):
                L_r = np.ascontiguousarray(L_r)
                no_jump_term_1 += -0.5*np.dot(np.transpose(np.conjugate(L_r)), L_r)
                no_jump_term_2 += -coherent_fields[n_r]*L_r
            
            rho_trajectory[n_it+1] += rho_trajectory[n_it] + op.H(-1j*drivingH(it) + no_jump_term_1 + no_jump_term_2, rho_trajectory[n_it])*dt + inefficient_term
            no_jump_term_1, no_jump_term_2, inefficient_term = 0*no_jump_term_1, 0*no_jump_term_2, 0*inefficient_term

    return rho_trajectory

##@njit(complex128[:,:,:](complex128[:,:], float64[:], complex128[:,:], complex128[:,:,:], complex128[:], complex128[:,:,:], complex128[:], int64))
#@njit(complex128[:,:,:](complex128[:,:], float64[:], complex128[:,:], complex128[:,:,:], complex128[:], complex128[:,:,:], complex128[:], int64))
def jumpRhoTrajectory_(initialStateRho, timelist, drivingH, original_lindbladList, eta_diag, lindbladList, coherent_fields, seed):
    ## Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]

    ## Number of Lindblad operators
    num_lindblad_channels = np.shape(lindbladList)[0]

    rho_trajectory = np.ascontiguousarray(np.zeros(np.shape(timelist) + np.shape(initialStateRho), dtype=np.complex128))
    rho_trajectory[0] += initialStateRho

    no_jump_term_1 = np.ascontiguousarray(np.zeros(np.shape(initialStateRho), dtype=np.complex128))
    no_jump_term_2 = np.ascontiguousarray(np.zeros(np.shape(initialStateRho), dtype=np.complex128))
    inefficient_term = np.ascontiguousarray(np.zeros(np.shape(initialStateRho), dtype=np.complex128))
    for n_it, it in enumerate(timelist[:-1]):
        L_it = lindbladList
        original_L_it = original_lindbladList

        for n_i, L_i in enumerate(original_L_it):
            inefficient_term += (1 - eta_diag[n_i])*op.D(L_i, rho_trajectory[n_it])*dt

        ## coherent_field mix
        #ortogonal_ops = ortogonal_mixing(oMatrix, coherent_fields, L_it)
        coherent_field_ops = coherent_field_mixing(coherent_fields, L_it)

        ## Jump index
        jump_index = dNRho(rho_trajectory[n_it], coherent_field_ops, dt, seed*timeSteps+n_it)

        ## Euler step
        if jump_index < num_lindblad_channels:
            rho_trajectory[n_it+1] += rho_trajectory[n_it] + op.G(coherent_field_ops[jump_index], rho_trajectory[n_it]) + inefficient_term

        elif jump_index == num_lindblad_channels:
            for n_r, L_r in enumerate(L_it):
                L_r = np.ascontiguousarray(L_r)
                no_jump_term_1 += -0.5*np.dot(np.transpose(np.conjugate(L_r)), L_r)
                no_jump_term_2 += -coherent_fields[n_r]*L_r
            
            rho_trajectory[n_it+1] += rho_trajectory[n_it] + op.H(-1j*drivingH + no_jump_term_1 + no_jump_term_2, rho_trajectory[n_it])*dt + inefficient_term
            no_jump_term_1, no_jump_term_2, inefficient_term = 0*no_jump_term_1, 0*no_jump_term_2, 0*inefficient_term

    return rho_trajectory