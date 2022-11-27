'''
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
'''
import numpy as np
from numpy.random import normal, seed
from numba import njit, objmode, float64, complex128, int64
import usual_operators_ as op

@njit
def diffusiveRhoEulerStep_(stateRho, drivingH, original_L_it, L_it, zeta, dt):
    ## Lindblad super op
    D_c = op.D_vec(original_L_it, stateRho)*dt

    ## Stocastic contribution
    Hw = np.zeros(np.shape(stateRho), dtype=np.complex128)
    for n_L, L in enumerate(L_it):
        Hw += op.H(L, stateRho)*zeta[n_L]*np.sqrt(dt)

    return -1j*op.Com(drivingH, stateRho)*dt + D_c + Hw

@njit
def diffusiveRhoTrajectory_td(initialStateRho, timelist, drivingH, original_lindbladList, lindbladList, seed=0):
    ## Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]
    
    ## Number of Lindblad operators
    number_lindblad_op = np.shape(lindbladList(timelist[0],initialStateRho))[0]

    ## Stocastics increments
    zeta = 0
    with objmode(zeta='float64[:,:]'):
        rng = np.random.default_rng(143525+seed**10)
        zeta = rng.normal(loc=0, scale=1, size=(number_lindblad_op, timeSteps))

    rho_trajectory = np.ascontiguousarray(np.empty(np.shape(timelist) + np.shape(initialStateRho), dtype=np.complex128))
    rho_trajectory[0,:,:] = np.ascontiguousarray(initialStateRho)
    for n_it, it in enumerate(timelist[:-1]):
        ## Lindblad ops for time it
        L_it = lindbladList(it, rho_trajectory[n_it])
        original_L_it = original_lindbladList(it)
        
        rho_trajectory[n_it+1,:,:] = rho_trajectory[n_it] + diffusiveRhoEulerStep_(rho_trajectory[n_it], drivingH(it), original_L_it, L_it, zeta[:, n_it], dt)
        rho_trajectory[n_it+1,:,:] = rho_trajectory[n_it+1,:,:]/np.linalg.norm(rho_trajectory[n_it+1,:,:])
    return rho_trajectory

@njit(complex128[:,:,:](complex128[:,:], float64[:], complex128[:,:], complex128[:,:,:], complex128[:,:,:], int64))
def diffusiveRhoTrajectory_(initialStateRho, timelist, drivingH, original_lindbladList, lindbladList, seed=0):
    ## Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]
    
    ## Number of Lindblad operators
    number_lindblad_op = np.shape(lindbladList)[0]

    ## Stocastics increments
    zeta = 0
    with objmode(zeta='float64[:,:]'):
        rng = np.random.default_rng(143525+seed**10)
        zeta = rng.normal(loc=0, scale=1, size=(number_lindblad_op, timeSteps))

    rho_trajectory = np.ascontiguousarray(np.empty(np.shape(timelist) + np.shape(initialStateRho), dtype=np.complex128))
    rho_trajectory[0,:,:] = np.ascontiguousarray(initialStateRho)
    for n_it, it in enumerate(timelist[:-1]):       
        rho_trajectory[n_it+1,:,:] = rho_trajectory[n_it] + diffusiveRhoEulerStep_(rho_trajectory[n_it], drivingH, original_lindbladList, lindbladList, zeta[:, n_it], dt)
        rho_trajectory[n_it+1,:,:] = rho_trajectory[n_it+1,:,:]/np.linalg.norm(rho_trajectory[n_it+1,:,:])
    return rho_trajectory