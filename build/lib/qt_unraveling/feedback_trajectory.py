'''
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
'''
import numpy as np
from numba import njit, objmode

import qt_unraveling.usual_operators_ as op

##########################################################################################
## Euler integrator 
##########################################################################################
@njit
def feedbackRhoEulerStep_(stateRho, drivingH, L_it, F_it, zeta, dt):
    stateRho = np.ascontiguousarray(stateRho)
    ## Lindblad super op
    D_c = op.D_vec(L_it, stateRho)*dt
    D_f = op.D_vec(F_it, stateRho)*dt

    ## Stocastic contribution
    comm_extra_term = np.zeros(np.shape(stateRho), dtype=np.complex128)
    Hw = np.zeros(np.shape(stateRho), dtype=np.complex128)
    for n_L, L in enumerate(L_it):
        L = np.ascontiguousarray(L)
        drho_rhod = np.dot(L, stateRho) + np.dot(stateRho, np.conjugate(np.transpose(L)))
        comm_extra_term += op.Com(F_it[n_L], drho_rhod)

        Hw += op.H(L - 1j*F_it[n_L], stateRho)*zeta[n_L]*np.sqrt(dt)

    return -1j*dt*(op.Com(drivingH, stateRho) + comm_extra_term) + D_c + D_f + Hw

###########################################
######  feedback trajectory functions #####
###########################################
@njit
def feedbackRhoTrajectory_(initialStateRho, timelist, drivingH, lindbladList, Flist, seed=0):
    ## Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]
    
    ## Number of Lindblad operators
    number_lindblad_op = np.shape(lindbladList(timelist[0],initialStateRho))[0]

    ## Stocastics increments
    zeta = 0
    with objmode(zeta='float64[:,:]'):
        rng = np.random.default_rng(seed)
        zeta = rng.normal(loc=0, scale=1, size=(number_lindblad_op, timeSteps))

    rho_trajectory = np.ascontiguousarray(np.zeros((timeSteps, np.shape(initialStateRho)[0], np.shape(initialStateRho)[0]), dtype=np.complex128))
    rho_trajectory[0] += np.ascontiguousarray(initialStateRho)
    for n_it, it in enumerate(timelist[:-1]):
        ## Lindblad ops for time it
        L_it = lindbladList(it, rho_trajectory[n_it])
        F_it = Flist(it, rho_trajectory[n_it])
        rho_trajectory[n_it+1] = rho_trajectory[n_it] + feedbackRhoEulerStep_(rho_trajectory[n_it], drivingH(it), L_it, F_it, zeta[:, n_it], dt)
    return rho_trajectory