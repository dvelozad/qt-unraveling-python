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
from qt_unraveling.current_measurement import get_currens_measurement
from qt_unraveling.diffusive_trajectory import diffusiveRhoEulerStep_

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

## Implementing equation 
@njit
def feedbackRhoTrajectory_delay(initialStateRho, timelist, drivingH, original_lindbladList, lindbladList, Flist, tau, seed=0):
    ## Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]
    n_tau = int(tau/dt)

    ## Hamiltonian shape
    hamiltonian_dimension = np.shape(drivingH(0))
    
    ## Number of Lindblad operators
    number_lindblad_op = np.shape(lindbladList(timelist[0], initialStateRho))[0]

    ## Stocastics increments
    zeta = 0
    with objmode(zeta='float64[:,:]'):
        rng = np.random.default_rng(seed)
        zeta = rng.normal(loc=0, scale=1, size=(number_lindblad_op, timeSteps))

    feedback_hamiltonian = np.zeros(hamiltonian_dimension, dtype=np.complex128)
    second_order_expansion_term = np.zeros(np.shape(initialStateRho), dtype=np.complex128)
    missing_term_order_dt = np.zeros(np.shape(initialStateRho), dtype=np.complex128)
    original_L_it = original_lindbladList

    rho_trajectory = np.ascontiguousarray(np.zeros((timeSteps, np.shape(initialStateRho)[0], np.shape(initialStateRho)[0]), dtype=np.complex128))
    rho_trajectory[0] += np.ascontiguousarray(initialStateRho)
    for n_it, it in enumerate(timelist[:-1]):
        ## Lindblad ops for time it
        L_it = lindbladList(it, rho_trajectory[n_it])

        ## reset terms
        feedback_hamiltonian *= 0
        second_order_expansion_term *= 0
        missing_term_order_dt *= 0
        if n_it >= n_tau:
            F_it = Flist(it - tau, rho_trajectory[n_it - n_tau])
            measurements = get_currens_measurement(rho_trajectory[n_it - n_tau], L_it)
            for n_measure, measure in enumerate(measurements):
                ## feedback hamiltonian like term
                feedback_hamiltonian += (measure + zeta[n_measure, n_it - n_tau]/np.sqrt(dt))*F_it[n_measure]
                ## missing term construction - check description
                for m_measure, _measure in enumerate(measurements): 
                    missing_term_order_dt += -1j*dt*zeta[n_measure, n_it - n_tau]*zeta[m_measure, n_it]*op.Com(F_it[n_measure], op.H(L_it[m_measure], rho_trajectory[n_it]))
            ## K**2 term
            second_order_expansion_term += op.D_vec(F_it, rho_trajectory[n_it])*dt

        rho_trajectory[n_it+1] = rho_trajectory[n_it] + diffusiveRhoEulerStep_(rho_trajectory[n_it], drivingH(it) + feedback_hamiltonian, original_L_it, L_it, zeta[:, n_it], dt) + second_order_expansion_term + missing_term_order_dt

    return rho_trajectory