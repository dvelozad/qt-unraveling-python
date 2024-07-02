'''
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
'''
import numpy as np
from numba import njit, objmode, float64, complex128, int64, types
import qt_unraveling.usual_operators as op

from numba.pycc import CC

cc = CC('diffusive_trajectory')
cc.verbose = True

##########################################################################################
## Euler integrator 
##########################################################################################
@njit(complex128[:,:](complex128[:,:], complex128[:,:], complex128[:,:,:], complex128[:,:,:], float64[:], float64))
def diffusiveRhoEulerStep_(stateRho, drivingH, original_L_it, L_it, zeta, dt):
    ## Lindblad super op
    D_c = op.D_vec(original_L_it, stateRho)*dt

    ## Stocastic contribution
    Hw = np.zeros(np.shape(stateRho), dtype=np.complex128)
    for n_L, L in enumerate(L_it):
        Hw += op.H(L, stateRho)*zeta[n_L]*np.sqrt(dt)

    return -1j*op.Com(drivingH, stateRho)*dt + D_c + Hw

##########################################################################################
## Milstein integrator 
##########################################################################################   
@njit(complex128[:,:](complex128[:,:], complex128[:,:,:], complex128[:,:]))
def diffusiveRhoMilstein_f(drivingH, original_L_it, stateRho):
    commu1 = -1j*op.Com(drivingH, stateRho)                     
    Dc = op.D_vec(original_L_it, stateRho)
    return commu1 + Dc 

@njit(complex128[:,:,:](complex128[:,:,:], complex128[:,:]))
def diffusiveRhoMilstein_g(L_it, stateRho):
    HW = np.zeros((np.shape(L_it)[0],)+np.shape(stateRho), dtype = np.complex128)
    for n_L, L_i in enumerate(L_it):
        HW[n_L] += op.H(L_i, stateRho)
    return HW

@njit(complex128[:,:](complex128[:,:], complex128[:,:], complex128[:,:,:],  complex128[:,:,:], float64[:], float64))
def diffusiveRhoMilsteinStep(stateRho, drivingH, original_L_it, L_it, zeta, dt):

    gi = diffusiveRhoMilstein_g(L_it, stateRho)
    fi = diffusiveRhoMilstein_f(drivingH, original_L_it, stateRho)

    G_w = np.zeros(np.shape(stateRho), dtype = np.complex128)
    G_aux = np.zeros(np.shape(stateRho), dtype = np.complex128)

    for n_Li, L_i in enumerate(L_it):
        dw_i = zeta[n_Li]*np.sqrt(dt)
        G_w += dw_i*gi[n_Li]

        for n_Lj, L_j in enumerate(L_it):
            dw_j = zeta[n_Lj]*np.sqrt(dt)

            if n_Li == n_Lj:
                G_aux += 0.5*op.H_DH(L_i, L_j, stateRho)*(dw_i*dw_j - dt)
            else:
                G_aux += 0.5*op.H_DH(L_i, L_j, stateRho)*dw_i*dw_j

    return fi*dt + G_aux + G_w

############################################
######  diffusive trajectory functions #####
############################################
#@njit(complex128[:,:,:](complex128[:,:], float64[:], complex128[:,:], complex128[:,:,:], complex128[:,:,:], types.unicode_type, int64))
#@cc.export('diffusiveRhoTrajectory_td', complex128[:,:,:](complex128[:,:], float64[:], types.FunctionType, types.FunctionType, types.FunctionType, types.unicode_type, int64))
#@njit
def diffusiveRhoTrajectory_td(initialStateRho, timelist, drivingH, original_lindbladList, lindbladList, method='euler', seed=0):
    ## Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]
    
    ## Number of Lindblad operators
    number_lindblad_op = np.shape(lindbladList(timelist[0],initialStateRho))[0]

    ## Stocastics increments
    #rng = np.random.default_rng(seed)
    np.random.seed(seed) 
    zeta = np.random.normal(loc=0, scale=1, size=(number_lindblad_op, timeSteps))

    rho_trajectory = np.ascontiguousarray(np.empty(np.shape(timelist) + np.shape(initialStateRho), dtype=np.complex128))
    rho_trajectory[0,:,:] = np.ascontiguousarray(initialStateRho)

    ## Integrator
    if method == 'euler':
        for n_it, it in enumerate(timelist[:-1]):
            ## Lindblad ops for time it
            L_it = lindbladList(it, rho_trajectory[n_it])
            original_L_it = original_lindbladList(it)
            rho_trajectory[n_it+1,:,:] = rho_trajectory[n_it] + diffusiveRhoEulerStep_(rho_trajectory[n_it], drivingH(it), original_L_it, L_it, zeta[:, n_it], dt)
            
    elif method == 'milstein':
        for n_it, it in enumerate(timelist[:-1]):
            ## Lindblad ops for time it
            L_it = lindbladList(it, rho_trajectory[n_it])
            original_L_it = original_lindbladList(it)
            rho_trajectory[n_it+1,:,:] = rho_trajectory[n_it] + diffusiveRhoMilsteinStep(rho_trajectory[n_it], drivingH(it), original_L_it, L_it, zeta[:, n_it], dt)

    return rho_trajectory

#@njit(complex128[:,:,:](complex128[:,:], float64[:], complex128[:,:], complex128[:,:,:], complex128[:,:,:], types.unicode_type, int64))
#@cc.export('diffusiveRhoTrajectory_', complex128[:,:,:](complex128[:,:], float64[:], complex128[:,:], complex128[:,:,:], complex128[:,:,:], types.unicode_type, int64))
#@njit
def diffusiveRhoTrajectory_(initialStateRho, timelist, drivingH, original_lindbladList, lindbladList, method='euler', seed=0):
    ## Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]
    
    ## Number of Lindblad operators
    number_lindblad_op = np.shape(lindbladList)[0]

    ## Stocastics increments
    #rng = np.random.default_rng(seed)
    np.random.seed(seed) 
    zeta = np.random.normal(loc=0, scale=1, size=(number_lindblad_op, timeSteps))

    rho_trajectory = np.ascontiguousarray(np.empty(np.shape(timelist) + np.shape(initialStateRho), dtype=np.complex128))
    rho_trajectory[0,:,:] = np.ascontiguousarray(initialStateRho)

    ## Integrator
    if method == 'euler':
        for n_it, it in enumerate(timelist[:-1]):       
            rho_trajectory[n_it+1,:,:] = rho_trajectory[n_it] + diffusiveRhoEulerStep_(rho_trajectory[n_it], drivingH, original_lindbladList, lindbladList, zeta[:, n_it], dt)

    elif method == 'milstein':
        for n_it, it in enumerate(timelist[:-1]):       
            rho_trajectory[n_it+1,:,:] = rho_trajectory[n_it] + diffusiveRhoMilsteinStep(rho_trajectory[n_it], drivingH, original_lindbladList, lindbladList, zeta[:, n_it], dt)

    return rho_trajectory


if __name__ == "__main__":
    cc.compile()

# @cc.export('diffusiveRhoEulerStep_', complex128[:,:](complex128[:,:], complex128[:,:], complex128[:,:,:], complex128[:,:,:], float64[:], float64)) 
# @cc.export('diffusiveRhoMilstein_f', complex128[:,:](complex128[:,:], complex128[:,:,:], complex128[:,:]))
# @cc.export('diffusiveRhoMilsteinStep', complex128[:,:](complex128[:,:], complex128[:,:], complex128[:,:,:],  complex128[:,:,:], float64[:], float64))

