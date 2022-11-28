'''
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
'''
import numpy as np
from numba import njit, complex128, float64
from scipy.integrate import solve_ivp

import qt_unraveling.usual_operators_ as op

def scipy_integrator(differential_operator, initialStateRho, timeList, method = 'BDF', rrtol = 1e-5, aatol=1e-5, last_point=False):
    dim = np.shape(initialStateRho)[0]
    x0 = initialStateRho.reshape(-1) 
    ##################################################################
    def odefun(t,x):
        rho = x.reshape([dim,dim])
        dx = differential_operator(rho, t)
        return dx.reshape(-1)
    ##################################################################
    if not last_point:
        sol = solve_ivp(odefun, [timeList[0], timeList[-1]], x0, t_eval = timeList, method = method, rtol = rrtol, atol = aatol)
        rho_T = [sol.y[:,i].reshape([dim,dim]) for i in range(len(sol.t))] 
        return rho_T
    else:
        sol = solve_ivp(odefun, [timeList[0], timeList[-1]], x0, t_eval = [timeList[-1]], method = method, rtol = rrtol, atol = aatol)
        rho_T = [sol.y[:,i].reshape([dim,dim]) for i in range(len(sol.t))] 
        return rho_T[0]

def vonneumann_operator(drivingH, stateRho, it):
    unitary_evolution = -1j*op.Com(drivingH(it), stateRho)
    return unitary_evolution

def standartLindblad_operator(drivingH, lindbladList, stateRho, it):
    c = lindbladList(it, stateRho) ## Original operators
    unitary_evolution = -1j*op.Com(drivingH(it), stateRho)
    Dc = op.D_vec(c, stateRho)
    return unitary_evolution + Dc

def feedbackEvoladaptative_operator(System_obj, stateRho, it):
    drivingH = System_obj.H
    lindbladList = System_obj.original_cList
    Flist = System_obj.FList
    
    if System_obj.nonfixedUnraveling:
        mMatrix = System_obj.update_defintions(it, stateRho)[1]
    else:
        mMatrix = System_obj.M_rep
    
    L_it = lindbladList(it)   ## Original operators
    F_it = Flist(it, stateRho)

    ## M rep related definitons
    mMatrix_dag = np.transpose(np.conjugate(mMatrix))
    sqrt_eta = op.sqrt_jit(np.eye(2*np.shape(L_it)[0]) - np.dot(mMatrix_dag, mMatrix))

    f_M_dag = np.zeros((np.shape(L_it)[0],)+np.shape(stateRho), dtype = np.complex128)
    M_f = np.zeros((np.shape(L_it)[0],)+np.shape(stateRho), dtype = np.complex128)
    c_M_f = np.zeros((np.shape(L_it)[0],)+np.shape(stateRho), dtype = np.complex128)

    Hfe = np.zeros(np.shape(stateRho), dtype = np.complex128)

    for n_L, L in enumerate(L_it):
        f_M_dag_i = np.zeros(np.shape(stateRho), dtype = np.complex128)
        M_f_i = np.zeros(np.shape(stateRho), dtype = np.complex128)

        for n_F, F in enumerate(F_it):
            f_M_dag_i += F*mMatrix_dag[n_F,n_L]
            M_f_i += mMatrix[n_L,n_F]*F

        f_M_dag[n_L] += np.dot(f_M_dag_i, L)
        M_f[n_L] += np.dot(np.transpose(np.conjugate(L)), M_f_i)
        c_M_f[n_L] += L - 1j*M_f_i

        Hfe += 0.5*(M_f[n_L] + f_M_dag[n_L])

    unitary_evolution = -1j*op.Com(drivingH(it) + Hfe, stateRho)

    S_f = np.zeros((np.shape(F_it)[0],)+np.shape(stateRho), dtype = np.complex128)
    for ni_F in range(np.shape(F_it)[0]):
        for nj_F, F_j in enumerate(F_it):
            S_f[ni_F] += sqrt_eta[ni_F][nj_F]*F_j
    
    DSf = op.D_vec(S_f, stateRho)
    Dcf = op.D_vec(c_M_f, stateRho)

    return unitary_evolution + Dcf + DSf


# def simple_feedbackUnconditional_operator(drivingH, lindbladList, Flist, mMatrix, stateRho, it):
#     #L_it = lindbladList(it)  ## Original operators
#     L_it = lindbladList(it, stateRho)  ## Original operators
#     F_it = Flist(it, stateRho)

#     stateRho = np.ascontiguousarray(stateRho)
#     ## Lindblad super op
#     D_c = op.D_vec(L_it, stateRho)
#     D_f = op.D_vec(F_it, stateRho)

#     ## Stocastic contribution
#     comm_extra_term = np.zeros(np.shape(stateRho), dtype=np.complex128)
#     Hw = np.zeros(np.shape(stateRho), dtype=np.complex128)
#     for n_L, L in enumerate(L_it):
#         L = np.ascontiguousarray(L)
#         drho_rhod = np.dot(L, stateRho) + np.dot(stateRho, np.conjugate(np.transpose(L)))
#         comm_extra_term += op.Com(F_it[n_L], drho_rhod)

#     return -1j*(op.Com(drivingH(it), stateRho) + comm_extra_term) + D_c + D_f 

def feedbackEvol_operator(drivingH, original_lindbladList, lindbladList, Flist, stateRho, it):
    L_it = lindbladList(it, stateRho)   ## Original operators
    O_L_it = original_lindbladList(it)
    F_it = Flist(it, stateRho)

    stateRho = np.ascontiguousarray(stateRho)
    ## Lindblad super op
    D_c = op.D_vec(O_L_it, stateRho)
    D_f = op.D_vec(F_it, stateRho)

    ## Stocastic contribution
    comm_extra_term = np.zeros(np.shape(stateRho), dtype=np.complex128)
    for n_L, L in enumerate(L_it):
        L = np.ascontiguousarray(L)
        drho_rhod = np.dot(L, stateRho) + np.dot(stateRho, np.conjugate(np.transpose(L)))
        comm_extra_term += op.Com(F_it[n_L], drho_rhod)

    return -1j*(op.Com(drivingH(it), stateRho) + comm_extra_term) + D_c + D_f 