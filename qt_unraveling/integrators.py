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

import qt_unraveling.usual_operators as op

def custom_rungekutta_integrator(differential_operator, initialStateRho : np.ndarray, timeList : np.ndarray, last_point : bool = False) -> np.ndarray:
    """
    Performs the integration of the time-evolution of the differential_operator
    
    Parameters:
    differential_operator (function): the differential operator to integrate as a njitted function
    initialStateRho (array): Initial state
    timeList (array): Time list to integrate
    last_point (bool) : True to return just the final point of the evolution
    
    Returns:
    rho_list (array): resulting array of density matrices for each time step if last_point False, the last point of evolution otherwise
    """
    if last_point:
        rho_list = custom_rungekutta_integrator_last_point(differential_operator, initialStateRho, timeList)
    else:
        rho_list = custom_rungekutta_integrator_full_range(differential_operator, initialStateRho, timeList)
    return rho_list
    
def scipy_integrator(differential_operator, initialStateRho : np.ndarray, timeList : np.ndarray, method : str = 'BDF', rrtol : float = 1e-5, aatol : float = 1e-5, last_point : bool = False) -> np.ndarray:
    """
    Performs the integration of the time-evolution of the differential_operator using the scipy solve_ivp function
    
    Parameters:
    differential_operator (function): the differential operator to integrate as a njitted function
    initialStateRho (array): Initial state
    timeList (array): Time list to integrate
    method (str) : scipy integrator method
    rrtol (float) : If the relative error estimate is larger than rtol, the computation continues until the error is reduced below this threshold, or until a maximum number of iterations is reached
    aatol (float) : The atol parameter sets the minimum absolute error tolerance for the solution. If the numerical method used by the function estimates that the absolute error in the solution is smaller than atol, the computation is considered to have converged and the function returns the solution
    last_point (bool) : True to return just the final point of the evolution
    
    Returns:
    rho_list (array): resulting array of density matrices for each time step if last_point False, the last point of evolution otherwise
    """
    dim = np.shape(initialStateRho)[0]
    # Convert to a flattened real-valued array for solve_ivp
    x0_real = np.concatenate([initialStateRho.real.reshape(-1), initialStateRho.imag.reshape(-1)])
    
    ##################################################################
    def odefun(t, x):
        # Reconstruct the complex density matrix
        x_real = x[:dim*dim]
        x_imag = x[dim*dim:]
        rho_real = x_real.reshape([dim, dim])
        rho_imag = x_imag.reshape([dim, dim])
        rho = rho_real + 1j * rho_imag
        
        # Compute the derivative
        dx_complex = differential_operator(rho, t)
        
        # Convert back to a flattened real-valued array
        dx_real = dx_complex.real.reshape(-1)
        dx_imag = dx_complex.imag.reshape(-1)
        return np.concatenate([dx_real, dx_imag])
    ##################################################################
    
    if not last_point:
        sol = solve_ivp(odefun, [timeList[0], timeList[-1]], x0_real, t_eval=timeList, 
                       method=method, rtol=rrtol, atol=aatol)
        
        # Reconstruct complex matrices
        rho_T = []
        for i in range(len(sol.t)):
            y_real = sol.y[:dim*dim, i].reshape([dim, dim])
            y_imag = sol.y[dim*dim:, i].reshape([dim, dim])
            rho_T.append(y_real + 1j * y_imag)
        
        return rho_T
    else:
        sol = solve_ivp(odefun, [timeList[0], timeList[-1]], x0_real, t_eval=[timeList[-1]], 
                       method=method, rtol=rrtol, atol=aatol)
        
        # Only reconstruct the last point
        y_real = sol.y[:dim*dim, -1].reshape([dim, dim])
        y_imag = sol.y[dim*dim:, -1].reshape([dim, dim])
        rho_final = y_real + 1j * y_imag
        
        return [rho_final]

@njit(complex128[:,:](complex128[:,:], complex128[:,:], float64), fastmath=True, cache=True)
def vonneumann_operator(drivingH, stateRho, it):
    """
    Gives the evaluation of the von neumann equation differential operator for a given state and time
    
    Parameters:
    drivingH (function): The Hamiltonian operator as a function of time
    stateRho (array): density matrix state
    it (float): time to evaluate

    Returns:
    unitary_evolution (array): resulting array of density matrices for each time step if last_point False, the last point of evolution otherwise
    """
    unitary_evolution = -1j * op.Com(drivingH, stateRho)
    return unitary_evolution

@njit(fastmath=True, cache=True)
def standartLindblad_operator(drivingH, lindbladList, stateRho, it):
    """
    Gives the evaluation of the Lindablad equation differential operator for a given state and time
    
    Parameters:
    drivingH (function): The Hamiltonian operator as a function of time
    lindbladList (function) : List of Lindablad operators as a function of time and the state
    stateRho (array): density matrix state
    it (float): time to evaluate

    Returns:
    unitary_evolution + Dc (array): resulting array of density matrices for each time step if last_point False, the last point of evolution otherwise
    """
    c = lindbladList(it, stateRho) ## Original operators
    unitary_evolution = -1j * op.Com(drivingH(it), stateRho)
    Dc = op.D_vec(c, stateRho)
    return unitary_evolution + Dc

# Add a pure Python version of the Lindblad operator
def standartLindblad_operator_py(drivingH, lindbladList, stateRho, it):
    """
    Pure Python version of the standard Lindblad operator without Numba decorators
    
    Parameters:
    drivingH (function): The Hamiltonian operator as a function of time
    lindbladList (function) : List of Lindblad operators as a function of time and the state
    stateRho (array): density matrix state
    it (float): time to evaluate

    Returns:
    Array: Result of applying the Lindblad operator
    """
    try:
        c = lindbladList(it, stateRho)  # Try with state as argument
    except TypeError:
        try:
            c = lindbladList(it)  # Try with just time
        except TypeError:
            c = lindbladList  # Use as is if it's not a function
    
    # Calculate Hamiltonian evolution part
    H_it = drivingH(it)
    H_rho = np.dot(H_it, stateRho)
    rho_H = np.dot(stateRho, H_it)
    unitary_evolution = -1j * (H_rho - rho_H)
    
    # Calculate dissipative part
    Dc = np.zeros(stateRho.shape, dtype=np.complex128)
    for L in c:
        L_rho = np.dot(L, stateRho)
        rho_Ldag = np.dot(stateRho, np.conjugate(np.transpose(L)))
        LdagL_rho = np.dot(np.dot(np.conjugate(np.transpose(L)), L), stateRho)
        rho_LdagL = np.dot(stateRho, np.dot(np.conjugate(np.transpose(L)), L))
        Dc += L_rho @ np.conjugate(np.transpose(L)) - 0.5 * (LdagL_rho + rho_LdagL)
    
    return unitary_evolution + Dc

@njit(fastmath=True, cache=True)
def feedbackEvol_operator(drivingH, original_lindbladList, lindbladList, Flist, stateRho, it):
    """
    Gives the evaluation of the feedback equation differential operator for a given state and time for the case of fixed unraveling parametrization
    
    Parameters:
    drivingH (function): The Hamiltonian operator as a function of time
    original_lindbladList (function) : List of Linblad operators prior to applying the unraveling parametrization as a function of time
    lindbladList (function) : List of Linblad operators posterior to applying the unraveling parametrization as a function of time and the state
    Flist (function) : List of feedback operators as a function of time and state
    stateRho (array): density matrix state
    it (float): time to evaluate

    Returns:
    List of density matrices (array): resulting array of density matrices for each time step if last_point False, the last point of evolution otherwise
    """
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

# Add a pure Python version of the feedback evolution operator
def feedbackEvol_operator_py(drivingH, original_lindbladList, lindbladList, Flist, stateRho, it):
    """
    Pure Python version of the feedback evolution operator without Numba decorators
    
    Parameters:
    drivingH (function): The Hamiltonian operator as a function of time
    original_lindbladList (function) : List of Linblad operators prior to applying the unraveling parametrization as a function of time
    lindbladList (function) : List of Linblad operators posterior to applying the unraveling parametrization as a function of time and the state
    Flist (function) : List of feedback operators as a function of time and state
    stateRho (array): density matrix state
    it (float): time to evaluate

    Returns:
    array: Result of applying the feedback evolution operator
    """
    # Handle different function signatures for lindbladList
    try:
        L_it = lindbladList(it, stateRho)
    except TypeError:
        try:
            L_it = lindbladList(it)
        except TypeError:
            L_it = lindbladList

    # Handle different function signatures for original_lindbladList
    try:
        O_L_it = original_lindbladList(it, stateRho)
    except TypeError:
        try:
            O_L_it = original_lindbladList(it)
        except TypeError:
            O_L_it = original_lindbladList

    # Handle different function signatures for Flist
    try:
        F_it = Flist(it, stateRho)
    except TypeError:
        try:
            F_it = Flist(it)
        except TypeError:
            F_it = Flist

    # Calculate the Hamiltonian
    try:
        H_it = drivingH(it)
    except TypeError:
        H_it = drivingH

    stateRho = np.ascontiguousarray(stateRho)
    
    # Pure Python implementation of D_vec (Lindblad dissipator)
    def D_vec_py(operators, rho):
        result = np.zeros_like(rho, dtype=np.complex128)
        for A in operators:
            # Calculate the Lindblad dissipator: D[A](rho) = A rho A† - (1/2){A†A, rho}
            A_dag = np.conjugate(np.transpose(A))
            A_dag_A = np.dot(A_dag, A)
            
            # A rho A†
            term1 = np.dot(np.dot(A, rho), A_dag)
            
            # (1/2)(A†A rho + rho A†A)
            term2 = 0.5 * (np.dot(A_dag_A, rho) + np.dot(rho, A_dag_A))
            
            result += term1 - term2
        return result
    
    # Lindblad super op using our pure Python implementation
    D_c = D_vec_py(O_L_it, stateRho)
    D_f = D_vec_py(F_it, stateRho)

    # Stochastic contribution - Pure Python implementation of commutator
    comm_extra_term = np.zeros(np.shape(stateRho), dtype=np.complex128)
    for n_L, L in enumerate(L_it):
        L = np.ascontiguousarray(L)
        # drho_rhod = L·rho + rho·L†
        drho_rhod = np.dot(L, stateRho) + np.dot(stateRho, np.conjugate(np.transpose(L)))
        
        # Commutator: [F, X] = F·X - X·F
        F = F_it[n_L]
        comm = np.dot(F, drho_rhod) - np.dot(drho_rhod, F)
        
        comm_extra_term += comm

    # Calculate commutator [H, rho] = H·rho - rho·H
    H_rho = np.dot(H_it, stateRho)
    rho_H = np.dot(stateRho, H_it)
    H_comm = H_rho - rho_H

    return -1j*(H_comm + comm_extra_term) + D_c + D_f

def feedbackEvoladaptative_operator(System_obj, stateRho : np.ndarray, it : float) -> np.ndarray:
    """
    Gives the evaluation of the feedback equation differential operator for a given state and time for adapative parametrization
    
    Parameters:
    System_obj (System) : System class object where all the systems details are defined
    stateRho (array): density matrix state
    it (float): time to evaluate

    Returns:
    List of density matrices (array): resulting array of density matrices for each time step if last_point False, the last point of evolution otherwise
    """
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


@njit(fastmath=True, cache=True)
def custom_rungekutta_integrator_last_point(differential_operator, initialStateRho, timeList):
    """
    Performs the integration of the time-evolution of the differential_operator
    
    Parameters:
    differential_operator (function): the differential operator to integrate as a njitted function
    initialStateRho (array): Initial state
    timeList (array): Time list to integrate
    
    Returns:
    np.array([rho_list]) (array): resulting final density matrix
    """
    dt = timeList[1] - timeList[0]
    # Pre-allocate memory and avoid copies
    rho_list = np.ascontiguousarray(initialStateRho)
    
    for t_i in timeList[:-1]:
        a = differential_operator(rho_list, t_i)
        b = differential_operator(rho_list + 0.5*dt*a, t_i + 0.5*dt)
        c = differential_operator(rho_list + 0.5*dt*b, t_i + 0.5*dt)
        d = differential_operator(rho_list + dt*c, t_i + dt)
        # Update in-place, avoid intermediates
        rho_list += (dt/6.0)*(a + 2.0*b + 2.0*c + d)
    
    return np.array([rho_list])

@njit(fastmath=True, cache=True)
def custom_rungekutta_integrator_full_range(differential_operator, initialStateRho, timeList):
    """
    Performs the integration of the time-evolution of the differential_operator
    
    Parameters:
    differential_operator (function): the differential operator to integrate as a njitted function
    initialStateRho (array): Initial state
    timeList (array): Time list to integrate
    
    Returns:
    np.array([rho_list]) (array): resulting array of density matrices for each time step
    """
    dt = timeList[1] - timeList[0]
    time_steps = timeList.shape[0]
    dim1, dim2 = initialStateRho.shape
    
    # Pre-allocate with correct dimensions
    rho_list = np.zeros((time_steps, dim1, dim2), dtype=np.complex128)
    rho_list[0] = initialStateRho.copy()
    
    for n_ti in range(time_steps-1):
        t_i = timeList[n_ti]
        
        # Use temporary arrays to avoid creating new arrays in each step
        a = differential_operator(rho_list[n_ti], t_i)
        b = differential_operator(rho_list[n_ti] + 0.5*dt*a, t_i + 0.5*dt)
        c = differential_operator(rho_list[n_ti] + 0.5*dt*b, t_i + 0.5*dt)
        d = differential_operator(rho_list[n_ti] + dt*c, t_i + dt)
        
        # Update next step directly
        rho_list[n_ti+1] = rho_list[n_ti] + (dt/6.0)*(a + 2.0*b + 2.0*c + d)
    
    return rho_list
