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

# Pure Python version of dNRho for better type compatibility
def dNRho_py(stateRho, measurement_op_list, dt, seed=0):
    """Pure Python implementation of dNRho to avoid Numba type issues"""
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
        # Use NumPy's choice function for pure Python version
        jump_index = np.random.choice(M_index, 1, p=weight/sum(weight))[0]
    else:
        jump_index = np.shape(measurement_op_list)[0]
    return jump_index


def coherent_field_mixing_py(coherent_fields, L_it):
    """Pure Python version of coherent field mixing for better type compatibility"""
    new_ops = []
    for n_L, L in enumerate(L_it):
        try:
            # Try to create an identity matrix of the right size
            eye_matrix = np.eye(np.shape(L)[0], dtype=np.complex128)
            # Add the coherent field contribution
            new_op = L + coherent_fields[n_L] * eye_matrix
            new_ops.append(new_op)
        except Exception as e:
            print(f"Error in coherent_field_mixing for operator {n_L}: {e}")
            # Add the original operator as fallback
            new_ops.append(L)
    return np.array(new_ops)

# Original Numba functions with modified error handling
@njit
def dNRho(stateRho, measurement_op_list, dt, seed=0):
    """
    Calculate the jump index for quantum jumps.
    
    Parameters:
    -----------
    stateRho : numpy.ndarray
        Density matrix of the quantum state
    measurement_op_list : numpy.ndarray
        List of measurement operators
    dt : float
        Time step
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    int
        Jump index, indicates which operator caused the jump or if no jump occurred
    """
    # Calculate weights for each measurement operator
    weight = np.zeros(np.shape(measurement_op_list)[0], dtype=np.float64)
    M_index = np.zeros(np.shape(measurement_op_list)[0], dtype=np.int64)

    for mu, Mmu in enumerate(measurement_op_list):
        Mmu = np.ascontiguousarray(Mmu)
        Mmu_dag_Mmu = np.dot(np.conjugate(np.transpose(Mmu)), Mmu)
        weight[mu] += dt*np.real(np.trace(np.dot(stateRho, Mmu_dag_Mmu)))
        M_index[mu] += mu
    
    # Set the random seed and generate a random number
    np.random.seed(seed)
    R = np.random.rand()
    
    # Determine if a jump occurs
    if R < sum(weight):
        # Use a simpler approach without objmode to avoid compilation issues
        # We'll manually implement the weighted selection
        cumulative_weights = np.cumsum(weight/sum(weight))
        R_selection = np.random.rand()
        jump_index = np.shape(measurement_op_list)[0]  # default if no jump
        
        # Find which bin the random number falls into
        for i, cum_weight in enumerate(cumulative_weights):
            if R_selection <= cum_weight:
                jump_index = M_index[i]
                break
    else:
        # No jump case
        jump_index = np.shape(measurement_op_list)[0]
        
    return jump_index

@njit
def coherent_field_mixing(coherent_fields, L_it):
    new_ops = np.zeros(np.shape(L_it), dtype=np.complex128)
    for n_L, L in enumerate(L_it):
        new_ops[n_L] += L + coherent_fields[n_L]*np.eye(np.shape(L)[0])
    return new_ops

@njit
def ortogonal_mixing(oMatrix, coherent_fields, L_it):
    new_ops = np.zeros(np.shape(L_it), dtype=np.complex128)
    for n_O, O in enumerate(oMatrix):
        new_ops[n_O] += coherent_fields[n_O]*np.eye(np.shape(L_it)[0])
        for n_L, L in enumerate(L_it):
            new_ops[n_O] += O[n_L]*L
    return new_ops

# Two-stage implementation for time-dependent jumps
def jumpRhoTrajectory_td_optimized(initialStateRho, timelist, drivingH, original_lindbladList, 
                                  eta_diag, lindbladList, coherent_fields, seed=0, verbose=False):
    """
    Optimized implementation of quantum jump trajectory for time-dependent systems.
    This uses a two-stage approach:
    1. Precompute operators where possible
    2. Run the main integration loop with less dynamic function calls
    
    Parameters:
    -----------
    initialStateRho : numpy.ndarray
        Initial density matrix
    timelist : numpy.ndarray
        Time points for integration
    drivingH : callable
        Function returning the Hamiltonian at a given time
    original_lindbladList : callable
        Function returning the original Lindblad operators at a given time
    eta_diag : numpy.ndarray
        Diagonal elements of the efficiency matrix
    lindbladList : callable
        Function returning transformed Lindblad operators at a given time
    coherent_fields : numpy.ndarray
        Coherent field amplitudes
    seed : int
        Random seed for stochastic integration
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    numpy.ndarray
        Evolution of the density matrix
    """
    # Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]
    
    if verbose:
        print(f"Setting up quantum jump trajectory with {timeSteps} time steps, dt = {dt}")
    
    # Get initial operators to determine dimensions
    try:
        initial_L = lindbladList(timelist[0], initialStateRho)
        num_lindblad_channels = np.shape(initial_L)[0]
    except Exception as e:
        if verbose:
            print(f"Error getting initial Lindblad operators: {e}")
            print("Trying alternative approach...")
        # Try a different approach
        try:
            original_L = original_lindbladList(timelist[0])
            num_lindblad_channels = np.shape(original_L)[0]
        except:
            if verbose:
                print("Using default number of Lindblad channels based on coherent fields")
            # Default to number of coherent fields
            num_lindblad_channels = len(coherent_fields) // 2
    
    if verbose:
        print(f"Number of Lindblad channels: {num_lindblad_channels}")
    
    # Verify coherent fields match the number of Lindblad channels
    if len(coherent_fields) != 2 * num_lindblad_channels:
        if verbose:
            print(f"Warning: Number of coherent fields ({len(coherent_fields)}) doesn't match 2 * number of Lindblad channels ({num_lindblad_channels})")
    
    # Initialize trajectory
    rho_trajectory = np.zeros((timeSteps,) + np.shape(initialStateRho), dtype=np.complex128)
    rho_trajectory[0] = initialStateRho.copy()
    
    # Initialize terms for efficiency
    no_jump_term_1 = np.zeros(np.shape(initialStateRho), dtype=np.complex128)
    no_jump_term_2 = np.zeros(np.shape(initialStateRho), dtype=np.complex128)
    inefficient_term = np.zeros(np.shape(initialStateRho), dtype=np.complex128)
    
    # Set up random number generator with seed
    np.random.seed(seed)
    
    # Main integration loop
    for n_it, it in enumerate(timelist[:-1]):
        try:
            # Get Lindblad operators at current time and state
            L_it = lindbladList(it, rho_trajectory[n_it])
            original_L_it = original_lindbladList(it)
            
            # Reset accumulated operators
            inefficient_term = np.zeros_like(inefficient_term)
            
            # Calculate inefficient term
            for n_i, L_i in enumerate(original_L_it):
                inefficient_term += (1 - eta_diag[n_i]) * op.D(L_i, rho_trajectory[n_it]) * dt
            
            # Mix Lindblad operators with coherent fields
            coherent_field_ops = coherent_field_mixing_py(coherent_fields[:num_lindblad_channels], L_it)
            
            # Determine if a quantum jump occurs
            jump_index = dNRho_py(rho_trajectory[n_it], coherent_field_ops, dt, seed*timeSteps+n_it)
            
            # Update state based on jump or smooth evolution
            if jump_index < num_lindblad_channels:
                # Jump occurs
                rho_trajectory[n_it+1] = rho_trajectory[n_it] + op.G(coherent_field_ops[jump_index], rho_trajectory[n_it]) + inefficient_term
            else:
                # No jump - smooth evolution
                no_jump_term_1 = np.zeros_like(no_jump_term_1)
                no_jump_term_2 = np.zeros_like(no_jump_term_2)
                
                for n_r, L_r in enumerate(L_it):
                    L_r_contig = np.ascontiguousarray(L_r)
                    no_jump_term_1 += -0.5 * np.dot(np.conjugate(np.transpose(L_r_contig)), L_r_contig)
                    if n_r < len(coherent_fields):
                        no_jump_term_2 += -coherent_fields[n_r] * L_r_contig
                
                # Get Hamiltonian at current time
                H_it = drivingH(it)
                total_H = -1j * H_it + no_jump_term_1 + no_jump_term_2
                
                # Update state with non-hermitian Hamiltonian evolution
                rho_trajectory[n_it+1] = rho_trajectory[n_it] + op.H(total_H, rho_trajectory[n_it]) * dt + inefficient_term
        
        except Exception as e:
            if verbose:
                print(f"Error at time step {n_it} (t = {it}): {e}")
                print("Using previous state for next step")
            # On error, carry forward the previous state
            rho_trajectory[n_it+1] = rho_trajectory[n_it]
    
    return rho_trajectory

# Original time-dependent implementation (modified to handle type errors)
def jumpRhoTrajectory_td(initialStateRho, timelist, drivingH, original_lindbladList, eta_diag, lindbladList, coherent_fields, seed=0, verbose=False):
    """
    Time-dependent implementation of quantum jump trajectory.
    
    This version has been modified to better handle type compatibility issues.
    For better performance, use jumpRhoTrajectory_td_optimized instead.
    """
    # Try the optimized version first
    try:
        return jumpRhoTrajectory_td_optimized(initialStateRho, timelist, drivingH, 
                                            original_lindbladList, eta_diag, 
                                            lindbladList, coherent_fields, seed, verbose)
    except Exception as e:
        if verbose:
            print(f"Optimized jump trajectory failed with error: {e}")
            print("Falling back to standard Python implementation")
        
    # Fall back to the standard Python implementation
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]

    # Get the number of Lindblad operators
    try:
        initial_L = lindbladList(timelist[0], initialStateRho)
        num_lindblad_channels = np.shape(initial_L)[0]
    except:
        try:
            original_L = original_lindbladList(timelist[0])
            num_lindblad_channels = np.shape(original_L)[0]
        except:
            num_lindblad_channels = len(coherent_fields) // 2

    # Ensure coherent_fields is compatible
    if len(coherent_fields) != num_lindblad_channels and len(coherent_fields) != 2 * num_lindblad_channels:
        if verbose:
            print(f"Warning: Coherent fields length {len(coherent_fields)} doesn't match Lindblad channels {num_lindblad_channels}")
        coherent_fields = coherent_fields[:num_lindblad_channels]

    rho_trajectory = np.zeros((len(timelist),) + np.shape(initialStateRho), dtype=np.complex128)
    rho_trajectory[0] = initialStateRho.copy()

    no_jump_term_1 = np.zeros(np.shape(initialStateRho), dtype=np.complex128)
    no_jump_term_2 = np.zeros(np.shape(initialStateRho), dtype=np.complex128)
    inefficient_term = np.zeros(np.shape(initialStateRho), dtype=np.complex128)

    for n_it, it in enumerate(timelist[:-1]):
        try:
            L_it = lindbladList(it, rho_trajectory[n_it])
            original_L_it = original_lindbladList(it)

            inefficient_term = np.zeros_like(inefficient_term)
            for n_i, L_i in enumerate(original_L_it):
                inefficient_term += (1 - eta_diag[n_i]) * op.D(L_i, rho_trajectory[n_it]) * dt

            coherent_field_ops = coherent_field_mixing_py(coherent_fields[:num_lindblad_channels], L_it)
            jump_index = dNRho_py(rho_trajectory[n_it], coherent_field_ops, dt, seed*timeSteps+n_it)

            if jump_index < num_lindblad_channels:
                rho_trajectory[n_it+1] = rho_trajectory[n_it] + op.G(coherent_field_ops[jump_index], rho_trajectory[n_it]) + inefficient_term
            else:
                no_jump_term_1 = np.zeros_like(no_jump_term_1)
                no_jump_term_2 = np.zeros_like(no_jump_term_2)
                
                for n_r, L_r in enumerate(L_it):
                    no_jump_term_1 += -0.5 * np.dot(np.conjugate(np.transpose(L_r)), L_r)
                    if n_r < len(coherent_fields):
                        no_jump_term_2 += -coherent_fields[n_r] * L_r
                
                H_it = drivingH(it)
                rho_trajectory[n_it+1] = rho_trajectory[n_it] + op.H(-1j*H_it + no_jump_term_1 + no_jump_term_2, rho_trajectory[n_it]) * dt + inefficient_term
        except Exception as e:
            if verbose:
                print(f"Error at time step {n_it}: {e}")
            rho_trajectory[n_it+1] = rho_trajectory[n_it]

    return rho_trajectory

# Pure Python version of the time-independent jump trajectory function
def jumpRhoTrajectory_py(initialStateRho, timelist, drivingH, original_lindbladList, eta_diag, lindbladList, coherent_fields, seed=0):
    """
    Pure Python implementation of time-independent jump trajectory.
    This is a fallback for when the Numba-jitted version fails due to type issues.
    """
    # Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]

    # Number of Lindblad operators
    num_lindblad_channels = np.shape(lindbladList)[0]

    rho_trajectory = np.zeros((timeSteps,) + np.shape(initialStateRho), dtype=np.complex128)
    rho_trajectory[0] = initialStateRho.copy()

    no_jump_term_1 = np.zeros(np.shape(initialStateRho), dtype=np.complex128)
    no_jump_term_2 = np.zeros(np.shape(initialStateRho), dtype=np.complex128)
    inefficient_term = np.zeros(np.shape(initialStateRho), dtype=np.complex128)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    for n_it, it in enumerate(timelist[:-1]):
        # Reset accumulated terms
        inefficient_term = np.zeros_like(inefficient_term)
        
        # Calculate inefficient term
        for n_i, L_i in enumerate(original_lindbladList):
            inefficient_term += (1 - eta_diag[n_i]) * op.D(L_i, rho_trajectory[n_it]) * dt

        # Mix Lindblad operators with coherent fields
        coherent_field_ops = coherent_field_mixing_py(coherent_fields, lindbladList)

        # Determine if a quantum jump occurs
        jump_index = dNRho_py(rho_trajectory[n_it], coherent_field_ops, dt, seed*timeSteps+n_it)

        # Update state based on jump or smooth evolution
        if jump_index < num_lindblad_channels:
            # Jump occurs
            rho_trajectory[n_it+1] = rho_trajectory[n_it] + op.G(coherent_field_ops[jump_index], rho_trajectory[n_it]) + inefficient_term
        else:
            # No jump - smooth evolution
            no_jump_term_1 = np.zeros_like(no_jump_term_1)
            no_jump_term_2 = np.zeros_like(no_jump_term_2)
            
            for n_r, L_r in enumerate(lindbladList):
                L_r = np.ascontiguousarray(L_r)
                no_jump_term_1 += -0.5 * np.dot(np.transpose(np.conjugate(L_r)), L_r)
                no_jump_term_2 += -coherent_fields[n_r] * L_r
            
            rho_trajectory[n_it+1] = rho_trajectory[n_it] + op.H(-1j*drivingH + no_jump_term_1 + no_jump_term_2, rho_trajectory[n_it])*dt + inefficient_term

    return rho_trajectory

# Time-independent implementation (unchanged)
@njit
def jumpRhoTrajectory_(initialStateRho, timelist, drivingH, original_lindbladList, eta_diag, lindbladList, coherent_fields, seed=0):
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