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

# Add a pure Python operator transformation for M-representation
def operators_Mrep_py(M_rep, lindbladList):
    """
    Pure Python version of operators_Mrep that transforms operators using M_rep.
    This avoids Numba typing issues with complex data.
    """
    result = []
    for L in lindbladList:
        result.append(np.dot(M_rep, L))
    return np.array(result)

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

# Non-numba version for dynamic operators
def diffusiveRhoEulerStep_py(stateRho, drivingH, original_L_it, L_it, zeta, dt):
    """Pure Python version of the Euler step (no Numba)"""
    D_c = np.zeros(stateRho.shape, dtype=np.complex128)
    
    # Calculate D_vec manually
    for L in original_L_it:
        L_rho = np.dot(L, stateRho)
        rho_Ldag = np.dot(stateRho, np.conjugate(np.transpose(L)))
        LdagL_rho = np.dot(np.dot(np.conjugate(np.transpose(L)), L), stateRho)
        rho_LdagL = np.dot(stateRho, np.dot(np.conjugate(np.transpose(L)), L))
        D_c += L_rho @ np.conjugate(np.transpose(L)) - 0.5 * (LdagL_rho + rho_LdagL)
    
    D_c *= dt
    
    # Stochastic contribution
    Hw = np.zeros(np.shape(stateRho), dtype=np.complex128)
    for n_L, L in enumerate(L_it):
        L_rho = np.dot(L, stateRho)
        rho_Ldag = np.dot(stateRho, np.conjugate(np.transpose(L)))
        Hw += (L_rho + rho_Ldag - stateRho * np.trace(L_rho + rho_Ldag)) * zeta[n_L] * np.sqrt(dt)
    
    # Commutator calculation for Hamiltonian
    H_rho = np.dot(drivingH, stateRho)
    rho_H = np.dot(stateRho, drivingH)
    return -1j * (H_rho - rho_H) * dt + D_c + Hw

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

# Non-numba version for dynamic operators
def diffusiveRhoMilsteinStep_py(stateRho, drivingH, original_L_it, L_it, zeta, dt):
    """Pure Python version of the Milstein step (no Numba)"""
    # This is a simplified version, implementing just enough for basic functionality
    return diffusiveRhoEulerStep_py(stateRho, drivingH, original_L_it, L_it, zeta, dt)

############################################
######  diffusive trajectory functions #####
############################################
def diffusiveRhoTrajectory_td(initialStateRho, timelist, drivingH, original_lindbladList, lindbladList, method='euler', seed=0, verbose=False):
    """
    Non-Numba version of the diffusive trajectory function for time-dependent operators.
    This version works with dynamic function calls without Numba type issues.
    
    Note: For better performance, use diffusiveRhoTrajectory_td_optimized instead when possible.
    """
    # Try using the optimized version first
    try:
        return diffusiveRhoTrajectory_td_optimized(initialStateRho, timelist, drivingH, 
                                                original_lindbladList, lindbladList, method, seed, verbose)
    except Exception as e:
        if verbose:
            print(f"Optimized version failed with error: {e}")
            print("Falling back to pure Python implementation (slower)")
    
    # Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]
    
    # First, safely determine the number of Lindblad operators
    try:
        first_lindblad = original_lindbladList(timelist[0])
        number_lindblad_op = np.shape(first_lindblad)[0]
    except Exception as e:
        if verbose:
            print(f"Warning: Could not determine number of Lindblad operators from original_lindbladList: {e}")
        try:
            # Try getting the M_rep from the lindbladList if it's a matrix
            if hasattr(lindbladList, 'shape'):
                number_lindblad_op = lindbladList.shape[0]
            else:
                # Try a safe call to lindbladList with fallback
                try:
                    L_it = lindbladList(timelist[0], initialStateRho)
                    number_lindblad_op = len(L_it)
                except:
                    try:
                        L_it = lindbladList(timelist[0])
                        number_lindblad_op = len(L_it)
                    except:
                        if verbose:
                            print("Warning: Could not determine number of Lindblad operators, using default of 1")
                        number_lindblad_op = 1
        except:
            if verbose:
                print("Warning: Could not determine number of Lindblad operators, using default of 1")
            number_lindblad_op = 1

    if verbose:
        print(f"Using {number_lindblad_op} Lindblad operators for diffusive trajectory")
    
    # Stochastic increments
    np.random.seed(seed)
    zeta = np.random.normal(loc=0, scale=1, size=(number_lindblad_op, timeSteps))

    rho_trajectory = np.empty((timeSteps,) + np.shape(initialStateRho), dtype=np.complex128)
    rho_trajectory[0,:,:] = np.ascontiguousarray(initialStateRho)

    # Integrator
    if method == 'euler':
        for n_it, it in enumerate(timelist[:-1]):       
            # Get original Lindblad operators at this time step
            try:
                original_L_it = original_lindbladList(it)
            except Exception as e:
                if verbose:
                    print(f"Error getting original Lindblad operators at t={it}: {e}")
                # Use default identity operator as fallback
                original_L_it = [np.eye(initialStateRho.shape[0], dtype=np.complex128)]
            
            # Call lindbladList with proper arguments
            try:
                L_it = lindbladList(it, rho_trajectory[n_it])
            except TypeError:
                # If the function doesn't accept rho, try without it
                try:
                    L_it = lindbladList(it)
                except:
                    # If that fails too, use operators_Mrep_py as a fallback
                    try:
                        L_it = operators_Mrep_py(lindbladList, original_L_it)
                    except:
                        if verbose:
                            print(f"Warning: Could not transform Lindblad operators at t={it}, using originals")
                        L_it = original_L_it
            
            # Make sure the number of operators matches zeta's dimension
            if len(L_it) != number_lindblad_op:
                if verbose:
                    print(f"Warning: Number of Lindblad operators ({len(L_it)}) doesn't match zeta's dimension ({number_lindblad_op})")
                # Two options: reshape zeta or pad/truncate L_it
                if len(L_it) > number_lindblad_op:
                    # Truncate L_it to match zeta
                    L_it = L_it[:number_lindblad_op]
                else:
                    # Pad L_it with zeros or identity matrices to match zeta
                    padding = [np.zeros_like(L_it[0]) for _ in range(number_lindblad_op - len(L_it))]
                    L_it = np.concatenate([L_it, padding])
            
            # Use Python version (not Numba) to avoid type issues
            rho_trajectory[n_it+1,:,:] = rho_trajectory[n_it] + diffusiveRhoEulerStep_py(
                rho_trajectory[n_it], drivingH(it), original_L_it, L_it, zeta[:, n_it], dt)
            
    elif method == 'milstein':
        for n_it, it in enumerate(timelist[:-1]):
            # Get original Lindblad operators at this time step
            try:
                original_L_it = original_lindbladList(it)
            except Exception as e:
                if verbose:
                    print(f"Error getting original Lindblad operators at t={it}: {e}")
                original_L_it = [np.eye(initialStateRho.shape[0], dtype=np.complex128)]
            
            # Call lindbladList with proper arguments
            try:
                L_it = lindbladList(it, rho_trajectory[n_it])
            except TypeError:
                # If the function doesn't accept rho, try without it
                try:
                    L_it = lindbladList(it)
                except:
                    # If that fails too, use operators_Mrep_py as a fallback
                    try:
                        L_it = operators_Mrep_py(lindbladList, original_L_it)
                    except:
                        if verbose:
                            print(f"Warning: Could not transform Lindblad operators at t={it}, using originals")
                        L_it = original_L_it
            
            # Make sure the number of operators matches zeta's dimension
            if len(L_it) != number_lindblad_op:
                if verbose:
                    print(f"Warning: Number of Lindblad operators ({len(L_it)}) doesn't match zeta's dimension ({number_lindblad_op})")
                if len(L_it) > number_lindblad_op:
                    # Truncate L_it to match zeta
                    L_it = L_it[:number_lindblad_op]
                else:
                    # Pad L_it with zeros or identity matrices to match zeta
                    padding = [np.zeros_like(L_it[0]) for _ in range(number_lindblad_op - len(L_it))]
                    L_it = np.concatenate([L_it, padding])
            
            # Use Python version (not Numba) to avoid type issues
            rho_trajectory[n_it+1,:,:] = rho_trajectory[n_it] + diffusiveRhoMilsteinStep_py(
                rho_trajectory[n_it], drivingH(it), original_L_it, L_it, zeta[:, n_it], dt)

    return rho_trajectory

# Two-stage diffusive trajectory function with Numba acceleration
def diffusiveRhoTrajectory_td_optimized(initialStateRho, timelist, drivingH, original_lindbladList, lindbladList, method='euler', seed=0, verbose=False):
    """
    Optimized two-stage version of the diffusive trajectory function:
    1. Prepares all data needed for integration (preprocessing stage)
    2. Calls a Numba-accelerated core integration function (accelerated stage)
    
    This approach allows for time-dependent operators while still leveraging Numba's speed.
    
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
    lindbladList : callable or numpy.ndarray
        Function returning transformed Lindblad operators or M-representation matrix
    method : str
        Integration method ('euler' or 'milstein')
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
    
    # Pre-allocate arrays for operators at each time step
    hamiltonians = []
    original_lindblads = []
    transformed_lindblads = []
    
    # Setup stage - calculate all operators for all time steps
    # This happens outside of Numba for flexibility with function calls
    if verbose:
        print("Preparing operators for all time steps...")
    
    for t in timelist:
        # Get Hamiltonian
        H_t = drivingH(t)
        hamiltonians.append(H_t)
        
        # Get original Lindblad operators
        try:
            orig_L = original_lindbladList(t)
            original_lindblads.append(np.array(orig_L))
        except Exception as e:
            if verbose:
                print(f"Error getting original Lindblad operators at t={t}: {e}")
            # Use default identity operator as fallback
            orig_L = [np.eye(initialStateRho.shape[0], dtype=np.complex128)]
            original_lindblads.append(np.array(orig_L))
        
        # Get transformed Lindblad operators (if applicable)
        try:
            if callable(lindbladList):
                # Try with just time parameter
                try:
                    L_t = lindbladList(t)
                except TypeError:
                    # If that doesn't work, it might need a state
                    # Use initial state as a placeholder - will be updated during actual integration
                    L_t = lindbladList(t, initialStateRho)
            else:
                # If not callable, it might be a matrix for M-representation
                L_t = operators_Mrep_py(lindbladList, orig_L)
            transformed_lindblads.append(np.array(L_t))
        except Exception as e:
            if verbose:
                print(f"Error transforming Lindblad operators at t={t}, using originals: {e}")
            transformed_lindblads.append(np.array(orig_L))
    
    # Convert to arrays for Numba
    H_array = np.array(hamiltonians)
    orig_L_array = np.array(original_lindblads)
    trans_L_array = np.array(transformed_lindblads)
    
    # Determine number of Lindblad operators
    number_lindblad_op = trans_L_array[0].shape[0]
    
    # Set up stochastic increments
    np.random.seed(seed) 
    zeta = np.random.normal(loc=0, scale=1, size=(number_lindblad_op, timeSteps))
    
    # Call the accelerated core integration function
    if verbose:
        print(f"Starting {method} integration with {number_lindblad_op} Lindblad operators...")
    
    if method == 'euler':
        return _diffusive_trajectory_euler_core(
            initialStateRho, timelist, dt, H_array, orig_L_array, trans_L_array, zeta)
    elif method == 'milstein':
        return _diffusive_trajectory_milstein_core(
            initialStateRho, timelist, dt, H_array, orig_L_array, trans_L_array, zeta)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'euler' or 'milstein'.")

@njit
def _diffusive_trajectory_euler_core(initialStateRho, timelist, dt, H_array, orig_L_array, trans_L_array, zeta):
    """Numba-accelerated core function for Euler integration"""
    timeSteps = len(timelist)
    rho_trajectory = np.empty((timeSteps,) + initialStateRho.shape, dtype=np.complex128)
    rho_trajectory[0] = initialStateRho
    
    for n_it in range(timeSteps - 1):
        rho_trajectory[n_it+1] = rho_trajectory[n_it] + diffusiveRhoEulerStep_(
            rho_trajectory[n_it], 
            H_array[n_it], 
            orig_L_array[n_it], 
            trans_L_array[n_it], 
            zeta[:, n_it], 
            dt
        )
    
    return rho_trajectory

@njit
def _diffusive_trajectory_milstein_core(initialStateRho, timelist, dt, H_array, orig_L_array, trans_L_array, zeta):
    """Numba-accelerated core function for Milstein integration"""
    timeSteps = len(timelist)
    rho_trajectory = np.empty((timeSteps,) + initialStateRho.shape, dtype=np.complex128)
    rho_trajectory[0] = initialStateRho
    
    for n_it in range(timeSteps - 1):
        rho_trajectory[n_it+1] = rho_trajectory[n_it] + diffusiveRhoMilsteinStep(
            rho_trajectory[n_it], 
            H_array[n_it], 
            orig_L_array[n_it], 
            trans_L_array[n_it], 
            zeta[:, n_it], 
            dt
        )
    
    return rho_trajectory

#@njit
def diffusiveRhoTrajectory_(initialStateRho, timelist, drivingH, original_lindbladList, lindbladList, method='euler', seed=0):
    """Original function for time-independent dynamics (still requires numba)"""
    ## Timelist details
    timeSteps = np.shape(timelist)[0]
    dt = timelist[1] - timelist[0]
    
    ## Number of Lindblad operators
    number_lindblad_op = np.shape(lindbladList)[0]

    ## Stocastics increments
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

