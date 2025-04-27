'''
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
'''
import numpy as np
from numba import njit, complex128, float64, prange

## Pauli matrices - pre-compiled constants for faster access
_SIGMAX = np.ascontiguousarray(np.array([[0,1],[1,0]], dtype=np.complex128))
_SIGMAY = np.ascontiguousarray(np.array([[0,-1j],[1j,0]], dtype=np.complex128))
_SIGMAZ = np.ascontiguousarray(np.array([[1,0],[0,-1]], dtype=np.complex128))

## Ladder operators (pre-compiled)
_SIGMAP = np.ascontiguousarray(0.5*(_SIGMAX + 1j*_SIGMAY))
_SIGMAM = np.ascontiguousarray(0.5*(_SIGMAX - 1j*_SIGMAY))

# For backward compatibility
sigmax = _SIGMAX
sigmay = _SIGMAY
sigmaz = _SIGMAZ
sigmap = _SIGMAP
sigmam = _SIGMAM

@njit(fastmath=True, cache=True)
def Com(a, b):
    """Commutator [a, b] = ab - ba"""
    return np.dot(a, b) - np.dot(b, a)

@njit(fastmath=True, cache=True)
def AntiCom(a, b):
    """Anti-commutator {a, b} = ab + ba"""
    return np.dot(a, b) + np.dot(b, a)

@njit(fastmath=True, cache=True)
def D(A, B):
    """Lindblad dissipator D[A](B) = ABA† - (1/2){A†A, B}"""
    AT = np.conjugate(np.transpose(A))
    ATA = np.dot(AT, A)
    return np.dot(A, np.dot(B, AT)) - 0.5 * (np.dot(ATA, B) + np.dot(B, ATA))

@njit(fastmath=True, cache=True)
def D_vec(A, B):
    """Vectorized Lindblad dissipator for multiple operators"""
    D_ = np.zeros(B.shape, dtype=np.complex128)
    for A_i in A:
        D_ += D(A_i, B)
    return D_

@njit(fastmath=True, cache=True)
def H_vec(A, B):
    """Vectorized H superoperator for multiple operators"""
    n_ops = A.shape[0]
    dim1, dim2 = B.shape
    H = np.zeros((n_ops, dim1, dim2), dtype=np.complex128)
    
    for n_A in range(n_ops):
        A_i = A[n_A]
        A_dag = np.conjugate(np.transpose(A_i))
        # Compute once and reuse
        AB = np.dot(A_i, B)
        BA_dag = np.dot(B, A_dag)
        Aux = AB + BA_dag
        tr_Aux = np.trace(Aux)
        H[n_A] = Aux - tr_Aux * B
    
    return H

@njit(fastmath=True, cache=True)
def H(A, B):
    """H superoperator H[A](B) = AB + BA† - Tr(AB + BA†)B"""
    A_dag = np.conjugate(np.transpose(A))
    # Compute once and reuse
    AB = np.dot(A, B)
    BA_dag = np.dot(B, A_dag)
    Aux = AB + BA_dag
    return Aux - np.trace(Aux) * B

@njit(fastmath=True, cache=True)
def G(A, B):
    """G superoperator"""
    A_dag = np.conjugate(np.transpose(A))
    ABA_dag = np.dot(np.dot(A, B), A_dag)
    tr_ABA_dag = np.trace(ABA_dag)
    return (1.0 / tr_ABA_dag) * ABA_dag - B

@njit(fastmath=True, cache=True)
def tensordot_loop(number_list, matrix_list):
    """Computes a tensor dot product efficiently in a loop"""
    shape = matrix_list.shape[1:]
    tmp = np.zeros(shape, dtype=np.complex128)
    
    # Use direct indexing for better performance
    for n_number in range(len(number_list)):
        tmp += number_list[n_number] * matrix_list[n_number]
    
    return tmp

@njit(fastmath=True, cache=True)
def operators_Mrep(mMatrix, c):
    """Transform operators using M representation matrix"""
    # Number of associated operators
    transformed_num_op = mMatrix.shape[1]
    dim1, dim2 = c.shape[1], c.shape[2]
    new_ops = np.zeros((transformed_num_op, dim1, dim2), dtype=np.complex128)
    
    # Pre-conjugate mMatrix for better performance
    mMatrix_conj = np.conjugate(mMatrix)
    
    for i in range(transformed_num_op):
        new_ops[i] = tensordot_loop(mMatrix_conj[:,i], c)
    
    return new_ops

@njit(fastmath=True, cache=True)
def sqrt_jit(M):
    """Efficient matrix square root using eigendecomposition"""
    # Computing diagonalization
    evalues, evectors = np.linalg.eig(M)
    
    # Take square root of eigenvalues, avoiding potential numerical errors
    sqrt_evalues = np.sqrt(np.abs(evalues))
    
    # Direct matrix multiplication rather than multiple steps
    sqrt_matrix = evectors @ np.diag(sqrt_evalues) @ np.linalg.inv(evectors)
    
    return sqrt_matrix

@njit(fastmath=True, cache=True)
def H_DH(CA, CB, rho):
    """Combined H and D superoperator for improved performance"""
    # Precompute common terms
    CA_dag = np.conjugate(np.transpose(CA))
    CB_dag = np.conjugate(np.transpose(CB))
    
    # Compute products once
    CB_CA = np.dot(CB, CA)
    CB_CA_rho = np.dot(CB_CA, rho)
    CB_rho = np.dot(CB, rho)
    CA_rho = np.dot(CA, rho)
    rho_CA_dag = np.dot(rho, CA_dag)
    rho_CB_dag = np.dot(rho, CB_dag)
    
    # Compute combined terms
    A1 = CB_CA_rho + np.dot(CB_rho, CA_dag) + np.dot(CA_rho, CB_dag) + np.dot(rho_CA_dag, CB_dag)
    A2 = CA_rho + rho_CA_dag
    A3 = CB_rho + rho_CB_dag
    
    # Compute traces
    tr_A1 = np.trace(A1)
    tr_A2 = np.trace(A2)
    tr_A3 = np.trace(A3)
    
    return A1 - tr_A2*A3 - tr_A3*A2 + 2*tr_A2*tr_A3*rho - rho*tr_A1

@njit(fastmath=True, cache=True)
def fidelity(rho1, rho2):
    """Compute quantum fidelity between two density matrices"""
    # Compute sqrt(rho1) once
    srho1 = sqrt_jit(rho1)
    
    # Compute the chain of matrix multiplications efficiently
    inner = np.dot(srho1, np.dot(rho2, srho1))
    
    # Return only the real part of the trace
    return np.real(np.trace(sqrt_jit(inner)))

@njit(parallel=True, fastmath=True, cache=True)
def batch_operator_application(ops, states):
    """Apply operators to multiple states in parallel"""
    n_ops, n_states = ops.shape[0], states.shape[0]
    dim1, dim2 = states.shape[1], states.shape[2]
    results = np.zeros((n_ops, n_states, dim1, dim2), dtype=np.complex128)
    
    for i in prange(n_ops):
        op = ops[i]
        for j in range(n_states):
            results[i, j] = D(op, states[j])
    
    return results