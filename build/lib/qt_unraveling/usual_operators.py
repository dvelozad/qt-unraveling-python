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

## Pauli matrices
sigmax = np.array([[0,1],[1,0]], dtype = np.complex128)
sigmay = np.array([[0,-1j],[1j,0]], dtype = np.complex128)
sigmaz = np.array([[1,0],[0,-1]], dtype = np.complex128)

## Lader operators
sigmap = 0.5*(sigmax + 1j*sigmay)
sigmam = 0.5*(sigmax - 1j*sigmay)


## Commutator
@njit(complex128[:,:](complex128[:,:], complex128[:,:]))
def Com(a,b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return np.dot(a,b) - np.dot(b,a)

## Anticommutator
@njit(complex128[:,:](complex128[:,:], complex128[:,:]))
def AntiCom(a,b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return np.dot(a,b) + np.dot(b,a)

## Dissipator superoperator in the Lindblad equation
@njit(complex128[:,:](complex128[:,:], complex128[:,:]))
def D(A, B):
    A, B = np.ascontiguousarray(A), np.ascontiguousarray(B)
    AT = np.transpose(np.conjugate(A))
    return np.dot(A, np.dot(B,AT)) - 0.5*AntiCom(np.dot(AT,A),B)

## Dissipator superoperator for multiple Lindblad operators
@njit(complex128[:,:](complex128[:,:,:], complex128[:,:]))
def D_vec(A, B):
    A, B = np.ascontiguousarray(A), np.ascontiguousarray(B)
    D_ = np.zeros(np.shape(B), dtype=np.complex128)
    for A_i in A:
        D_ += D(A_i, B)
    return D_

@njit(complex128[:,:,:](complex128[:,:,:], complex128[:,:]))
def H_vec(A, B):
    A, B = np.ascontiguousarray(A), np.ascontiguousarray(B)
    H = np.zeros((np.shape(A)[0], np.shape(B)[0], np.shape(B)[1]), dtype=np.complex128)
    for n_A, A_i in enumerate(A):
        A_i = np.ascontiguousarray(A_i)
        Aux = np.dot(A_i,B) + np.dot(B,np.transpose(np.conjugate(A_i)))
        H[n_A] += Aux - np.trace(Aux)*B
    return H

## Caligraphic H superoperator for the diffusive unraveling
@njit(complex128[:,:](complex128[:,:], complex128[:,:]))
def H(A, B):
    A, B = np.ascontiguousarray(A), np.ascontiguousarray(B)
    Aux = np.dot(A,B) + np.dot(B,np.transpose(np.conjugate(A)))
    return Aux - np.trace(Aux)*B

## G superoperator, from representing the SME as in Wiseman's book
@njit(complex128[:,:](complex128[:,:], complex128[:,:]))
def G(A, B):
    A, B = np.ascontiguousarray(A), np.ascontiguousarray(B)
    Aux = np.dot(np.dot(A,B), np.transpose(np.conjugate(A)))
    return (1/np.trace(Aux))*Aux - B


@njit(complex128[:,:](complex128[:], complex128[:,:,:]))
def tensordot_loop(number_list, matrix_list):
    tmp = np.zeros(np.shape(matrix_list)[1:], dtype=np.complex128)
    for n_number, number in enumerate(number_list):
        tmp += number*matrix_list[n_number] 
    return np.ascontiguousarray(tmp)

@njit(complex128[:,:,:](complex128[:,:], complex128[:,:,:]))
def operators_Mrep(mMatrix, c):
    ## Number of associated operators
    transformed_num_op = np.shape(mMatrix)[1]
    new_ops = np.zeros((transformed_num_op, np.shape(c)[1], np.shape(c)[2]), dtype=np.complex128)
    for i in range(transformed_num_op):
        new_ops[i] += tensordot_loop(np.conjugate(mMatrix[:,i]), c)
    return new_ops

@njit(complex128[:,:](complex128[:,:]))
def sqrt_jit(M):
    # Computing diagonalization
    evalues, evectors = np.linalg.eig(M)

    # Ensuring square root matrix exists
    sqrt_matrix = evectors @ np.diag(np.sqrt(evalues)) @ np.linalg.inv(evectors)
    return np.ascontiguousarray(sqrt_matrix)

@njit(complex128[:,:](complex128[:,:], complex128[:,:], complex128[:,:]))
def H_DH(CA, CB, rho):
    CA, CB, rho = np.ascontiguousarray(CA), np.ascontiguousarray(CB), np.ascontiguousarray(rho)
    A1 = np.dot(np.dot(CB,CA),rho) + np.dot(np.dot(CB,rho),np.transpose(np.conjugate(CA))) + np.dot(np.dot(CA,rho),np.transpose(np.conjugate(CB))) + np.dot(np.dot(rho,np.transpose(np.conjugate(CA))),np.transpose(np.conjugate(CB)))
    A2 = np.dot(CA,rho) + np.dot(rho,np.conjugate(np.transpose(CA)))
    A3 = np.dot(CB,rho) + np.dot(rho,np.conjugate(np.transpose(CB)))
    return A1 - np.trace(A2)*A3 - np.trace(A3)*A2 + 2*np.trace(A2)*np.trace(A3)*rho - rho*np.trace(A1)

@njit(float64(complex128[:,:], complex128[:,:]))
def fidelity(rho1, rho2):
    rho1 = np.ascontiguousarray(rho1)
    rho2 = np.ascontiguousarray(rho2)
    srho1 = np.ascontiguousarray(sqrt_jit(rho1))
    return np.real(np.trace(sqrt_jit(srho1.dot(rho2.dot(srho1)))))
