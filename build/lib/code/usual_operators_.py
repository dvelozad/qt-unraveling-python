'''
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
'''
import numpy as np
from numba import njit, jit, complex128

## Pauli matrices
sigmax = np.array([[0,1],[1,0]], dtype = np.complex128)
sigmay = np.array([[0,-1j],[1j,0]], dtype = np.complex128)
sigmaz = np.array([[1,0],[0,-1]], dtype = np.complex128)

## Lader operators
sigmap = 0.5*(sigmax + 1j*sigmay)
sigmam = 0.5*(sigmax - 1j*sigmay)

@njit(complex128[:,:](complex128[:,:], complex128[:,:]))
def Com(a,b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return np.dot(a,b) - np.dot(b,a)

@njit(complex128[:,:](complex128[:,:], complex128[:,:]))
def AntiCom(a,b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return np.dot(a,b) + np.dot(b,a)

@njit(complex128[:,:](complex128[:,:], complex128[:,:]))
def D(A, B):
    A, B = np.ascontiguousarray(A), np.ascontiguousarray(B)
    AT = np.transpose(np.conjugate(A))
    return np.dot(A, np.dot(B,AT)) - 0.5*AntiCom(np.dot(AT,A),B)

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

@njit(complex128[:,:](complex128[:,:], complex128[:,:]))
def H(A, B):
    A, B = np.ascontiguousarray(A), np.ascontiguousarray(B)
    Aux = np.dot(A,B) + np.dot(B,np.transpose(np.conjugate(A)))
    return Aux - np.trace(Aux)*B

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
    return sqrt_matrix