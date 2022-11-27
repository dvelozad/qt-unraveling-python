'''
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
'''
import numpy as np
from numba import complex128
from numba.pycc import CC

cc = CC('usual_operators')
# Uncomment the following line to print out the compilation steps
# cc.verbose = True

@cc.export('Com', complex128[:,:](complex128[:,:], complex128[:,:]))
def Com(a,b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return np.dot(a,b) - np.dot(b,a)

@cc.export('AntiCom', complex128[:,:](complex128[:,:], complex128[:,:]))
def AntiCom(a,b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return np.dot(a,b) + np.dot(b,a)

@cc.export('D', complex128[:,:](complex128[:,:], complex128[:,:]))
def D(A, B):
    A, B = np.ascontiguousarray(A), np.ascontiguousarray(B)
    AT = np.transpose(np.conjugate(A))
    AntiCom = np.dot(np.dot(AT,A),B) + np.dot(B,np.dot(AT,A))
    return np.dot(A, np.dot(B,AT)) - 0.5*AntiCom

@cc.export('D_vec', complex128[:,:](complex128[:,:,:], complex128[:,:]))
def D_vec(A, B):
    B = np.ascontiguousarray(B)
    D_ = np.zeros(np.shape(B), dtype=np.complex128)
    for A_i in A:
        A_i = np.ascontiguousarray(A_i)
        AT_i = np.ascontiguousarray(np.transpose(np.conjugate(A_i)))
        AntiCom = np.dot(np.dot(AT_i,A_i),B) + np.dot(B,np.dot(AT_i,A_i))
        D_ += np.dot(A_i, np.dot(B,AT_i)) - 0.5*AntiCom
    return D_

@cc.export('H_vec', complex128[:,:,:](complex128[:,:,:], complex128[:,:]))
def H_vec(A, B):
    A, B = np.ascontiguousarray(A), np.ascontiguousarray(B)
    H = np.zeros((np.shape(A)[0], np.shape(B)[0], np.shape(B)[1]), dtype=np.complex128)
    for n_A, A_i in enumerate(A):
        A_i = np.ascontiguousarray(A_i)
        Aux = np.dot(A_i,B) + np.dot(B,np.transpose(np.conjugate(A_i)))
        H[n_A] += Aux - np.trace(Aux)*B
    return H

@cc.export('H', complex128[:,:](complex128[:,:], complex128[:,:]))
def H(A, B):
    A, B = np.ascontiguousarray(A), np.ascontiguousarray(B)
    Aux = np.dot(A,B) + np.dot(B,np.transpose(np.conjugate(A)))
    return Aux - np.trace(Aux)*B

@cc.export('operators_Mrep', complex128[:,:,:](complex128[:,:], complex128[:,:,:]))
def operators_Mrep(mMatrix, c):
    ## Number of associated operators
    transformed_num_op = np.shape(mMatrix)[1]
    new_ops = np.zeros((transformed_num_op, np.shape(c)[1], np.shape(c)[2]), dtype=np.complex128)
    for i in range(transformed_num_op):
        tmp = np.zeros(np.shape(c)[1:], dtype=np.complex128)
        for n_number, number in enumerate(np.conjugate(mMatrix[:,i])):
            tmp += number*c[n_number] 
        new_ops[i] += np.ascontiguousarray(tmp)
    return new_ops

@cc.export('sqrt_jit', complex128[:,:](complex128[:,:]))
def sqrt_jit(M):
    # Computing diagonalization
    evalues, evectors = np.linalg.eig(M)
    # Ensuring square root matrix exists
    sqrt_matrix = evectors @ np.diag(np.sqrt(evalues)) @ np.linalg.inv(evectors)
    return sqrt_matrix

cc.compile()