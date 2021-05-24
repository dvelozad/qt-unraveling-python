import gc
import copy
import random
from multiprocessing import Pool, cpu_count
from functools import partial
from inspect import getfullargspec

from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm
from scipy.integrate._ivp.rk import DOP853 as DOP853_

import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

def F_dummy(x):
    return np.zeros(10)

def L_dummy(x):
    return np.zeros(10)

### System class
class System:  
    def __init__(self, drivingH, initialState, timeList, *, lindbladList = L_dummy, FList = F_dummy, uMatrix = [], mMatrix = [], oMatrix = [], HMatrix = [], amp= None):
        #########################################################
        #### Class parameter. Numpy objects and lists
        #########################################################
        self.timeList=timeList
        self.initialState = initialState

        self.dimH = len(initialState)
        
        self.H = drivingH
        self.tmax = timeList[-1]
        self.t0 = timeList[0]
        self.dt = abs(timeList[1]-timeList[0])
        self.maxiter = len(timeList)

        #########################################################
        #### Initial state definitions
        #########################################################
        if len(np.shape(initialState)) == 1:
            self.initial_counter = 0
            # In case the intial state isn't normalized
            if np.round(np.linalg.norm(initialState),5) != 1:
                #print('Initial state is unnormalized. Normalized state taken instead')
                self.jumpPsi0 = (1./np.linalg.norm(initialState))*initialState
            else:
                self.jumpPsi0 = initialState  
            self.initialRho = np.asarray(np.transpose(np.asmatrix(self.jumpPsi0)).dot(np.conjugate(np.asmatrix(self.jumpPsi0))))   
                                         
        elif len(np.shape(initialState)) == 2:
            self.initial_counter = 1
            self.initialRho = np.asarray(initialState)

        #########################################################
        #### Lindblad and Feedback operator related definitions 
        #########################################################
        if len(getfullargspec(lindbladList).args) == 0:
            self.cList = partial(self.mod_fun0, lindbladList, rho = 0)
        elif len(getfullargspec(lindbladList).args) == 1:
            self.cList = lindbladList

        if len(getfullargspec(FList).args) == 0:
            self.F = partial(self.mod_fun0, FList)
        elif len(getfullargspec(FList).args) == 1:
            self.F = partial(self.mod_fun1, FList)
        elif len(getfullargspec(FList).args) == 2:
            self.F = FList

        self.c0 = self.cList(0)
        self.F0 = self.F(0,0)
        self.num_op = len(self.c0)
        #########################################################
        #### QJump finite external oscillator amplitude
        #########################################################
        if amp == None:
            self.default_amp = self.dt**(-2/3)*np.ones(2*self.num_op)
            self.DT = self.dt
        else:
            if len(amp) == 2*self.num_op:
                self.default_amp = amp
            else:
                self.default_amp = np.concatenate((np.array(amp), np.zeros(int(2*self.num_op - len(amp))))).flatten()

        ######################################################################
        ###################### M and U definitions ###########################
        ######################################################################
        self.nonfixedUnraveling = 0
        mMatrix_suprt = mMatrix
        uMatrix_suprt = uMatrix
        HMatrix_suprt = HMatrix
        oMatrix_suprt = oMatrix
        self.mMatrix = mMatrix
        self.uMatrix = uMatrix

        ## O Matrix
        if oMatrix == []:
            self.oMatrix = np.eye(2*self.num_op)
        else:
            self.oMatrix = oMatrix

        ### H matrix
        if type(HMatrix) != type(F_dummy):
            if len(HMatrix) == 0:
                self.HMatrix = np.eye(self.num_op)  
            else:
                self.HMatrix = HMatrix
        else:
            self.HMatrix = HMatrix

        if type(mMatrix) == type(F_dummy) and type(uMatrix) == type(F_dummy):
            raise ValueError('Both U and M representation matrices are defined, this could lead to errors. Please just define one.')
        if type(mMatrix) == type(F_dummy) and type(HMatrix) == type(F_dummy):
            raise ValueError('Both U and M representation matrices are defined, this could lead to errors. Please just define one.')
        
        if type(uMatrix) == type(F_dummy) and type(HMatrix) == type(F_dummy):
            self.nonfixedUnraveling = 1
            uMatrix_suprt = []
            HMatrix_suprt = []
            self.update_defintions = self.update_defintions_uH
        
        if type(uMatrix) == type(F_dummy) and type(HMatrix) != type(F_dummy):
            self.nonfixedUnraveling = 1
            uMatrix_suprt = []
            HMatrix_suprt = []
            self.update_defintions = self.update_defintions_u
        
        if type(HMatrix) == type(F_dummy) and type(uMatrix) != type(F_dummy):
            self.nonfixedUnraveling = 1
            uMatrix_suprt = []
            HMatrix_suprt = []
            self.update_defintions = self.update_defintions_H 
        
        if type(mMatrix) == type(F_dummy):
            self.nonfixedUnraveling = 1
            mMatrix_suprt = []  
            self.update_defintions = self.update_defintions_M
        
        #####
        self.U_rep, self.M_rep, self.T_rep = representation(self.num_op, mMatrix_suprt, uMatrix_suprt, HMatrix_suprt, oMatrix_suprt)

        ### Adittional definitions 
        self.Mt_aux = np.conjugate(np.transpose(self.M_rep))
        self.MMT = np.round(np.asmatrix(self.M_rep).dot(np.conjugate(np.transpose(np.asmatrix(self.M_rep)))),6)
        self.sqrt_MMT = sqrtm(np.identity(self.num_op) - self.MMT)

        ######################################################################
        ####################### M and U conditions ###########################
        ######################################################################
        self.inefficient, self.eta_diag = condition_check(self.U_rep, self.M_rep)

        ######################################################################################
        #### for time-independent cases
        ######################################################################################
        self.MtC0 = []
        self.MTC0 = []
        for i in range(2*self.num_op):
            MtC0_i = np.zeros((self.dimH, self.dimH), dtype = np.complex128, order='C')
            MTC0_i = np.zeros((self.dimH, self.dimH), dtype = np.complex128, order='C')
            for j in range(self.num_op):
                MtC0_i += self.Mt_aux[i,j]*self.c0[j]
                MTC0_i += self.M_rep[j,i]*np.transpose(np.conjugate(self.c0[j]))
            self.MTC0.append(MTC0_i)
            self.MtC0.append(MtC0_i)
        MtC = self.MtC0 
        self.non_unitary = np.zeros((self.dimH, self.dimH),dtype=np.complex128)       
        for i in range(2*self.num_op):
            self.non_unitary += np.dot(np.conjugate(np.transpose(MtC[i] + self.default_amp[i]*np.eye(self.dimH))), MtC[i] + self.default_amp[i]*np.eye(self.dimH))
            self.non_unitary += (np.conjugate(self.default_amp[i])*MtC[i] - self.default_amp[i]*np.transpose(np.conjugate(MtC[i])))

        ######################################################################################
        ## Feedback realted definitions
        ######################################################################################
        if FList != F_dummy:           
            self.sqrt_eta = sqrtm(np.asmatrix(np.eye(2*self.num_op)) - np.transpose(np.conjugate(np.asmatrix(self.M_rep))).dot(np.asmatrix(self.M_rep)))
            sqrt_eta_F0 = []
            ciMF0 = []
            for i in range(2*self.num_op):
                aux_eta_F0 = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                for j in range(2*self.num_op):
                    aux_eta_F0 += np.array(self.sqrt_eta[i][j]*self.F0[j])

                if np.max(np.abs(aux_eta_F0)) != 0:
                    sqrt_eta_F0.append(aux_eta_F0)
                    
            for i in range(self.num_op):
                aux_ciMF0 = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                for j in range(2*self.num_op):
                    aux_ciMF0 += np.array(-1j*self.M_rep[i][j]*self.F0[j])
                    
                ciMF0.append(self.c0[i] + aux_ciMF0)
                    
            self.sqrt_eta_F0 = sqrt_eta_F0
            self.ciMF0 = ciMF0
            
            if lindbladList != L_dummy:
                H_feed = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                for i in range(self.num_op):
                    aux_MF0 = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                    aux_MF0T = np.zeros((self.dimH,self.dimH), dtype = np.complex128)

                    for j in range(2*self.num_op):
                        aux_MF0 += np.array(self.M_rep[i][j]*self.F0[j])
                        aux_MF0T += np.array(np.conjugate(self.M_rep[i][j])*self.F0[j])

                    H_feed += 0.5*(aux_MF0T.dot(self.c0[i]) + np.transpose(np.conjugate(np.asmatrix(self.c0[i]))).dot(aux_MF0))
                self.H_feed = np.asarray(H_feed)

            ########################################################################################################################
            ## for time-independent cases
            ########################################################################################################################
            self.MtCF = []
            for i in range(2*self.num_op):
                MtCF_i = np.zeros((self.dimH, self.dimH), dtype = np.complex128, order='C')
                for j in range(self.num_op):
                    MtCF_i += self.Mt_aux[i,j]*self.c0[j]
                self.MtCF.append(MtCF_i - 1j*self.F0[i]) 

    ##########################################################################################
    ## Useful functions
    ##########################################################################################
    def eta_c(self, t):
        c = self.cList(t)
        eta_vec = []
        for k in range(self.num_op):
            eta_aux = np.zeros((self.dimH,self.dimH), dtype=np.complex128)
            for l in range(self.num_op):
                eta_aux += self.sqrt_MMT[l][k]*(c[l])
            eta_vec.append(eta_aux)
        return eta_vec
    
    def MtC(self, t):
        MtC = []
        c = self.cList(t)
        for i in range(2*self.num_op):
            MtC_i = np.zeros((self.dimH, self.dimH), dtype = np.complex128, order='C')
            for j in range(self.num_op):
                MtC_i += self.Mt_aux[i,j]*c[j]
            MtC.append(MtC_i)
        return MtC

    def MtC_gamma(self, t):
        MtC = []
        c = self.cList(t)
        for i in range(2*self.num_op):
            MtC_i = np.zeros((self.dimH, self.dimH), dtype = np.complex128, order='C')
            for j in range(self.num_op):
                MtC_i += self.Mt_aux[i,j]*c[j]
            MtC.append(MtC_i + self.default_amp[i]*np.eye(self.dimH))
        return MtC

    def dZeta(self,n): #gives a matrix with the noise
        rng = RandomState(MT19937(SeedSequence(156324*n)))
        N = self.num_op
        mean = np.zeros(2*N) ###dt*w
        cov = self.U_rep*self.dt
        A= rng.multivariate_normal(mean, cov, size=(self.maxiter,))
        B=[]
        for i in range(N):
            B.append(A[:,i] + 1j*A[:,N+(i+N)%N])
        return np.transpose(B)  

    def dZeta_nf(self, U, n): #gives a matrix with the noise
        rng = RandomState(MT19937(SeedSequence(156324*n)))
        N = self.num_op
        mean = np.zeros(2*N) ###dt*w
        cov = U*self.dt
        A= rng.multivariate_normal(mean, cov)
        B=[]
        for i in range(N):
            B.append(A[i] + 1j*A[N+(i+N)%N])
        return np.transpose(B)  

    def dZ_gauss(self, seed_n, N): #gives a matrix with the noise
        rng = RandomState(MT19937(SeedSequence(156324*seed_n)))
        mean = np.zeros(N) ###dt*w
        cov = np.eye(N)*self.dt
        A = rng.multivariate_normal(mean, cov, size=(self.maxiter,))
        B=[]
        for i in range(N):
            B.append(A[:,i])
        return np.transpose(B) 

    def H_gamma(self, t):
        MtC = self.MtC(t)
        H_ = self.H(t)
        R = np.zeros((self.dimH, self.dimH),dtype=np.complex128)
        for i in range(2*self.num_op):
            H_ += -1j*0.5*(np.conjugate(self.default_amp[i])*MtC[i] - self.default_amp[i]*np.transpose(np.conjugate(MtC[i])))
            R += -0.5j*np.dot(np.conjugate(np.transpose(MtC[i] + self.default_amp[i]*np.eye(self.dimH))), MtC[i] + self.default_amp[i]*np.eye(self.dimH))
        return H_ + R
    ##########################################################################################
    #####################################-Diffusive-##########################################
    ##########################################################################################
          
    ##########################################################################################
    ## Euler integrator 
    ##########################################################################################
    
    def diffusiveRhoEulerStep(self, it, rho_c, dz):
        c = self.cList(self.t0 + self.dt*it)
        MtC = self.MtC(self.t0 + self.dt*it)
        commu1 = -1j*opCom(self.H(self.t0 + self.dt*it), rho_c)
        Dc = opD(c, rho_c)
        
        dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz), np.imag(dz)]).flatten())
        A1 = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
        for i in range(2*self.num_op):
            A1 += dw[i]*MtC[i]   
        Hw = opH(A1, rho_c) 
        
        return (self.dt*(commu1 + Dc) + Hw)

    def diffuisiveQuadrature(self, MtC, psi):
        Q = []
        for i in range(2*self.num_op):
            Q.append(np.dot(np.conjugate(psi),np.dot(MtC[i] + np.transpose(np.conjugate(MtC[i])),psi)))
        return np.real(np.array(Q, dtype = np.complex128))
    
    def diffusivePureEulerStep(self, it, psi, dz):
        dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz), np.imag(dz)]).flatten())
        c = self.cList(self.t0 + self.dt*it)
        MtC = self.MtC(self.t0 + self.dt*it)
        Q = self.diffuisiveQuadrature(MtC, psi)
        aux = -1j*self.H(self.t0 + self.dt*it)*self.dt
        for i in range(self.num_op):
            aux += -0.5*np.dot(np.conjugate(np.transpose(c[i])),c[i])*self.dt 
            aux += 0.5*(Q[i]*MtC[i] + Q[i + self.num_op]*MtC[i + self.num_op])*self.dt
            aux += (-1./8.)*(Q[i]**2. + Q[i + self.num_op]**2.)*np.eye(self.dimH)*self.dt
            aux += (MtC[i] - 0.5*Q[i]*np.eye(self.dimH))*dw[i]
            aux += (MtC[i + self.num_op] - 0.5*Q[i + self.num_op]*np.eye(self.dimH))*dw[i + self.num_op]
        return np.dot(aux, psi)
    
    ##########################################################################################
    ## Heun integrator 
    ##########################################################################################
    def diffusivePureHeunStep(self, it, psi, dz):
        dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz), np.imag(dz)]).flatten())

        gi = self.diffusivePureMilstein_g(it, psi)
        fi = self.diffusivePureHeun_f_strat(it, psi, gi)
        
        G = np.zeros(self.dimH, dtype = np.complex128)
        for i in range(2*self.num_op):
            G += dw[i]*gi[i]
        
        gi_= self.diffusivePureMilstein_g(it, psi + G)

        G = np.zeros(self.dimH, dtype = np.complex128)
        for i in range(2*self.num_op):
            G += 0.5*dw[i]*(gi[i] + gi_[i])
            
        return fi*self.dt + G
    
    def diffusivePureHeun_f_strat(self, it, psi, gi):
        f_ito = self.diffusivePureMilstein_f(it, psi)      
        strat_corr = np.zeros(self.dimH, dtype = np.complex128, order='C')
        for i in range(2*self.num_op):     
            gi_= self.diffusivePureMilstein_g(it, psi + f_ito*self.dt + np.sqrt(self.dt)*gi[i])
            strat_corr += (1/np.sqrt(self.dt))*(gi_[i] - gi[i])        
        return f_ito - 0.5*strat_corr
    
    def diffusiveRhoHeunStep(self, it, rho_c, dz):
        dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz), np.imag(dz)]).flatten())

        gi = self.diffusiveRhoHeun_g(it, rho_c)
        fi = self.diffusiveRhoHeun_f_strat(it, rho_c)
        
        G = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
        for i in range(2*self.num_op):
            G += dw[i]*gi[i]
        
        gi_= self.diffusiveRhoHeun_g(it, rho_c + G)

        G = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
        for i in range(self.num_op):
            G += 0.5*dw[i]*(gi[i] + gi_[i])
            G += 0.5*dw[i + self.num_op]*(gi[i + self.num_op] + gi_[i + self.num_op])
        return fi*self.dt + G
    
    def diffusiveRhoHeun_f_strat(self, it, rho_c):
        c = self.cList(self.t0 + self.dt*it)
        MtC = self.MtC(self.t0 + self.dt*it)
        commu1 = -1j*opCom(self.H(self.t0 + self.dt*it), rho_c)                     
        Dc = opD(c, rho_c)
   
        f_ito = commu1 + Dc
        strat_corr = np.zeros((self.dimH, self.dimH), dtype = np.complex128, order='C')
        for i in range(2*self.num_op):
            strat_corr += opH_DH(MtC[i], MtC[i], rho_c)
        return f_ito - 0.5*strat_corr
    
    def diffusiveRhoHeun_g(self, it, rho_c):
        MtC = self.MtC(self.t0 + self.dt*it)
        HW = []
        for i in range(2*self.num_op):
            A1 = MtC[i]
            HW.append(opH(A1, rho_c)) 
        return HW
    ##########################################################################################
    ## Milstein integrator 
    ##########################################################################################   
    def diffusivePureMilstein_f(self, it, psi):
        c = self.cList(self.t0 + self.dt*it)
        MtC = self.MtC(self.t0 + self.dt*it)
        Q = self.diffuisiveQuadrature(MtC, psi)
        aux = -1j*self.H(self.t0 + self.dt*it)
        for i in range(self.num_op):
            aux += -0.5*np.dot(np.conjugate(np.transpose(c[i])),c[i])
            aux += 0.5*(Q[i]*MtC[i] + Q[i + self.num_op]*MtC[i + self.num_op])
            aux += (-1./8.)*(Q[i]**2. + Q[i + self.num_op]**2.)*np.eye(self.dimH)
        return np.dot(aux, psi)
    
    def diffusivePureMilstein_g(self, it, psi):
        R = []
        MtC = self.MtC(self.t0 + self.dt*it)
        Q = self.diffuisiveQuadrature(MtC, psi)
        for i in range(2*self.num_op):
            aux = MtC[i] - 0.5*Q[i]*np.eye(self.dimH)
            R.append(np.dot(aux, psi))
        return R
    
    def diffusivePureMilsteinStep(self, it, psi, dz):
        dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz), np.imag(dz)]).flatten())
        
        gi = self.diffusivePureMilstein_g(it, psi)
        fi = self.diffusivePureMilstein_f(it, psi)

        G_w = np.zeros(self.dimH, dtype = np.complex128)
        G_aux = np.zeros(self.dimH, dtype = np.complex128)
        
        for i in range(2*self.num_op):
            ## G bar
            gi_= self.diffusivePureMilstein_g(it, psi + fi*self.dt + np.sqrt(self.dt)*gi[i])
            
            G_w += dw[i]*gi[i]          
            for j in range(2*self.num_op):
                if i == j:
                    G_aux += (0.5/np.sqrt(self.dt))*(gi_[j] - gi[j])*((dw[j])**2. - self.dt)
                else:
                    G_aux += (0.5/np.sqrt(self.dt))*(gi_[j] - gi[j])*dw[i]*dw[j]
            
        return fi*self.dt + G_aux + G_w

    def diffusiveRhoMilstein_f(self, it, rho_c):
        c = self.cList(self.t0 + self.dt*it)
        commu1 = -1j*opCom(self.H(self.t0 + self.dt*it), rho_c)                     
        Dc = opD(c, rho_c)
        f = commu1 + Dc 
        return f
    
    def diffusiveRhoMilstein_g(self, it, rho_c):
        MtC = self.MtC(self.t0 + self.dt*it)
        HW = []
        for i in range(2*self.num_op):
            A1 = MtC[i]
            HW.append(opH(A1, rho_c)) 
        return HW
    
    def diffusiveRhoMilsteinStep(self, it, rho_c, dz):
        dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz), np.imag(dz)]).flatten())
        MtC = self.MtC(self.t0 + self.dt*it)
        gi = self.diffusiveRhoMilstein_g(it, rho_c)
        fi = self.diffusiveRhoMilstein_f(it, rho_c)

        G_w = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
        G_aux = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
        
        for i in range(2*self.num_op):
            G_w += dw[i]*gi[i]
            
            for j in range(2*self.num_op):
                if i == j:
                    G_aux += 0.5*opH_DH(MtC[i], MtC[j], rho_c)*(dw[i]*dw[j] - self.dt)
                else:
                    G_aux += 0.5*opH_DH(MtC[i], MtC[j], rho_c)*dw[i]*dw[j]
            
        return fi*self.dt + G_aux + G_w
    ##########################################################################################
    ### Trajectory functions 
    ##########################################################################################
    def diffusivePsiTrajectory(self, seed_n, method = 1): #gives a single state trajectory 
        if self.initial_counter == 1:
            raise ValueError('Initial mixed state defined')
        if self.inefficient == True:
            raise ValueError('Inefficient unraveling defined, please use a density matrix method instead')
        state = self.jumpPsi0
        vec = [state]
        if self.nonfixedUnraveling == 0:
            dz = self.dZeta(seed_n)
            # Heun
            if method == 0:
                for it in range(self.maxiter):
                    state = state + self.diffusivePureHeunStep(it, state, dz[it])
                    state = (1/np.linalg.norm(state))*state
                    vec.append(state)
            # Euler
            if method == 1:
                for it in range(self.maxiter):
                    state = state + self.diffusivePureEulerStep(it, state, dz[it])
                    state = (1/np.linalg.norm(state))*state
                    vec.append(state)
            # Milstein
            if method == 2:
                for it in range(self.maxiter):
                    state = state + self.diffusivePureMilsteinStep(it, state, dz[it])
                    state = (1/np.linalg.norm(state))*state
                    vec.append(state)
            return vec
        elif self.nonfixedUnraveling == 1:
            # Heun
            if method == 0:
                for it in range(self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, state)
                    dz = self.dZeta_nf(self.U_rep, seed_n + it)
                    state = state + self.diffusivePureHeunStep(it, state, dz)
                    state = (1/np.linalg.norm(state))*state
                    vec.append(state)
            # Euler
            if method == 1:
                for it in range(self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, state)
                    dz = self.dZeta_nf(self.U_rep, seed_n + it)
                    state = state + self.diffusivePureEulerStep(it, state, dz)
                    state = (1/np.linalg.norm(state))*state
                    vec.append(state)
            # Milstein
            if method == 2:
                for it in range(self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, state)
                    dz = self.dZeta_nf(self.U_rep, seed_n + it)
                    state = state + self.diffusivePureMilsteinStep(it, state, dz)
                    state = (1/np.linalg.norm(state))*state
                    vec.append(state)
            return vec
    
    def diffusiveRhoTrajectory(self, seed_nn, method = 1):
        rho = self.initialRho
        rho_T = [rho]
        if self.nonfixedUnraveling == 0:
            dz = self.dZeta(seed_nn)
            # Heun
            if method == 0:
                for it in range(1, self.maxiter):
                    rho = rho + self.diffusiveRhoHeunStep(it, rho, dz[it])
                    #rho = (1./np.linalg.norm(rho))*rho
                    rho_T.append(rho)
            # Euler
            elif method == 1:
                for it in range(1, self.maxiter):
                    rho = rho + self.diffusiveRhoEulerStep(it, rho, dz[it])
                    #rho = (1./np.linalg.norm(rho))*rho
                    rho_T.append(rho)
            # Milstein
            elif method == 2:
                for it in range(1, self.maxiter):
                    rho = rho + self.diffusiveRhoMilsteinStep(it, rho, dz[it])
                    #rho = (1./np.linalg.norm(rho))*rho
                    rho_T.append(rho)
            return rho_T
        elif self.nonfixedUnraveling == 1:
            # Heun
            if method == 0:
                for it in range(1, self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, rho)
                    dz = self.dZeta_nf(self.U_rep, seed_nn + it)
                    rho = rho + self.diffusiveRhoHeunStep(it, rho, dz)
                    #rho = (1./np.linalg.norm(rho))*rho
                    rho_T.append(rho)
            # Euler
            elif method == 1:
                for it in range(1, self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, rho)
                    dz = self.dZeta_nf(self.U_rep, seed_nn + it)
                    rho = rho + self.diffusiveRhoEulerStep(it, rho, dz)
                    #rho = (1./np.linalg.norm(rho))*rho
                    rho_T.append(rho)
            # Milstein
            elif method == 2:
                for it in range(1, self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, rho)
                    dz = self.dZeta_nf(self.U_rep, seed_nn + it)
                    rho = rho + self.diffusiveRhoMilsteinStep(it, rho, dz)
                    #rho = (1./np.linalg.norm(rho))*rho
                    rho_T.append(rho)
            return rho_T
            
    def diffusiveRhoAverage(self, n_trajectories, traj_type = 'density_matrix', method = 'euler', parallelfor = True):
        ########################################################
        if method == "heun":
            Me = 0
        elif method == "euler":
            Me = 1
        elif method == "milstein":
            Me = 2
        else:
            Me = 1
            #print("Method not supported. Euler selected")
        ########################################################
        if traj_type == 'vector':
            if parallelfor == True:
                
                rho_ = np.zeros((self.maxiter, self.dimH, self.dimH), dtype = np.complex128) 
                cpu_cores = 6*cpu_count()
                res = n_trajectories%cpu_cores
                A = np.rint(100000*np.random.rand(n_trajectories)).astype(int)
                B = A[0:n_trajectories-res]
                B = B.reshape(int((n_trajectories-res)/cpu_cores), cpu_cores)

                R = [B[i] for i in range(int((n_trajectories-res)/cpu_cores))]
                R.append(A[n_trajectories-res:n_trajectories])

                for seg in R:
                    m = self.segmented_parallel_run(partial(self.diffusivePsiTrajectory, method = Me), seg) 
                    ###################################################################################
                    for i in range(self.maxiter):
                        rho_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                        for n in range(len(seg)):
                            psi = np.asmatrix(m[n][i])
                            rho_i += (1./n_trajectories)*np.transpose(psi).dot(np.conjugate(psi))
                        rho_[i] += rho_i
                    ###################################################################################
                    del m
                return rho_
            ##########################################################################################
            else:
                m = []
                for i in range(n_trajectories):
                    m.append(self.diffusivePsiTrajectory(n, method = Me))
                rho_ = []
                for i in range(self.maxiter):
                    rho_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                    for n in range(n_trajectories):
                        psi = np.asmatrix(m[n][i])
                        rho_i += (1./n_trajectories)*np.transpose(psi).dot(np.conjugate(psi))
                    rho_.append(rho_i)
                return rho_
            ###########################################################################################
        elif traj_type == 'density_matrix':
            if parallelfor == True:
                
                rho_ = np.zeros((self.maxiter, self.dimH, self.dimH), dtype = np.complex128) 
                cpu_cores = 6*cpu_count()
                res = n_trajectories%cpu_cores
                A = np.rint(100000*np.random.rand(n_trajectories)).astype(int)
                B = A[0:n_trajectories-res]
                B = B.reshape(int((n_trajectories-res)/cpu_cores), cpu_cores)

                R = [B[i] for i in range(int((n_trajectories-res)/cpu_cores))]
                R.append(A[n_trajectories-res:n_trajectories])

                for seg in R:
                    m = self.segmented_parallel_run(partial(self.diffusiveRhoTrajectory, method = Me), seg) 
                    ##################################################################
                    for i in range(self.maxiter):
                        rho_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                        for n in range(len(seg)):
                            rho_i += (1./n_trajectories)*m[n][i]     
                        rho_[i] += rho_i
                    ##################################################################
                    del m
                return rho_
            else:
                m = []
                for i in range(n_trajectories):
                    m.append(self.diffusiveRhoTrajectory(n, method = Me))
                rho_ = []
                for i in range(self.maxiter):
                    rho_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                    for n in range(n_trajectories):
                        rho_i += (1./n_trajectories)*m[n][i]
                    rho_.append(rho_i)
                return rho_

    def diffusivePsiAverage(self, n_trajectories, method = 'euler', parallelfor = True): 
        ###########################################################
        if method == "heun":
            Me = 0
        elif method == "euler":
            Me = 1
        elif method == "milstein":
            Me = 2
        else:
            Me = 1
            #print("Method not supported. Euler selected")
        ###########################################################
        if self.initial_counter == 1:
            raise ValueError('Initial mixed state defined')
        ###########################################################
        m = []
        if parallelfor == True:
            
            psi_f = np.zeros((self.maxiter, self.dimH), dtype = np.complex128) 
            cpu_cores = 5*cpu_count()
            res = n_trajectories%cpu_cores
            A = np.rint(100000*np.random.rand(n_trajectories)).astype(int)
            B = A[0:n_trajectories-res]
            B = B.reshape(int((n_trajectories-res)/cpu_cores), cpu_cores)

            R = [B[i] for i in range(int((n_trajectories-res)/cpu_cores))]
            R.append(A[n_trajectories-res:n_trajectories])

            for seg in R:
                m = self.segmented_parallel_run(partial(self.diffusivePsiTrajectory, method = Me), seg) 
                ###################################################################################
                for i in range(self.maxiter):
                    psi_i = np.zeros((self.dimH), dtype = np.complex128)
                    for n in range(len(seg)):
                        psi_i += (1./n_trajectories)*m[n][i]
                    psi_f[i] += psi_i
                ###################################################################################
                del m
            return psi_f
        else:
            for n in range(1, n_trajectories+1):
                m.append(self.diffusivePsiTrajectory(n, method = Me))
            psi_f = []           
            for i in range(self.maxiter):
                psi_i = np.zeros(self.dimH, dtype = np.complex128)
                for n in range(n_trajectories):
                    psi_i += (1./n_trajectories)*m[n][i]
                psi_f.append(psi_i)
            return psi_f 
        
    def diffusivePsiEnsemble(self, n_trajectories, method = 'euler'): #gives an ensemble of state trajectories
        if method == "heun":
            Me = 0
        elif method == "euler":
            Me = 1
        elif method == "milstein":
            Me = 2
        else:
            Me = 1
            #print("Method not supported. Euler selected")
        if self.initial_counter == 1:
            raise ValueError('Initial mixed state defined')
        
        ensemble = self.parallel_run(partial(self.diffusivePsiTrajectory, method = Me), n_trajectories) 
        return ensemble 
    
    def diffusiveRhoEnsemble(self, n_trajectories, method = 'euler', traj_type = 'density_matrix', parallelfor = True):
        if method == "heun":
            Me = 0
        elif method == "euler":
            Me = 1
        elif method == "milstein":
            Me = 2
        else:
            Me = 1
            print("Method not supported. Euler selected")
        
        if traj_type == 'density_matrix':
            m = []
            if parallelfor == True:
                
                m = self.parallel_run(partial(self.diffusiveRhoTrajectory, method = Me), n_trajectories) 
            else:
                for n in range(1,n_trajectories+1):
                    m.append(self.diffusiveRhoTrajectory(n, method = Me))
            return m 
        elif traj_type == 'vector':
            if parallelfor == True:
                
                mm = self.parallel_run(partial(self.diffusivePsiTrajectory, method = Me), n_trajectories) 
            else:
                for n in range(1,n_trajectories+1):
                    mm.append(self.diffusivePsiTrajectory(n, method = Me))

            m = []
            for n in range(n_trajectories):
                rho_n = []
                for i in range(self.maxiter):
                    aux = np.asmatrix(mm[n][i])
                    rho_n.append(np.transpose(aux).dot(np.conjugate(aux)))
                m.append(rho_n)
            return m 
    
    ##########################################################################################
    #####################################--QJumps--###########################################
    ########################################################################################## 

    ### Non-unitary pure evolution
    def jumpEq_system(self, t, y):
        c = self.cList(t)
        non_unitary = np.zeros((self.dimH, self.dimH),dtype=np.complex128)  
        for i in range(self.num_op):
            non_unitary += np.dot(np.conjugate(np.transpose(c[i])),c[i])
        return (-1j)*(self.H(t) - 0.5j*non_unitary).dot(y) 
    
    ### Non-unitary pure evolution with coherent field gamma
    def jumpEq_gamma_system(self, t, y):
        H_ = self.H(t)
        MtC = self.MtC(int((t-self.t0)/self.dt))
        non_unitary = np.zeros((self.dimH, self.dimH),dtype=np.complex128)       
        for i in range(2*self.num_op):
            non_unitary += np.dot(np.conjugate(np.transpose(MtC[i] + self.default_amp[i]*np.eye(self.dimH))), MtC[i] + self.default_amp[i]*np.eye(self.dimH))
            H_ += -1j*0.5*(np.conjugate(self.default_amp[i])*MtC[i] - self.default_amp[i]*np.transpose(np.conjugate(MtC[i])))
        dydt = (-1j)*np.dot(H_ - 0.5j*non_unitary, y) 
        return dydt

    ### Non-unitary pure evolution with coherent field gamma for time-independent lindblad op
    def jumpEq_gamma_system_0(self, t, y):
        H_ = self.H(t)
        dydt = (-1j)*np.dot(H_ - 0.5j*self.non_unitary, y) 
        return dydt

    ### Lindblad operators expected value
    def jumpCexpect(self, c_mu, psi):
        cdagc = np.transpose(np.conjugate(c_mu)).dot(c_mu)
        expect = np.conjugate(psi).dot(cdagc.dot(psi))
        return expect
    
    ### Index that describes the Qjump based on the corresponding probabilities
    def jumpdN(self, op_list, psi):
        weight = []
        M_index =[]

        for mu in range(len(op_list)):
            expect = self.jumpCexpect(op_list[mu], psi)
            weight.append(self.dt*expect)
            M_index.append(mu)
        M = random.choices(M_index, weights=weight)[0]
        return M
    
    def jumpdN_2(self, op_list, psi):
        weight = []
        M_index =[]

        for mu in range(len(op_list)):
            expect = self.jumpCexpect(op_list[mu], psi)
            weight.append(self.dt*expect)
            M_index.append(mu)
        
        R = np.random.rand()
        if R < sum(weight): 
            M = random.choices(M_index, weights=weight)[0]
        else:
            M = len(op_list)
        return M
    
    def jumpdNRho(self, op_list, rho_c):
        weight = []
        M_index =[]

        for mu in range(len(op_list)):
            c = np.asmatrix(op_list[mu])
            cdagc = np.transpose(np.conjugate(c)).dot(c)
            weight.append(self.dt*np.trace(rho_c.dot(cdagc)))
            M_index.append(mu)
        
        R = np.random.rand()
        #print(sum(weight))
        if R < sum(weight): 
            M = random.choices(M_index, weights=weight)[0]
        else:
            M = len(op_list)
        return M

    ### Discrete time step for pure and mixed trajectories
    def jumpEulerPsiStep(self, t, y, gamma_amp, mu):
        h_gamma = np.zeros((self.dimH, self.dimH),dtype=np.complex128)    
        cdagc = np.zeros((self.dimH, self.dimH),dtype=np.complex128)  
        
        c = self.cList(t)
        for i in range(self.num_op):
            h_gamma += -0.5*1j*gamma_amp*(c[i] - np.conjugate(np.transpose(c[i])))
            cdagc += -0.5*np.transpose(np.conjugate(c[i])).dot(c[i])
            
        if mu == len(c):
            psi_step = (-1j*self.dt*(self.H(t) + h_gamma) + self.dt*(cdagc)).dot(y)
            return (1/np.linalg.norm((y + psi_step)))*(y + psi_step)
        
        elif mu < len(c):
            jump_term = (1/np.sqrt(self.jumpCexpect(c[mu], y)))*c[mu] 
            psi_step = (jump_term).dot(y) 
            return (1/np.linalg.norm(psi_step))*(psi_step)

    def jumpEulerRhoStep(self, t, rho, gamma_amp, op_list, mu, unraveling, eta_MtC = 0):
        if unraveling == False:  
            cdagc = np.zeros((self.dimH, self.dimH),dtype=np.complex128)  

            c = self.cList(t)
            for i in range(self.num_op):
                cdagc += np.transpose(np.conjugate(c[i])).dot(c[i])

            if mu == len(op_list):
                rho_step = self.dt*opH(-1j*self.H(t) - 0.5*cdagc, rho) 
                return (rho + rho_step)

            elif mu < len(op_list):
                rho_step = opG(c[mu], rho) 
                return (rho + rho_step)
        else:
            if mu == len(op_list):
                rho_step = self.dt*(opH(-1j*self.H_gamma(t), rho) + opD(eta_MtC, rho))
                return (rho + rho_step)

            elif mu < len(op_list):
                rho_step = opG(op_list[mu], rho)
                return (rho + rho_step)

    def jumpRungeRhoStep(self, t, rho, gamma_amp, op_list, mu, unraveling, eta_MtC = 0):
        if unraveling == False:  
            cdagc = np.zeros((self.dimH, self.dimH),dtype=np.complex128)  

            c = self.cList(t)
            for i in range(self.num_op):
                cdagc += np.transpose(np.conjugate(c[i])).dot(c[i])

            if mu == len(op_list):
                rho_step_1 = self.dt*opH(-1j*self.H(t) - 0.5*cdagc, rho)
                rho_step_2 = self.dt*opH(-1j*self.H(t + 0.5*self.dt) - 0.5*cdagc, rho + 0.5*rho_step_1) 
                rho_step_3 = self.dt*opH(-1j*self.H(t + 0.5*self.dt) - 0.5*cdagc, rho + 0.5*rho_step_2) 
                rho_step_4 = self.dt*opH(-1j*self.H(t + self.dt) - 0.5*cdagc, rho + rho_step_3) 
                return (rho + (1/6)*rho_step_1 + (1/3)*rho_step_2 + (1/3)*rho_step_3 + (1/6)*rho_step_4) 

            elif mu < len(op_list):
                rho_step = opG(c[mu], rho) 
                return (rho + rho_step)
        else:
            if mu == len(op_list):
                rho_step_1 = self.dt*(opH(-1j*self.H_gamma(t), rho) + opD(eta_MtC, rho))
                rho_step_2 = self.dt*(opH(-1j*self.H_gamma(t + 0.5*self.dt), rho + 0.5*rho_step_1) + opD(eta_MtC, rho + 0.5*rho_step_1))
                rho_step_3 = self.dt*(opH(-1j*self.H_gamma(t + 0.5*self.dt), rho + 0.5*rho_step_2) + opD(eta_MtC, rho + 0.5*rho_step_2))
                rho_step_4 = self.dt*(opH(-1j*self.H_gamma(t + self.dt), rho + rho_step_3) + opD(eta_MtC, rho + rho_step_3))
                return (rho + (1/6)*rho_step_1 + (1/3)*rho_step_2 + (1/3)*rho_step_3 + (1/6)*rho_step_4) 

            elif mu < len(op_list):
                rho_step = opG(op_list[mu], rho)
                return (rho + rho_step)   

    ### Individual pure and mixed trajectories
    def jumpRhoTrajectory(self, nn, time_dep = True, unraveling = False):
        rho_t = self.initialRho
        rho_total = [rho_t]
        
        c0 = self.cList(0)
        MtC0 = self.MtC_gamma(0)
        eta_vec0 = self.eta_c(0)
        
        if unraveling == False:
            if time_dep == True:
                for i in range(1,self.maxiter):
                    #####################################################################
                    c = self.cList(self.t0 + self.dt*i)
                    #####################################################################
                    eta_vec = self.eta_c(self.t0 + self.dt*i)
                    rho_t = self.jumpRungeRhoStep(self.t0 + self.dt*i, rho_t, 0, c, self.jumpdNRho(c, rho_t), unraveling = False, eta_MtC = eta_vec)
                    rho_total.append(rho_t)
                    #####################################################################
            elif time_dep == False:
                for i in range(self.maxiter):
                    rho_t = self.jumpRungeRhoStep(self.t0 + self.dt*i, rho_t, 0, c0, self.jumpdNRho(c0, rho_t), unraveling = False, eta_MtC = eta_vec0)
                    rho_total.append(rho_t)
            
            return rho_total
        else:
            if time_dep == True:
                if self.nonfixedUnraveling == 0:
                    for i in range(1,self.maxiter):
                        #####################################################################
                        c = self.cList(self.t0 + self.dt*i)
                        MtC = self.MtC_gamma(self.t0 + self.dt*i)
                        eta_vec = self.eta_c(self.t0 + self.dt*i)
                        #####################################################################
                        rho_t = self.jumpRungeRhoStep(self.t0 + self.dt*i, rho_t, self.default_amp, MtC, self.jumpdNRho(MtC, rho_t), unraveling = True, eta_MtC = eta_vec)
                        #rho_t = (1/np.linalg.norm(rho_t))*rho_t
                        rho_total.append(rho_t)
                        #####################################################################ç
                elif self.nonfixedUnraveling == 1:
                    for i in range(1,self.maxiter):
                        #####################################################################
                        self.update_defintions(self.t0 + i*self.dt, rho_t)
                        #####################################################################
                        c = self.cList(self.t0 + self.dt*i)
                        MtC = self.MtC_gamma(self.t0 + self.dt*i)
                        eta_vec = self.eta_c(self.t0 + self.dt*i)
                        #####################################################################
                        rho_t = self.jumpRungeRhoStep(self.t0 + self.dt*i, rho_t, self.default_amp, MtC, self.jumpdNRho(MtC, rho_t), unraveling = True, eta_MtC = eta_vec)
                        #rho_t = (1/np.linalg.norm(rho_t))*rho_t
                        rho_total.append(rho_t)
                        #####################################################################ç
                    
            elif time_dep == False:
                for i in range(self.maxiter):
                    rho_t = self.jumpRungeRhoStep(self.t0 + self.dt*i, rho_t, self.default_amp, MtC0, self.jumpdNRho(MtC0, rho_t), unraveling = True, eta_MtC = eta_vec0)
                    #rho_t = (1/np.linalg.norm(rho_t))*rho_t
                    rho_total.append(rho_t)
            return rho_total

    def jumpPsiTrajectory(self, nn, unraveling = False, time_dep = True):
        if self.initial_counter == 1:
            raise ValueError('Initial mixed state defined')
        if self.inefficient == True:
            raise ValueError('Inefficient unraveling defined, please use a density matrix method instead')
        if self.nonfixedUnraveling == 1:
            raise ValueError('Adaptive unraveling not supported. Please use instead jumpRhoTrajectory')
        psi = self.jumpPsi0
        psi_total = []
        nmax = 100000000
        Tmax = 0
        rng = RandomState(MT19937(SeedSequence(156324*nn)))
        for n in range(nmax):
            R = rng.rand()
            if Tmax >= self.maxiter:
                break
            if unraveling == False:
                sol = solve_ivp(self.jumpEq_system, [self.timeList[0], self.timeList[-1]], psi, t_eval=self.timeList,  method = 'BDF')
                for i in range(self.maxiter):     
                    Psi = sol.y[:,i]
                    norm = np.linalg.norm(Psi)
                    if norm**2. - R < 0:
                        c = self.cList(self.t0 + self.dt*i)  
                        Tmax += i
                        mu = self.jumpdN(c, Psi)
                        psi = c[mu].dot(Psi)
                        psi = (1./np.linalg.norm(psi))*psi
                        break
                    if Tmax + i >= self.maxiter:
                        Tmax += i
                        psi = (1./np.linalg.norm(Psi))*Psi
                        psi_total.append(psi)
                        break
                    psi_total.append((1./norm)*Psi)
            elif unraveling == True:
                if time_dep == True:
                    sol = solve_ivp(self.jumpEq_gamma_system,  [self.timeList[0], self.timeList[-1]], psi, t_eval=self.timeList,  method = 'BDF')
                elif time_dep == False:
                    sol = solve_ivp(self.jumpEq_gamma_system_0,  [self.timeList[0], self.timeList[-1]], psi, t_eval=self.timeList,  method = 'BDF')
            
                for i in range(self.maxiter):  
                    Psi = sol.y[:,i]
                    norm = np.linalg.norm(Psi)
                    if norm**2. - R < 0:
                        #####################################################################
                        MtC = self.MtC_gamma(self.t0 + i*self.dt)
                        #####################################################################
                        Tmax += i
                        mu = self.jumpdN(MtC, Psi)
                        psi = (MtC[mu]).dot(Psi)
                        psi = (1./np.linalg.norm(psi))*psi
                        break
                    if Tmax + i >= self.maxiter:
                        Tmax += i
                        psi = (1./np.linalg.norm(Psi))*Psi
                        psi_total.append(psi)
                        break
                    psi_total.append((1./norm)*Psi)               
        return psi_total
    
    def jumpPsiTrajectory_2(self, nn = 0, unraveling_ = False):
        if self.initial_counter == 1:
            raise ValueError('Initial mixed state defined')
        if self.inefficient == True:
            raise ValueError('Inefficient unraveling defined, please use a density matrix method instead')
        psi_t = self.jumpPsi0
        psi_total = [psi_t]
        if self.nonfixedUnraveling == 0:
            for i in range(1,self.maxiter):
                ##################################################
                c = self.cList(self.t0 + self.dt*i)
                ##################################################
                psi_t = self.jumpEulerPsiStep(self.t0 + self.dt*i, psi_t, 0, self.jumpdN_2(c, psi_t))
                psi_total.append(psi_t)
            return psi_total
        elif self.nonfixedUnraveling == 1:
            for i in range(1,self.maxiter):
                ##################################################
                self.update_defintions(self.t0 + i*self.dt, psi_t)
                ##################################################
                c = self.cList(self.t0 + self.dt*i)
                ##################################################
                psi_t = self.jumpEulerPsiStep(self.t0 + self.dt*i, psi_t, 0, self.jumpdN_2(c, psi_t))
                psi_total.append(psi_t)
            return psi_total
    
    ### Ensemble of pure and mixed trajectories
    def jumpPsiEnsemble(self, n_trajectories, parallelfor = True, unraveling = False, time_dep = True):
        if parallelfor == True:
            
            m = self.parallel_run(partial(self.jumpPsiTrajectory, unraveling = unraveling, time_dep = time_dep), n_trajectories) 
        else:
            m = []
        return m

    def jumpRhoEnsemble(self, n_trajectories, traj_type = 'vector', method = 1, time_dep = True,  parallelfor = True, unraveling = False):
        if self.initial_counter == 0:
            traj_type = traj_type
        elif self.initial_counter == 1:
            traj_type = 'density_matrix'
            
        if traj_type == 'vector':
            if method == 1:
                if parallelfor == True:
                    
                    m = self.parallel_run(partial(self.jumpPsiTrajectory, unraveling = unraveling, time_dep = time_dep), n_trajectories) 
                else:
                    m = []
                    for i in range(n_trajectories):
                        m.append(self.jumpPsiTrajectory(i, unraveling = unraveling))
            elif method == 2:
                if parallelfor == True:
                    
                    m = self.parallel_run(partial(self.jumpPsiTrajectory_2, unraveling = unraveling, time_dep = time_dep), n_trajectories) 
                else:
                    m = []
                    for i in range(n_trajectories):
                        m.append(self.jumpPsiTrajectory_2(i))
            rho_en = []
            for n in range(n_trajectories):
                rho_n = []
                for i in range(self.maxiter):
                    psi = np.asmatrix(m[n][i])
                    rho_i = np.transpose(psi).dot(np.conjugate(psi))
                    rho_n.append(rho_i)
                rho_en.append(rho_n)
            return rho_en
        elif traj_type == 'density_matrix':
            if parallelfor == True:
                
                m = self.parallel_run(partial(self.jumpRhoTrajectory, unraveling = unraveling, time_dep = time_dep), n_trajectories)
            else:
                m = []
                for i in range(n_trajectories):
                    m.append(self.jumpRhoTrajectory(i, time_dep = time_dep, unraveling = unraveling))
            rho_en = []
            for n in range(n_trajectories):
                rho_n = []
                for i in range(self.maxiter):
                    rho_i = np.asmatrix(m[n][i])
                    rho_n.append(rho_i)
                rho_en.append(rho_n)
                
            return rho_en

    ### Average pure and mixed evolution     
    def jumpPsiAverage(self, n_trajectories, parallelfor = True, time_dep = True, unraveling = False):             
        m = []
        if parallelfor == True:
            
            psi_f = np.zeros((self.maxiter, self.dimH), dtype = np.complex128) 
            cpu_cores = 5*cpu_count()
            res = n_trajectories%cpu_cores
            A = np.rint(100000*np.random.rand(n_trajectories)).astype(int)
            B = A[0:n_trajectories-res]
            B = B.reshape(int((n_trajectories-res)/cpu_cores), cpu_cores)

            R = [B[i] for i in range(int((n_trajectories-res)/cpu_cores))]
            R.append(A[n_trajectories-res:n_trajectories])

            for seg in R:
                m = self.segmented_parallel_run(partial(self.jumpPsiTrajectory, unraveling = unraveling, time_dep = time_dep), seg) 
                ###################################################################################
                for i in range(self.maxiter):
                    psi_i = np.zeros((self.dimH), dtype = np.complex128)
                    for n in range(len(seg)):
                        psi_i += (1./n_trajectories)*m[n][i]
                    psi_f[i] += psi_i
                ###################################################################################
                del m
            return psi_f
        else:
            for n in range(1, n_trajectories+1):
                m.append(self.jumpPsiTrajectory(n, time_dep = time_dep, unraveling = unraveling))
            psi_f = []           
            for i in range(self.maxiter):
                psi_i = np.zeros(self.dimH, dtype = np.complex128)
                for n in range(n_trajectories):
                    psi_i += (1./n_trajectories)*m[n][i]
                psi_f.append(psi_i)
            return psi_f 
    
    def jumpRhoAverage(self, n_trajectories, traj_type = 'vector', method = 1, time_dep = True, parallelfor = True, unraveling = False): 
        #####################################################
        if self.initial_counter == 0:
            traj_type = traj_type
        elif self.initial_counter == 1:
            traj_type = 'density_matrix'
        #####################################################
        if traj_type == 'vector':
            if parallelfor == True:
                
                rho_ = np.zeros((self.maxiter, self.dimH, self.dimH), dtype = np.complex128) 
                cpu_cores = 12*cpu_count()
                res = n_trajectories%cpu_cores
                A = np.rint(100000*np.random.rand(n_trajectories)).astype(int)
                B = A[0:n_trajectories-res]
                B = B.reshape(int((n_trajectories-res)/cpu_cores), cpu_cores)

                R = [B[i] for i in range(int((n_trajectories-res)/cpu_cores))]
                R.append(A[n_trajectories-res:n_trajectories])

                for seg in R:
                    m = self.segmented_parallel_run(partial(self.jumpPsiTrajectory, unraveling = unraveling, time_dep = time_dep), seg) 
                    ###################################################################################
                    for i in range(self.maxiter):
                        rho_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                        for n in range(len(seg)):
                            psi = np.asmatrix(m[n][i])
                            rho_i += (1./n_trajectories)*np.transpose(psi).dot(np.conjugate(psi))
                        rho_[i] += rho_i
                    ###################################################################################
                    del m
                return rho_
            ##########################################################################################
            else:
                m = []
                for i in range(n_trajectories):
                    m.append(self.jumpPsiTrajectory(i,unraveling = unraveling))
                rho_ = []
                for i in range(self.maxiter):
                    rho_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                    for n in range(n_trajectories):
                        psi = np.asmatrix(m[n][i])
                        rho_i += (1./n_trajectories)*np.transpose(psi).dot(np.conjugate(psi))
                    rho_.append(rho_i)
                return rho_
            ###########################################################################################
        elif traj_type == 'density_matrix':
            if parallelfor == True:
                
                rho_ = np.zeros((self.maxiter, self.dimH, self.dimH), dtype = np.complex128) 
                cpu_cores = 12*cpu_count()
                res = n_trajectories%cpu_cores
                A = np.rint(100000*np.random.rand(n_trajectories)).astype(int)
                B = A[0:n_trajectories-res]
                B = B.reshape(int((n_trajectories-res)/cpu_cores), cpu_cores)

                R = [B[i] for i in range(int((n_trajectories-res)/cpu_cores))]
                R.append(A[n_trajectories-res:n_trajectories])

                for seg in R:
                    m = self.segmented_parallel_run(partial(self.jumpRhoTrajectory, unraveling = unraveling, time_dep = time_dep), seg) 
                    ##################################################################
                    for i in range(self.maxiter):
                        rho_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                        for n in range(len(seg)):
                            rho_i += (1./n_trajectories)*m[n][i]     
                        rho_[i] += rho_i
                    ##################################################################
                    del m
                return rho_
            else:
                m = []
                for i in range(n_trajectories):
                    m.append(self.jumpRhoTrajectory(i, time_dep = time_dep, unraveling = unraveling))
                rho_ = []
                for i in range(self.maxiter):
                    rho_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                    for n in range(n_trajectories):
                        rho_i += (1./n_trajectories)*m[n][i]
                    rho_.append(rho_i)
                return rho_
 
    #######################################################################################
    #####################################-Feed-############################################
    #######################################################################################

    #######################################################################################
    ## Euler method
    #######################################################################################
    def feedPureEulerStep(self, it, psi, dz):
        R = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
        JJ = self.current_Y(psi, it, dz)
        FF = self.F(self.t0 + self.dt*it, psi)
        
        c = self.cList(self.t0 + self.dt*it)
        MtC_aux = np.transpose(np.conjugate(self.M_rep))
        MtC = []
        
        MTC_aux = np.transpose(self.M_rep)
        MTC = []
        
        for i in range(2*self.num_op):
            MtC_i = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
            MTC_i = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
            for j in range(self.num_op):
                MtC_i += MtC_aux[i,j]*c[j]
                MTC_i += MTC_aux[i,j]*np.transpose(np.conjugate(c[j]))
            MtC.append(MtC_i)
            MTC.append(MTC_i)

        MtC = np.array(MtC)
        MTC = np.array(MTC)
              
        for j in range(self.num_op):
            R += -0.5*(np.transpose(np.conjugate(c[j])).dot(c[j]))
        
        for i in range(2*self.num_op):
            R += -0.5*(2.*1j*FF[i].dot(MtC[i]) + FF[i].dot(FF[i])) + JJ[i]*(MtC[i] - 1j*FF[i])
            
        return self.dt*(-1j*self.H(self.t0 + self.dt*it) + R).dot(psi)
    
    def feedRhoEulerStep(self, it, rho_c, dz):
        f = self.F(self.t0 + self.dt*it, rho_c) 
        c = self.cList(self.t0 + self.dt*it)
        MtC = self.MtC(self.t0 + self.dt*it)
        
        commu1 = -1j*opCom(self.H(self.t0 + self.dt*it), rho_c)
        for i in range(2*self.num_op):
            aux1 = np.dot(MtC[i],rho_c)
            aux2 = np.dot(rho_c,np.conjugate(np.transpose(MtC[i])))
            commu1 += -1j*opCom(f[i], aux1 + aux2)
                      
        Dc = opD(c, rho_c)
        Df = opD(f, rho_c)
        
        dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz), np.imag(dz)]).flatten())
        A1 = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
        for i in range(2*self.num_op):
            A1 += dw[i]*(MtC[i] - 1j*f[i])
        
        Hw = opH(A1, rho_c)          
        return self.dt*(commu1 + Dc + Df) + Hw
    ######################################################################################
    ## Milstein method
    ######################################################################################
    def feedRhoMilstein_f(self, it, rho_c, f):
        c = self.cList(self.t0 + self.dt*it)
        MtC = self.MtC(self.t0 + self.dt*it)
        
        commu1 = -1j*opCom(self.H(self.t0 + self.dt*it), rho_c)
        for i in range(2*self.num_op):
            aux1 = np.dot(MtC[i],rho_c)
            aux2 = np.dot(rho_c,np.conjugate(np.transpose(MtC[i])))
            commu1 += -1j*opCom(f[i], aux1 + aux2)
                      
        Dc = opD(c, rho_c)
        Df = opD(f, rho_c)

        return commu1 + Dc + Df
    
    def feedRhoMilstein_g(self, it, rho_c, f):
        MtC = self.MtC(self.t0 + self.dt*it)
        HW = []
        for i in range(2*self.num_op):
            A1 = MtC[i] - 1j*f[i]
            HW.append(opH(A1, rho_c)) 
        return HW

    def feedRhoMilsteinStep(self, it, rho_c, dz):
        F = self.F(self.t0 + self.dt*it, rho_c) 
        #MtC = self.MtC(self.t0 + self.dt*it)
        dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz), np.imag(dz)]).flatten())
        
        gi = self.feedRhoMilstein_g(it, rho_c, F)
        fi = self.feedRhoMilstein_f(it, rho_c, F)

        G_w = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
        G_aux = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
        
        for i in range(2*self.num_op):
            gi_= self.feedRhoMilstein_g(it, rho_c + fi*self.dt + np.sqrt(self.dt)*gi[i], F)
            G_w += dw[i]*gi[i]          
            for j in range(2*self.num_op):
                if i == j:
                    #G_aux += 0.5*opH_DH(MtC[i] - 1j*F[i], MtC[i] - 1j*F[i], rho_c)*((dw[j])**2. - self.dt)
                    G_aux += (0.5/np.sqrt(self.dt))*(gi_[j] - gi[j])*((dw[j])**2. - self.dt)
                else:
                    #G_aux += 0.5*opH_DH(MtC[i] - 1j*F[i], MtC[j] - 1j*F[j], rho_c)*dw[i]*dw[j]
                    G_aux += (0.5/np.sqrt(self.dt))*(gi_[j] - gi[j])*dw[i]*dw[j]
            
        return fi*self.dt + G_aux + G_w
    ######################################################################################
    ## Heun method
    ######################################################################################
    def feedRhoHeunStep(self, it, rho_c, dz):
        f = self.F(self.t0 + self.dt*it, rho_c) 
        dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz), np.imag(dz)]).flatten())

        gi = self.feedRhoHeun_g(f, it, rho_c)
        fi = self.feedRhoHeunf_strat(f, it, rho_c)
        
        G = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
        for i in range(2*self.num_op):
            G += dw[i]*gi[i]
        
        gi_= self.feedRhoHeun_g(f, it, rho_c + G)

        G = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
        for i in range(self.num_op):
            G += 0.5*dw[i]*(gi[i] + gi_[i])
            G += 0.5*dw[i + self.num_op]*(gi[i + self.num_op] + gi_[i + self.num_op])
        return fi*self.dt + G

    def feedRhoHeunf_strat(self, f, it, rho_c):
        c = self.cList(self.t0 + self.dt*it)
        MtC = self.MtC(self.t0 + self.dt*it)
        
        commu1 = -1j*opCom(self.H(self.t0 + self.dt*it), rho_c)
        for i in range(2*self.num_op):
            aux1 = np.dot(MtC[i],rho_c)
            aux2 = np.dot(rho_c,np.conjugate(np.transpose(MtC[i])))
            commu1 += -1j*opCom(f[i], aux1 + aux2)
                      
        Dc = opD(c, rho_c)
        Df = opD(f, rho_c)
   
        f_ito = commu1 + Dc + Df
        strat_corr = np.zeros((self.dimH, self.dimH), dtype = np.complex128, order='C')
        for i in range(2*self.num_op):
            strat_corr += opH_DH(MtC[i] - 1j*f[i], MtC[i] - 1j*f[i], rho_c)
        return f_ito - 0.5*strat_corr
    
    def feedRhoHeun_g(self, f, it, rho_c):
        HW = []
        MtC = self.MtC(self.t0 + self.dt*it)
        for i in range(2*self.num_op):
            A1 = MtC[i] - 1j*f[i]
            HW.append(opH(A1, rho_c)) 
        return HW

    ######################################################################################
    ## Individual trajectory functions 
    ######################################################################################
    def feedRhoTrajectory(self, seed_nn, method):
        rho = self.initialRho
        rho_T = [rho]
        if self.nonfixedUnraveling == 0:
            dz = self.dZeta(seed_nn)
            # Heun
            if method == 0:
                for it in range(1, self.maxiter):
                    rho = rho + self.feedRhoHeunStep(it, rho, dz[it])
                    #rho = (1./np.linalg.norm(rho))*rho
                    rho_T.append(rho)
            # Euler
            elif method == 1:
                for it in range(1, self.maxiter):
                    rho = rho + self.feedRhoEulerStep(it, rho, dz[it])
                    #rho = (1./np.linalg.norm(rho))*rho
                    rho_T.append(rho)
            # Milstein
            elif method == 2:
                for it in range(1, self.maxiter):
                    rho = rho + self.feedRhoMilsteinStep(it, rho, dz[it])
                    #trace = np.trace(rho)
                    #rho = (1./trace)*rho
                    rho_T.append(rho)
            return rho_T
        elif self.nonfixedUnraveling == 1:
            # Heun
            if method == 0:
                for it in range(1, self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, rho)
                    dz = self.dZeta_nf(self.U_rep, seed_nn + it)
                    rho = rho + self.feedRhoHeunStep(it, rho, dz)
                    #rho = (1./np.linalg.norm(rho))*rho
                    rho_T.append(rho)
            # Euler
            elif method == 1:
                for it in range(1, self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, rho)
                    dz = self.dZeta_nf(self.U_rep, seed_nn + it)
                    rho = rho + self.feedRhoEulerStep(it, rho, dz)
                    #rho = (1./np.linalg.norm(rho))*rho
                    rho_T.append(rho)
            # Milstein
            elif method == 2:
                for it in range(1, self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, rho)
                    dz = self.dZeta_nf(self.U_rep, seed_nn + it)
                    rho = rho + self.feedRhoMilsteinStep(it, rho, dz)
                    #trace = np.trace(rho)
                    #rho = (1./trace)*rho
                    rho_T.append(rho)
            return rho_T
    ######################################################################################
    ## Trajectory emsemble functions 
    ######################################################################################
    def feedRhoAverage(self, n_trajectories, method = 'euler', parallelfor = True):
        if method == "heun":
            Me = 0
        elif method == "euler":
            Me = 1
        elif method == "milstein":
            Me = 2
        else:
            Me = 1
            #print("Method not supported. Euler selected")
            
        m = []
        if parallelfor == True:
            
            rho_f = np.zeros((self.maxiter, self.dimH, self.dimH), dtype = np.complex128) 
            cpu_cores = 5*cpu_count()
            res = n_trajectories%cpu_cores
            A = np.rint(100000*np.random.rand(n_trajectories)).astype(int)
            B = A[0:n_trajectories-res]
            B = B.reshape(int((n_trajectories-res)/cpu_cores), cpu_cores)
            
            R = [B[i] for i in range(int((n_trajectories-res)/cpu_cores))]
            R.append(A[n_trajectories-res:n_trajectories])
            
            for seg in R:
                m = self.segmented_parallel_run(partial(self.feedRhoTrajectory, method = Me), seg)
                ##################################################################

                ##################################################################
                for i in range(self.maxiter):
                    rho_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                    for n in range(len(seg)):
                        rho_i += (1./n_trajectories)*m[n][i]     
                    rho_f[i] += rho_i
                ##################################################################
                del m
            return rho_f
        else:
            for n in range(n_trajectories):
                m.append(self.feedRhoTrajectory(n, Me))
            rho_f = []           
            for i in range(self.maxiter):
                rho_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                for n in range(n_trajectories):
                    rho_i += (1./n_trajectories)*m[n][i]
                rho_f.append(rho_i)
            return rho_f
    
    def feedRhoEnsemble(self, n_trajectories, method = 'euler', parallelfor = True):
        if method == "heun":
            Me = 0
        elif method == "euler":
            Me = 1
        elif method == "milstein":
            Me = 2
        else:
            Me = 1
            #print("Method not supported. Euler selected")
            
        m = []
        if parallelfor == True:
            
            m = self.parallel_run(partial(self.feedRhoTrajectory, method = Me), n_trajectories)
        else:
            for n in range(n_trajectories):
                m.append(self.feedRhoTrajectory(n, Me))
        return m
    ######################################################################################
    ## Analitical unconditional evolution functions 
    ######################################################################################
    def feed_F0_op_efficiency(self, rho_c, t):              
        commu1 = -1j*opCom(self.H(t) + self.H_feed, rho_c)
        D_eta =  opD(self.sqrt_eta_F0, rho_c)
        Dcf = opD(self.ciMF0, rho_c)
        return commu1 + Dcf + D_eta
    
    def feed_F0_op(self, rho_c, t):              
        commu1 = -1j*opCom(self.H(t) + self.H_feed, rho_c)
        Dcf = opD(self.ciMF0, rho_c)
        return commu1 + Dcf
    
    def feedUnconditional_op(self, rho_c, t):
        c = self.cList(t)
        f = self.F(t, rho_c)
        M = self.M_rep
        Mf = []
        Mt_aux = np.transpose(np.conjugate(M))
        fMt = []
        
        for i in range(self.num_op):
            fMt_i = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
            Mf_i = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
            for j in range(2*self.num_op):
                fMt_i += Mt_aux[j,i]*f[j]
                Mf_i += M[i,j]*f[j]
            fMt.append(fMt_i.dot(c[i]))
            Mf.append(np.transpose(np.conjugate(c[i])).dot(Mf_i))
        
        cMf = []
        for i in range(self.num_op):
            Mf_i = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
            for j in range(2*self.num_op):
                Mf_i += M[i][j]*f[j]
            cMf.append(c[i] - 1j*Mf_i)
              
        Hfe = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
        for i in range(self.num_op):
            Hfe += 0.5*(Mf[i] + fMt[i])
        commu1 = -1j*opCom(self.H(t) + Hfe, rho_c)
        Sf = []
        for i in range(2*self.num_op):
            Sf_i = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
            for j in range(2*self.num_op):
                Sf_i += self.sqrt_eta[i][j]*f[j]
            Sf.append(Sf_i)
        
        DSf = opD(Sf, rho_c)
        Dcf = opD(cMf, rho_c)
        return commu1 + Dcf + DSf
    
    ##################################################################################
    ## Fourth order Runge-Kutta based 
    ##################################################################################
    def feedAnaliticalUncond_efficiency_RungeStep(self, t, dt, psi0):
        a = self.feed_F0_op_efficiency(psi0, t)
        b = self.feed_F0_op_efficiency(psi0 + 0.5*dt*a, t)
        c = self.feed_F0_op_efficiency(psi0 + 0.5*dt*b, t)
        d = self.feed_F0_op_efficiency(psi0 + dt*c, t)
        return psi0 + (1./6.)*dt*(a + 2.*b + 2.*c + d)

    def feedAnaliticalUncond_RungeStep(self, t, dt, psi0):
        a = self.feed_F0_op(psi0, t)
        b = self.feed_F0_op(psi0 + 0.5*dt*a, t)
        c = self.feed_F0_op(psi0 + 0.5*dt*b, t)
        d = self.feed_F0_op(psi0 + dt*c, t)
        return psi0 + (1./6.)*dt*(a + 2.*b + 2.*c + d)
 
    def feedAnaliticalUncondRunge(self, rungesteps, eff_ = 'no_unit', last_point = 0):
        rho0 = self.initialRho
        rho_T = rho0
        DT = (self.tmax - self.t0)/rungesteps
        if last_point == 1:
            rho_T = rho0
            if eff_ == 'unit':
                for m in range(rungesteps+1):
                    if np.round(np.linalg.norm(rho_T),3) != 1:
                        #print("Higher number of steps needed")
                        break
                    rho_T = self.feedAnaliticalUncond_RungeStep(self.t0 + DT*m, DT, (1./np.linalg.norm(rho_T))*rho_T)
                return (1./np.trace(rho_T))*rho_T
            elif eff_ == 'no_unit':
                for m in range(rungesteps+1):
                    if np.round(np.linalg.norm(rho_T),3) != 1:
                        #print("Higher number of steps needed")
                        break
                    rho_T = self.feedAnaliticalUncond_efficiency_RungeStep(self.t0 + DT*m, DT, (1./np.linalg.norm(rho_T))*rho_T)
                return (1./np.trace(rho_T))*rho_T 
        elif last_point == 0:
            rho_T = [rho0]
            rho_aux = rho0
            if eff_ == 'unit':
                for m in range(rungesteps+1):
                    if np.round(np.linalg.norm(rho_aux),3) != 1:
                        #print("Higher number of steps needed")
                        break
                    rho_aux = self.feedAnaliticalUncond_RungeStep(self.t0 + DT*m, DT, (1./np.linalg.norm(rho_aux))*rho_aux)
                    rho_T.append((1./np.trace(rho_aux))*rho_aux)
                return rho_T   
            elif eff_ == 'no_unit':
                for m in range(rungesteps+1):
                    if np.round(np.linalg.norm(rho_aux),3) != 1:
                        #print("Higher number of steps needed")
                        break
                    rho_aux = self.feedAnaliticalUncond_efficiency_RungeStep(self.t0 + DT*m, DT, (1./np.linalg.norm(rho_aux))*rho_aux)
                    rho_T.append((1./np.trace(rho_aux))*rho_aux)
                return rho_T 

    ###################################################################################
    ## Scipy integrator based 
    ###################################################################################
    def feedAnaliticalUncond(self, m = 0, rrtol = 1e-5, aatol = 1e-5, last_point = 0):
        rho0 = self.initialRho
        n = self.dimH       
        x0 = rho0.reshape(-1)                                       
        def odefun(t,x):
            rho = x.reshape([n,n])                                   # restore to matrix form
            dx= self.feedUnconditional_op(rho, t)             # perform matrix operations
            return dx.reshape(-1)                                # return 1-dimensional vector
        sol = 0
        if last_point == 0:
            if m == 0:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList,  method = 'BDF')
            elif m == 1:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList,  rtol = rrtol, atol = aatol)
            elif m == 2:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList, method = DOP853, rtol = rrtol, atol = aatol)
            elif m == 3:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList, method = 'BDF', rtol = rrtol, atol = aatol)
            gc.collect()
            rho_T = [sol.y[:,i].reshape([self.dimH,self.dimH]) for i in range(len(sol.t))] 
            return rho_T
        
        elif last_point == 1:
            if m == 0:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], method = 'BDF')
            elif m == 1:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], rtol = rrtol, atol = aatol)
            elif m == 2:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], method = DOP853, rtol = rrtol, atol = aatol)
            elif m == 3:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], method = 'BDF', rtol = rrtol, atol = aatol)
            gc.collect()
            rho_T = [sol.y[:,i].reshape([self.dimH,self.dimH]) for i in range(len(sol.t))] 
            return rho_T[0]

    def feedAnaliticalUncond_F0(self, m = 0, rrtol = 1e-5, aatol = 1e-5, eff_ = 'no_unit', last_point = 0):
        n = self.dimH
        rho0 = self.initialRho
        x0 = rho0.reshape(-1)                                       
        if eff_ == 'no_unit':
            def odefun(t,x):
                rho = x.reshape([n,n])                                   # restore to matrix form
                dx= self.feed_F0_op_efficiency(rho, t)             # perform matrix operations
                return dx.reshape(-1)                                # return 1-dimensional vector
        elif eff_ == 'unit':
            def odefun(t,x):
                rho = x.reshape([n,n])                                   # restore to matrix form
                dx= self.feed_F0_op(rho, t)             # perform matrix operations
                return dx.reshape(-1)                                # return 1-dimensional vector
        else:
            #print('Efficiency option not found')
            return 0
        sol = 0
        if last_point == 0:
            if m == 0:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList,  method = 'BDF')
            elif m == 1:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList,  rtol = rrtol, atol = aatol)
            elif m == 2:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList, method = DOP853, rtol = rrtol, atol = aatol)
            elif m == 3:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList, method = 'BDF', rtol = rrtol, atol = aatol)
            gc.collect()
            rho_T = [sol.y[:,i].reshape([self.dimH,self.dimH]) for i in range(len(sol.t))] 
            return rho_T

        elif last_point == 1:
            if m == 0:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], method = 'BDF')
            elif m == 1:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], rtol = rrtol, atol = aatol)
            elif m == 2:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], method = DOP853, rtol = rrtol, atol = aatol)
            elif m == 3:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], method = 'BDF', rtol = rrtol, atol = aatol)
            gc.collect()
            rho_T = [sol.y[:,i].reshape([self.dimH,self.dimH]) for i in range(len(sol.t))] 
            return rho_T[0]
    
    ##########################################################################################
    #####################################-Custom-#############################################
    ##########################################################################################

    ##########################################################################################
    ## Euler integrator 
    ##########################################################################################
    def customRhoEulerStep(self, it, rho_c, dz, L, G):
        dw = dz
        N = len(dw)
        G_i = G(self.t0 + it*self.dt, rho_c)
        L_ = L(self.t0 + it*self.dt, rho_c)

        A = np.zeros((self.dimH,self.dimH), dtype = np.complex128)

        for i in range(N):
            A += dw[i]*G_i[i]   
        
        return self.dt*L_ + A
    ##########################################################################################
    ## Milstein integrator 
    ##########################################################################################      
    def customRhoMilsteinStep(self, it, rho_c, dz, L, G):
        dw = dz
        N = len(dw)
        gi = G(self.t0 + it*self.dt, rho_c)
        fi = L(self.t0 + it*self.dt, rho_c)

        G_w = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
        G_aux = np.zeros((self.dimH, self.dimH), dtype = np.complex128)
                    
        for i in range(N):
            G_w += dw[i]*gi[i]
            gi_= G(self.t0 + it*self.dt, rho_c + fi*self.dt + np.sqrt(self.dt)*gi[i])
            for j in range(N):
                if i == j:
                    #G_aux += 0.5*opH_DH(MtC[i], MtC[j], rho_c)*(dw[i]*dw[j] - self.dt)
                    G_aux += (0.5/np.sqrt(self.dt))*(gi_[j] - gi[j])*((dw[j])**2. - self.dt)
                else:
                    #G_aux += 0.5*opH_DH(MtC[i], MtC[j], rho_c)*dw[i]*dw[j]
                    G_aux += (0.5/np.sqrt(self.dt))*(gi_[j] - gi[j])*dw[i]*dw[j]
            
        return fi*self.dt + G_aux + G_w

    ##########################################################################################
    ### Trajectory functions 
    ##########################################################################################    
    def customRhoTrajectory(self, seed_nn, L, G, method = 1):
        rho = self.initialRho
        rho_T = [rho]
        dz = self.dZ_gauss(seed_nn, len(G(0,rho)))
        # Euler
        if method == 1:
            for it in range(1, self.maxiter):
                rho = rho + self.customRhoEulerStep(it, rho, dz[it], L, G)
                rho = (1./np.linalg.norm(rho))*rho
                rho_T.append(rho)
        # Milstein
        elif method == 2:
            for it in range(1, self.maxiter):
                rho = rho + self.customRhoMilsteinStep(it, rho, dz[it], L, G)
                rho = (1./np.linalg.norm(rho))*rho
                rho_T.append(rho)
        return rho_T

    def customRhoAverage(self, n_trajectories, L, G, method = 'euler', parallelfor = True):
        if method == "euler":
            Me = 1
        elif method == "milstein":
            Me = 2
        else:
            Me = 1
            #print("Method not supported. Euler selected")
        
        m = []
        if parallelfor == True:
            
            rho_f = np.zeros((self.maxiter, self.dimH, self.dimH), dtype = np.complex128) 
            cpu_cores = 3*cpu_count()
            res = n_trajectories%cpu_cores
            A = np.rint(100000*np.random.rand(n_trajectories)).astype(int)
            B = A[0:n_trajectories-res]
            B = B.reshape(int((n_trajectories-res)/cpu_cores), cpu_cores)
            
            R = [B[i] for i in range(int((n_trajectories-res)/cpu_cores))]
            R.append(A[n_trajectories-res:n_trajectories])
            
            for seg in R:
                m = self.segmented_parallel_run(partial(self.customRhoTrajectory, method = Me, L = L, G = G), seg) 
                ##################################################################
                for i in range(self.maxiter):
                    rho_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                    for n in range(len(seg)):
                        rho_i += (1./n_trajectories)*m[n][i]     
                    rho_f[i] += rho_i
                ##################################################################
                del m
            return rho_f
        else:
            for n in range(1,n_trajectories+1):
                m.append(self.customRhoTrajectory(n, method = Me, L = L, G = G))
            rho_f = []           
            for i in range(self.maxiter):
                rho_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
                for n in range(n_trajectories):
                    rho_i += (1./n_trajectories)*m[n][i]
                rho_f.append(rho_i)
            return rho_f
    
    def customRhoEnsemble(self, n_trajectories, L, G, method = 'euler', parallelfor = True):
        if method == "euler":
            Me = 1
        elif method == "milstein":
            Me = 2
        else:
            Me = 1
            #print("Method not supported. Euler selected")

        m = []
        if parallelfor == True:
            
            m = self.parallel_run(partial(self.customRhoTrajectory, method = Me, L = L, G = G), n_trajectories) 
        else:
            for n in range(1,n_trajectories+1):
                m.append(self.customRhoTrajectory(n, method = Me, L = L, G = G))
        return m 

    def customAnalitical(self, L, m = 0, rrtol = 1e-5, aatol = 1e-5):
        n = self.dimH
        rho0 = self.initialRho
        x0 = rho0.reshape(-1) 

        def odefun(t,x):
            rho = x.reshape([n,n])
            dx= L(t, rho)
            return dx.reshape(-1)   
        ##################################################################
        sol = 0
        if m == 0:
            sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList,  method = 'BDF')
        elif m == 1:
            sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList,  rtol = rrtol, atol = aatol)
        elif m == 2:
            sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList, method = DOP853, rtol = rrtol, atol = aatol)
        elif m == 3:
            sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList, method = 'BDF', rtol = rrtol, atol = aatol)
        gc.collect()
        rho_T = [sol.y[:,i].reshape([self.dimH,self.dimH]) for i in range(len(sol.t))] 
        return rho_T
    ##########################################################################################
    ################################-Standard evolution-######################################
    ##########################################################################################

    def schrodingerAnalitical(self, m = 0, rrtol = 1e-5, aatol = 1e-5):
        if self.initial_counter == 1:
            raise ValueError('Initial mixed state defined')
        psi0 = self.jumpPsi0
        sol = 0
        if m == 0:
            sol = solve_ivp(self.standardSchrodinger_op, [self.timeList[0], self.timeList[-1]], psi0, t_eval=self.timeList,  method = 'BDF')
        elif m == 1:
            sol = solve_ivp(self.standardSchrodinger_op, [self.timeList[0], self.timeList[-1]], psi0, t_eval=self.timeList,  rtol = rrtol, atol = aatol)
        elif m == 2:
            sol = solve_ivp(self.standardSchrodinger_op, [self.timeList[0], self.timeList[-1]], psi0, t_eval=self.timeList, method = DOP853, rtol = rrtol, atol = aatol)
        elif m == 3:
            sol = solve_ivp(self.standardSchrodinger_op, [self.timeList[0], self.timeList[-1]], psi0, t_eval=self.timeList, method = 'BDF', rtol = rrtol, atol = aatol)
        gc.collect()
        return np.transpose(sol.y[:])

    ############################################################
    #*************** Lindblad evolution functions **************
    #############################################################
    def lindbladAnalitical(self, m = 0, rrtol = 1e-5, aatol = 1e-5, last_point = 0, gamma = False, inefficient = False):
        n = self.dimH
        rho0 = self.initialRho
        x0 = rho0.reshape(-1) 
        ##################################################################
        if gamma == True:
            def odefun(t,x):
                rho = x.reshape([n,n])                                   
                dx= self.standartlindbladSuperOp_gamma(rho, t)                   
                return dx.reshape(-1)   
        else:
            if inefficient == True:
                def odefun(t,x):
                    rho = x.reshape([n,n])
                    dx= self.standartlindbladSuperOp_inefficient(rho, t)
                    return dx.reshape(-1)  
            elif inefficient == False:
                def odefun(t,x):
                    rho = x.reshape([n,n])
                    dx= self.standartlindbladSuperOp(rho, t)
                    return dx.reshape(-1)   
        ##################################################################
        sol = 0
        if last_point == 0:
            if m == 0:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList,  method = 'BDF')
            elif m == 1:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList,  rtol = rrtol, atol = aatol)
            elif m == 2:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList, method = DOP853, rtol = rrtol, atol = aatol)
            elif m == 3:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList, method = 'BDF', rtol = rrtol, atol = aatol)
            gc.collect()
            rho_T = [sol.y[:,i].reshape([self.dimH,self.dimH]) for i in range(len(sol.t))] 
            return rho_T
        
        elif last_point == 1:
            if m == 0:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], method = 'BDF')
            elif m == 1:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], rtol = rrtol, atol = aatol)
            elif m == 2:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], method = DOP853, rtol = rrtol, atol = aatol)
            elif m == 3:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], method = 'BDF', rtol = rrtol, atol = aatol)
            gc.collect()
            rho_T = [sol.y[:,i].reshape([self.dimH,self.dimH]) for i in range(len(sol.t))] 
            return rho_T[0]

    def lindblad_gamma_RungeStep(self, t, dt, rho):
        a = self.standartlindbladSuperOp_gamma(rho, t)
        b = self.standartlindbladSuperOp_gamma(rho + 0.5*dt*a, t)
        c = self.standartlindbladSuperOp_gamma(rho + 0.5*dt*b, t)
        d = self.standartlindbladSuperOp_gamma(rho + dt*c, t)
        return rho + (1./6.)*dt*(a + 2.*b + 2.*c + d)
    
    def lindbladRungeStep(self, t, dt, rho):
        a = self.standartlindbladSuperOp(rho, t)
        b = self.standartlindbladSuperOp(rho + 0.5*dt*a, t)
        c = self.standartlindbladSuperOp(rho + 0.5*dt*b, t)
        d = self.standartlindbladSuperOp(rho + dt*c, t)
        return rho + (1./6.)*dt*(a + 2.*b + 2.*c + d)
    
    def lindbladAnaliticalRunge(self, rungesteps = 0, gamma = False):
        rho0 = self.initialRho
        rho_T = [rho0]
        rho_i = rho0
        if rungesteps == 0:
            rungesteps = self.maxiter
        DT = np.abs(self.timeList[-1]-self.timeList[0])/rungesteps
        if gamma == True:
            for it in range(1,rungesteps):
                rho_i = self.lindblad_gamma_RungeStep(self.t0 + it*DT, DT, rho_i)
                #rho_T = (1./np.trace(rho_T))*rho_T
                rho_T.append(rho_i)
            return rho_T
        else:
            for it in range(1,rungesteps):
                rho_i = self.lindbladRungeStep(self.t0 + it*DT, DT, rho_i)
                #rho_T = (1./np.trace(rho_T))*rho_T
                rho_T.append(rho_i)
            return rho_T
    ############################################################
    #************ Von Neumann evolution functions **************
    #############################################################
    def VonNeumannAnalitical(self, m = 0, rrtol = 1e-5, aatol = 1e-5, last_point = 0):
        n = self.dimH
        rho0 = self.initialRho
        x0 = rho0.reshape(-1)                                        # make data 1-dimensional
        def odefun(t,x):
            rho = x.reshape([n,n])                                   # restore to matrix form
            dx = self.standartLiouvilleOp(rho, t)                   # perform matrix operations
            return dx.reshape(-1)                                # return 1-dimensional vector
        sol = 0
        if last_point == 0:
            if m == 0:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList,  method = 'BDF')
            elif m == 1:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList,  rtol = rrtol, atol = aatol)
            elif m == 2:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList, method = DOP853, rtol = rrtol, atol = aatol)
            elif m == 3:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=self.timeList, method = 'BDF', rtol = rrtol, atol = aatol)
            gc.collect()
            rho_T = [sol.y[:,i].reshape([self.dimH,self.dimH]) for i in range(len(sol.t))] 
            return rho_T
        
        elif last_point == 1:
            if m == 0:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], method = 'BDF')
            elif m == 1:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], rtol = rrtol, atol = aatol)
            elif m == 2:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], method = DOP853, rtol = rrtol, atol = aatol)
            elif m == 3:
                sol = solve_ivp(odefun, [self.timeList[0], self.timeList[-1]], x0, t_eval=[self.timeList[-1]], method = 'BDF', rtol = rrtol, atol = aatol)
            gc.collect()
            rho_T = [sol.y[:,i].reshape([self.dimH,self.dimH]) for i in range(len(sol.t))] 
            return rho_T[0]
    
    def VonNeumannAnaliticalRungeStep(self, t, dt, psi0):
        a = self.standartLiouvilleOp(psi0, t)
        b = self.standartLiouvilleOp(psi0 + 0.5*dt*a, t)
        c = self.standartLiouvilleOp(psi0 + 0.5*dt*b, t)
        d = self.standartLiouvilleOp(psi0 + dt*c, t)
        return psi0 + (1./6.)*dt*(a + 2.*b + 2.*c + d)
    
    def VonNeumannAnaliticalRunge(self, rungesteps = 10, last_point = 0):
        rho0 = self.initialRho
        steps = rungesteps
        nn = 1
        t = self.t0
        Dt = self.tmax - self.t0
        if last_point == 1:
            rho_T = rho0
            while t < self.tmax:
                rho_T_aux = self.VonNeumannAnaliticalRungeStep(t, Dt/steps, (1./np.linalg.norm(rho_T))*rho_T)
                if np.round(np.linalg.norm(rho_T_aux),5) != 1:
                    steps += 0.5*((10)**nn)
                    nn += 1
                    continue
                rho_T = rho_T_aux
                t += Dt/steps
            return (1./np.trace(rho_T))*rho_T
        elif last_point == 0:
            rho_T = [rho0]
            rho_aux = rho0
            while t < self.tmax:
                rho_T_aux = self.VonNeumannAnaliticalRungeStep(t, Dt/steps, (1./np.linalg.norm(rho_aux))*rho_aux)
                if np.round(np.linalg.norm(rho_T_aux),5) != 1:
                    steps += 0.5*((10)**nn)
                    nn += 1
                    continue
                rho_aux = rho_T_aux
                t += Dt/steps
                rho_T.append((1./np.trace(rho_aux))*rho_aux)
            return rho_T

    ##########################################################################################
    ##################################-Miscellaneous-#########################################
    ##########################################################################################
    def quadrature_M(self, c):
        Mtc_Mct = []
        for i in range(2*self.num_op):
            mtc_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
            mct_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
            for j in range(self.num_op):
                mtc_i += np.conjugate(self.M_rep[j][i])*c[j]
                mct_i += self.M_rep[j][i]*np.transpose(np.conjugate(c[j]))
            Mtc_Mct.append(mct_i + mtc_i)
        return Mtc_Mct
    
    def quadrature_HY(self, c):
        Hc_Yct = []
        U = self.U_rep
        H = U[0:self.num_op, 0:self.num_op] + U[self.num_op:2*self.num_op, self.num_op:2*self.num_op]
        ReY = U[0:self.num_op, 0:self.num_op] - U[self.num_op:2*self.num_op, self.num_op:2*self.num_op]
        ImY = 2*U[0:self.num_op, self.num_op:2*self.num_op]
        Y = ReY + 1j*ImY
        for i in range(self.num_op):
            Hc_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
            Yct_i = np.zeros((self.dimH,self.dimH), dtype = np.complex128)
            for j in range(self.num_op):
                Hc_i += H[i,j]*c[j]
                Yct_i += Y[i,j]*np.transpose(np.conjugate(c[j]))
            Hc_Yct.append(Hc_i + Yct_i)
        return Hc_Yct

    # Real 2L-lenght current
    def current_Y(self, seed_nn, traj_type = 'density', method = 'euler'):
        rho = self.initialRho
        J = []
        if self.nonfixedUnraveling == 0:
            dz = self.dZeta(seed_nn)
            # Heun
            if method == 'heun':
                for it in range(1, self.maxiter):
                    dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz[it]), np.imag(dz[it])]).flatten())
                    rho = rho + self.diffusiveRhoHeunStep(it, rho, dz[it])
                    #rho = (1./np.linalg.norm(rho))*rho
                    J_it = []
                    c = self.cList(self.t0 + self.dt*it)
                    Q = self.quadrature_M(c)
                    for r in range(2*self.num_op):
                        jj_r = np.trace(np.dot(rho,Q[r]))+ dw[r]/self.dt
                        J_it.append(jj_r)
                    J.append(J_it)
            # Euler
            elif method == 'euler':
                for it in range(1, self.maxiter):
                    dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz[it]), np.imag(dz[it])]).flatten())
                    rho = rho + self.diffusiveRhoEulerStep(it, rho, dz[it])
                    #rho = (1./np.linalg.norm(rho))*rho
                    J_it = []
                    c = self.cList(self.t0 + self.dt*it)
                    Q = self.quadrature_M(c)
                    for r in range(2*self.num_op):
                        jj_r = np.trace(np.dot(rho,Q[r]))+ dw[r]/self.dt
                        J_it.append(jj_r)
                    J.append(J_it)
            # Milstein
            elif method == 'milstein':
                for it in range(1, self.maxiter):
                    dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz[it]), np.imag(dz[it])]).flatten())
                    rho = rho + self.diffusiveRhoMilsteinStep(it, rho, dz[it])
                    #rho = (1./np.linalg.norm(rho))*rho
                    J_it = []
                    c = self.cList(self.t0 + self.dt*it)
                    Q = self.quadrature_M(c)
                    for r in range(2*self.num_op):
                        jj_r = np.trace(np.dot(rho,Q[r]))+ dw[r]/self.dt
                        J_it.append(jj_r)
                    J.append(J_it)

        elif self.nonfixedUnraveling == 1:
            # Heun
            if method == 'heun':
                for it in range(1, self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, rho)
                    dz = self.dZeta_nf(self.U_rep, seed_nn + it)
                    dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz), np.imag(dz)]).flatten())
                    rho = rho + self.diffusiveRhoHeunStep(it, rho, dz)
                    #rho = (1./np.linalg.norm(rho))*rho
                    J_it = []
                    c = self.cList(self.t0 + self.dt*it)
                    Q = self.quadrature_M(c)
                    for r in range(2*self.num_op):
                        jj_r = np.trace(np.dot(rho, Q[r]))+ dw[r]/self.dt
                        J_it.append(jj_r)
                    J.append(J_it)
                    
            # Euler
            elif method == 'euler':
                for it in range(1, self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, rho)
                    dz = self.dZeta_nf(self.U_rep, seed_nn + it)
                    dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz), np.imag(dz)]).flatten())
                    rho = rho + self.diffusiveRhoEulerStep(it, rho, dz)
                    #rho = (1./np.linalg.norm(rho))*rho
                    J_it = []
                    c = self.cList(self.t0 + self.dt*it)
                    Q = self.quadrature_M(c)
                    for r in range(2*self.num_op):
                        jj_r = np.trace(np.dot(rho,Q[r]))+ dw[r]/self.dt
                        J_it.append(jj_r)
                    J.append(J_it)
            # Milstein
            elif method == 'milstein':
                for it in range(1, self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, rho)
                    dz = self.dZeta_nf(self.U_rep, seed_nn + it)
                    dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz), np.imag(dz)]).flatten())
                    rho = rho + self.diffusiveRhoMilsteinStep(it, rho, dz)
                    #rho = (1./np.linalg.norm(rho))*rho
                    J_it = []
                    c = self.cList(self.t0 + self.dt*it)
                    Q = self.quadrature_M(c)
                    for r in range(2*self.num_op):
                        jj_r = np.trace(np.dot(rho,Q[r]))+ dw[r]/self.dt
                        J_it.append(jj_r)
                    J.append(J_it)
        return J
    
    ## Imaginary L-lenght current
    def current_J(self, seed_nn, traj_type = 'density', method = 'euler'):
        rho = self.initialRho
        J = []
        if self.nonfixedUnraveling == 0:
            dz = self.dZeta(seed_nn)
            # Heun
            if method == 'heun':
                for it in range(1, self.maxiter):
                    rho = rho + self.diffusiveRhoHeunStep(it, rho, dz[it])
                    #rho = (1./np.linalg.norm(rho))*rho
                    J_it = []
                    c = self.cList(self.t0 + self.dt*it)
                    Q = self.quadrature_HY(c)
                    for r in range(self.num_op):
                        jj_r = np.trace(np.dot(rho,Q[r]))+ dz[it,r]/self.dt
                        J_it.append(jj_r)
                    J.append(J_it)
            # Euler
            elif method == 'euler':
                for it in range(1, self.maxiter):
                    dw = np.linalg.pinv(self.T_rep).dot(np.array([np.real(dz[it]), np.imag(dz[it])]).flatten())
                    rho = rho + self.diffusiveRhoEulerStep(it, rho, dz[it])
                    #rho = (1./np.linalg.norm(rho))*rho
                    J_it = []
                    c = self.cList(self.t0 + self.dt*it)
                    Q = self.quadrature_HY(c)
                    for r in range(self.num_op):
                        jj_r = np.trace(np.dot(rho,Q[r]))+ dz[it,r]/self.dt
                        J_it.append(jj_r)
                    J.append(J_it)
            # Milstein
            elif method == 'milstein':
                for it in range(1, self.maxiter):
                    rho = rho + self.diffusiveRhoMilsteinStep(it, rho, dz[it])
                    #rho = (1./np.linalg.norm(rho))*rho
                    J_it = []
                    c = self.cList(self.t0 + self.dt*it)
                    Q = self.quadrature_HY(c)
                    for r in range(self.num_op):
                        jj_r = np.trace(np.dot(rho,Q[r]))+ dz[it,r]/self.dt
                        J_it.append(jj_r)
                    J.append(J_it)

        elif self.nonfixedUnraveling == 1:
            # Heun
            if method == 'heun':
                for it in range(1, self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, rho)
                    dz = self.dZeta_nf(self.U_rep, seed_nn + it)
                    rho = rho + self.diffusiveRhoHeunStep(it, rho, dz)
                    #rho = (1./np.linalg.norm(rho))*rho
                    J_it = []
                    c = self.cList(self.t0 + self.dt*it)
                    Q = self.quadrature_HY(c)
                    for r in range(self.num_op):
                        jj_r = np.trace(np.dot(rho,Q[r]))+ dz[r]/self.dt
                        J_it.append(jj_r)
                    J.append(J_it)
                    
            # Euler
            elif method == 'euler':
                for it in range(1, self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, rho)
                    dz = self.dZeta_nf(self.U_rep, seed_nn + it)
                    rho = rho + self.diffusiveRhoEulerStep(it, rho, dz)
                    #rho = (1./np.linalg.norm(rho))*rho
                    J_it = []
                    c = self.cList(self.t0 + self.dt*it)
                    Q = self.quadrature_HY(c)
                    for r in range(self.num_op):
                        jj_r = np.trace(np.dot(rho,Q[r]))+ dz[r]/self.dt
                        J_it.append(jj_r)
                    J.append(J_it)
            # Milstein
            elif method == 'milstein':
                for it in range(1, self.maxiter):
                    self.update_defintions(self.t0 + it*self.dt, rho)
                    dz = self.dZeta_nf(self.U_rep, seed_nn + it)
                    rho = rho + self.diffusiveRhoMilsteinStep(it, rho, dz)
                    #rho = (1./np.linalg.norm(rho))*rho
                    J_it = []
                    c = self.cList(self.t0 + self.dt*it)
                    Q = self.quadrature_HY(c)
                    for r in range(self.num_op):
                        jj_r = np.trace(np.dot(rho,Q[r]))+ dz[r]/self.dt
                        J_it.append(jj_r)
                    J.append(J_it)
        return J
    
    def parallel_run(self, fun, n_trajectories):
        try:
            p = Pool(processes = cpu_count())
            m = p.map(fun, np.rint(100000*np.random.rand(n_trajectories)).astype(int))
            p.terminate()
            p.join() 
            return m
        except KeyboardInterrupt as e:
            p.terminate()
            p.join()
            raise e
            
    def segmented_parallel_run(self, fun, array):
        try:
            p = Pool(processes = cpu_count())
            m = p.map(fun, array)
            p.terminate()
            p.join() 
            return m
        except KeyboardInterrupt as e:
            p.terminate()
            p.join()
            raise e
    
    def mod_fun_dummy(self, fun, t, rho):
        return fun
    def mod_fun0(self, fun, t, rho):
        return fun()
    def mod_fun1(self, fun, t, rho):
        return fun(t)
    def mod_fun2(self, fun, t, rho):
        return fun(t, rho)

    def update_defintions_uH(self, t, rho):
        uMatrix = self.uMatrix(t, rho)
        HMatrix = self.HMatrix(t, rho)
        oMatrix = self.oMatrix
        self.U_rep = 0.5*np.reshape(np.block( [[HMatrix + np.real(uMatrix), np.imag(uMatrix) ],[np.imag(uMatrix), HMatrix - np.real(uMatrix) ]]), (2*self.num_op,2*self.num_op))
        self.T_rep = np.round(sqrtm(self.U_rep).dot(oMatrix),8)
        T1 = self.T_rep[0:self.num_op,:]
        T2 = self.T_rep[self.num_op:2*self.num_op,:]
        self.M_rep = np.round(T1 + 1j*T2,7)
        ###########################################################
        self.Mt_aux = np.conjugate(np.transpose(self.M_rep))
        self.MMT = np.round(np.asmatrix(self.M_rep).dot(np.conjugate(np.transpose(np.asmatrix(self.M_rep)))),6)
        self.sqrt_MMT = sqrtm(np.identity(self.num_op) - self.MMT)

    def update_defintions_u(self, t, rho):
        uMatrix = self.uMatrix(t, rho)
        HMatrix = self.HMatrix
        oMatrix = self.oMatrix
        self.U_rep = 0.5*np.reshape(np.block( [[HMatrix + np.real(uMatrix), np.imag(uMatrix) ],[np.imag(uMatrix), HMatrix - np.real(uMatrix) ]]), (2*self.num_op,2*self.num_op))
        self.T_rep = np.round(sqrtm(self.U_rep).dot(oMatrix),8)
        T1 = self.T_rep[0:self.num_op,:]
        T2 = self.T_rep[self.num_op:2*self.num_op,:]
        self.M_rep = np.round(T1 + 1j*T2,7)
        ###########################################################
        self.Mt_aux = np.conjugate(np.transpose(self.M_rep))
        self.MMT = np.round(np.asmatrix(self.M_rep).dot(np.conjugate(np.transpose(np.asmatrix(self.M_rep)))),6)
        self.sqrt_MMT = sqrtm(np.identity(self.num_op) - self.MMT)

    def update_defintions_H(self, t, rho):
        uMatrix = self.uMatrix
        HMatrix = self.HMatrix(t, rho)
        oMatrix = self.oMatrix
        self.U_rep = 0.5*np.reshape(np.block( [[HMatrix + np.real(uMatrix), np.imag(uMatrix) ],[np.imag(uMatrix), HMatrix - np.real(uMatrix) ]]), (2*self.num_op,2*self.num_op))
        self.T_rep = np.round(sqrtm(self.U_rep).dot(oMatrix),8)
        T1 = self.T_rep[0:self.num_op,:]
        T2 = self.T_rep[self.num_op:2*self.num_op,:]
        self.M_rep = np.round(T1 + 1j*T2,7)
        ###########################################################
        self.Mt_aux = np.conjugate(np.transpose(self.M_rep))
        self.MMT = np.round(np.asmatrix(self.M_rep).dot(np.conjugate(np.transpose(np.asmatrix(self.M_rep)))),6)
        self.sqrt_MMT = sqrtm(np.identity(self.num_op) - self.MMT)

    def update_defintions_M(self, t, rho):
        M_rep = self.mMatrix(t, rho)
        self.M_rep = M_rep
        T_rep = np.array([np.real(M_rep),np.imag(M_rep)]).reshape([2*self.num_op,2*self.num_op])
        self.T_rep = T_rep
        self.U_rep = np.asmatrix(T_rep).dot(np.transpose(np.asmatrix(T_rep)))
        ###########################################################
        self.Mt_aux = np.conjugate(np.transpose(self.M_rep))
        self.MMT = np.round(np.asmatrix(self.M_rep).dot(np.conjugate(np.transpose(np.asmatrix(self.M_rep)))),6)
        self.sqrt_MMT = sqrtm(np.identity(self.num_op) - self.MMT)

    def standardSchrodinger_op(self, t, psi):
        return np.dot(-1j*self.H(t), psi)
    
    def standartlindbladSuperOp(self, rho, t):
        c = self.cList(t)
        commu1 = -1j*opCom(self.H(t), rho)
        Dc = opD(c, rho)
        return commu1 + Dc

    def standartlindbladSuperOp_inefficient(self, rho, t):
        c = self.cList(t)
        d = []
        commu1 = -1j*opCom(self.H(t), rho)
        for i in range(self.num_op):
            d.append(np.sqrt(self.eta_diag[i])*c[i])
        Dc = opD(d, rho)
        return commu1 + Dc

    def standartlindbladSuperOp_gamma(self, rho, t):
        c = self.cList(t)
        MtC = self.MtC(int((t-self.t0)/self.dt))
        
        H_ = self.H(t)
        for i in range(2*self.num_op):
            H_ += 1j*0.5*(np.conjugate(self.default_amp[i])*MtC[i] - self.default_amp[i]*np.transpose(np.conjugate(MtC[i])))
            
        commu1 = -1j*opCom(H_, rho)
        Dc = opD(c, rho)
        return commu1 + Dc
    
    def standartLiouvilleOp(self, rho, t):
        commu1 = -1j*opCom(self.H(t), rho)
        return commu1   
    
##########################################################################################
########################################################################################## 

class DOP853(DOP853_):
    def _estimate_error_norm(self, K, h, scale):
        err5 = np.dot(K.T, self.E5) / scale
        err3 = np.dot(K.T, self.E3) / scale

        # fixes bug in scipy 1.4.1
        from scipy.integrate._ivp.common import norm
        err5_norm_2 = norm(err5)
        err3_norm_2 = norm(err3)
        denom = err5_norm_2 + 0.01 * err3_norm_2
        return np.abs(h) * err5_norm_2 / np.sqrt(denom * len(scale))

def condition_check(U_rep, M_rep):
    inefficient = False
    num_op = len(M_rep)
    MMT = np.round(np.asmatrix(M_rep).dot(np.conjugate(np.transpose(np.asmatrix(M_rep)))),6)

    diagMMT = np.zeros((num_op,num_op), dtype = np.complex128)
    diagVec = np.zeros(num_op, dtype = np.complex128)
    for i in range(num_op):
        diagMMT[i,i] = MMT[i,i]
        diagVec[i] = MMT[i,i]
    eta_diag = diagVec
    if ((MMT - diagMMT)!= 0).any():
        print('MM^dag=',repr(MMT))
        raise ValueError('Invalid M matrix. Remember that M must be defined in such a way to ensure M.M^dag to be diagonal')

    if (np.round(np.real(diagMMT),10) > 1).any() or (np.round(np.real(diagMMT),8) < 0).any() or (np.round(np.imag(diagMMT),8) != 0).any():
        print('MM^dag=',repr(MMT))
        raise ValueError('Invalid M matrix. Remember that M must be defined in such a way to ensure 0 <= M.M^dag <= 1')
    elif (np.round(np.real(diagVec),8) < 1).any():
        inefficient = True

    U11 = np.round(U_rep[0:num_op,0:num_op],10)
    U22 = np.round(U_rep[num_op:2*num_op,num_op:2*num_op],10)
    UtU = U11 + U22

    diagUtU = np.zeros((num_op,num_op), dtype = np.complex128)
    for i in range(num_op):
        diagUtU[i,i] = UtU[i,i]
    
    if ((UtU - diagUtU)!= 0).any():
        print(UtU)
        raise ValueError('Invalid U matrix. Remember that U must be defined in such a way to ensure U11 + U22 to be diagonal')

    if (np.round(np.real(diagUtU),10) > 1).any() or (np.round(np.real(diagUtU),8) < 0).any() or (np.round(np.imag(diagUtU),8) != 0).any():
        raise ValueError('Invalid U matrix. Remember that U must be defined in such a way to ensure 0 <= U11 + U22 <= 1')
    return inefficient, eta_diag

def representation(num_op, mMatrix, uMatrix, HMatrix, oMatrix):
    #######################################################################     
    ## Unraveling parametrization / Orthogonal matrix taken as the identity 
    ######################################################################
    ### O Matrix
    if oMatrix == []:
        O = np.eye(2*num_op)
    else:
        O = oMatrix
        
    ### H matrix
    if len(HMatrix) == 0:
        HMatrix = np.eye(num_op)  
    
    if len(uMatrix)!= 0 and len(mMatrix)!= 0:
        raise ValueError('Both U and M representation matrices are defined, this could lead to errors. Please just define one.')

    if len(uMatrix) == 0 and len(mMatrix) == 0:
        h_matrix = np.eye(num_op)
        u_matrix = np.eye(num_op)
        U_rep = 0.5*np.reshape(np.block( [[h_matrix + np.real(u_matrix), np.imag(u_matrix) ],[np.imag(u_matrix), h_matrix - np.real(u_matrix) ]]), (2*num_op,2*num_op))
        T_rep = np.round(sqrtm(U_rep).dot(O),8)
        T1 = T_rep[0:num_op,:]
        T2 = T_rep[num_op:2*num_op,:]
        M_rep = np.round(T1 + 1j*T2,7)

    elif len(uMatrix) == 0 and len(mMatrix) != 0:
        if np.shape(mMatrix) != (num_op,num_op*2):
            raise ValueError("Wrong M-Matrix dimension. Remember that M must have dimension (L,2L)")

        M_rep = mMatrix
        T_rep = np.array([np.real(M_rep),np.imag(M_rep)]).reshape([2*num_op,2*num_op])
        U_rep = np.asmatrix(T_rep).dot(np.transpose(np.asmatrix(T_rep)))
    else:
        if np.shape(uMatrix) != (num_op,num_op):
            raise ValueError("Wrong u-Matrix dimension. Remember that u must have dimension (L,L)")
        U_rep = 0.5*np.reshape(np.block( [[HMatrix + np.real(uMatrix), np.imag(uMatrix) ],[np.imag(uMatrix), HMatrix - np.real(uMatrix) ]]), (2*num_op,2*num_op))
        T_rep = np.round(sqrtm(U_rep).dot(O),8)
        T1 = T_rep[0:num_op,:]
        T2 = T_rep[num_op:2*num_op,:]
        M_rep = np.round(T1 + 1j*T2,7)
    return U_rep, M_rep, T_rep

### Misc functions
def rhoBlochrep_data(rho):
    t_steps = len(rho)
    rx = []
    ry = []
    rz = []
    for i in range(t_steps):
        rx.append(np.real(np.trace(rho[i].dot(np.array([[0,1],[1,0]])))))
        ry.append(np.real(np.trace(rho[i].dot(np.array([[0,-1j],[1j,0]])))))
        rz.append(np.real(np.trace(rho[i].dot(np.array([[1,0],[0,-1]])))))
    return [rx,ry,rz]

def fidelity(rho1, rho2):
    srho1 = sqrtm(rho1)
    return np.real(np.trace(sqrtm(srho1.dot(rho2.dot(srho1)))))

def opCom(a,b):
    return np.dot(a,b) - np.dot(b,a)

def opAntiCom(a,b):
    return np.dot(a,b) + np.dot(b,a)

def opD(A, B):
    dimH = len(B)
    D = np.zeros((dimH, dimH), dtype = np.complex128, order='C')
    NN = len(A)
    for i in range(NN):
        AT = np.transpose(np.conjugate(A[i]))
        D += np.dot(A[i],np.dot(B,AT)) - 0.5*opAntiCom(np.dot(AT,A[i]),B)
    return D

def opH(A, B):
    Aux = np.dot(A,B) + np.dot(B,np.transpose(np.conjugate(A)))
    return Aux - np.trace(Aux)*B

def opg_Dg_psi(CA, CB, QA, QB, psi):
    gA = np.dot(CA,psi) - 0.5*QA*psi
    A1 = np.dot(CB,gA)
    A2 = -0.5*(-1*np.dot(np.dot(gA,QB),psi)*psi + np.dot(np.dot(np.conjugate(psi),QB),gA)*psi + np.dot(np.dot(np.conjugate(psi),QB),psi)*gA)
    
    return A1 + A2

def opH_DH(CA, CB, rho):
    A1 = np.dot(np.dot(CB,CA),rho) + np.dot(np.dot(CB,rho),np.transpose(np.conjugate(CA))) + np.dot(np.dot(CA,rho),np.transpose(np.conjugate(CB))) + np.dot(np.dot(rho,np.transpose(np.conjugate(CA))),np.transpose(np.conjugate(CB)))
    A2 = np.dot(CA,rho) + np.dot(rho,np.conjugate(np.transpose(CA)))
    A3 = np.dot(CB,rho) + np.dot(rho,np.conjugate(np.transpose(CB)))
    return A1 - np.trace(A2)*A3 - np.trace(A3)*A2 + 2*np.trace(A2)*np.trace(A3)*rho - rho*np.trace(A1)

def opG(A, B):
    Aux = np.dot(np.dot(A,B), np.transpose(np.conjugate(A)))
    return (1/np.trace(Aux))*Aux - B

def op_expect(self, op_, psi):
    expect = np.conjugate(psi).dot(op_.dot(psi))
    return expect