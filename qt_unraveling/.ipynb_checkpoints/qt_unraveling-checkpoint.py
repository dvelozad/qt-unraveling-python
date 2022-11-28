'''
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
'''
import numpy as np
from scipy.linalg import sqrtm
from numba import jit, objmode
import warnings

from functools import partial

import misc_func as misc
from usual_operators_ import *

## Trajectory modules
from diffusive_trajectory import *
from feedback_trajectory import *
from jumpy_trajectory import *

class System:
    def __init__(self, drivingH, initialState, timeList, *, lindbladList = [], FList = [], uMatrix = [], mMatrix = [], oMatrix = [], HMatrix = [], TMatrix = [], WMatrix = [], PhiMatrix = []):
        #########################################################
        #### Time interval definitions
        #########################################################
        ## Time span
        self.timeList = timeList
        self.t0 = timeList[0]

        ## Time interval
        self.tmax = timeList[-1]
        self.maxiter = np.shape(timeList)[0]
        self.dt = abs(timeList[1]-timeList[0])
        
        #########################################################
        #### Initial state definitions
        #########################################################
        ## System dimension
        self.dimH = np.shape(initialState)[0]

        ## Check initial state type: vector or densisty matrix 
        if len(np.shape(initialState)) == 1:
            ## Intial state type 0 if ket, 1 if density matrix
            self.initial_state_type = 0

            # In case the intial state isn't normalized
            if np.round(np.linalg.norm(initialState),5) != 1:
                warnings.warn('Initial state is unnormalized. Normalized state taken instead')
                self.initialStatePsi = (1./np.linalg.norm(initialState))*initialState
            else:
                self.initialStatePsi = initialState  

            ## Density matrix
            self.initialStateRho = np.asarray(np.transpose(np.asmatrix(self.initialStatePsi)).dot(np.conjugate(np.asmatrix(self.initialStatePsi))))   
                                         
        elif len(np.shape(initialState)) == 2:
            ## Intial state type 0 if ket, 1 if density matrix
            self.initial_state_type = 1
            self.initialStateRho = np.asarray(initialState)

        #########################################################
        #### Hamiltonian definitions
        #########################################################
        ## Check if drivingH is a list or a function
        if not (type(drivingH).__name__  in ['ndarray', 'CPUDispatcher']):
            raise ValueError('System Hamiltonian must be passed as a jitted function or ndarray of dtype complex128')

        ## If it is not a function with these characteristics we redefine them as such
        self.timedepent_hamiltonian = False
        if (type(drivingH).__name__ == 'ndarray'):
            if not (drivingH.dtype.name == 'complex128'):
                raise ValueError('System Hamiltonian must be passed as a contiguousarray ndarray of dtype complex128')
            else:
                ## Define a numba compatible function for the Hamiltonian
                @jit(nopython=True)
                def Hamiltonian(t, drivingH_=drivingH):
                    return drivingH_
                self.H = Hamiltonian
                self.drivingH = drivingH
        else:
            if not (drivingH(0).dtype.name == 'complex128'):
                raise ValueError('System Hamiltonian must be passed as a function returning a contiguousarray ndarray of dtype complex128')
            else:
                self.timedepent_hamiltonian = True
                self.H = drivingH

        ######################################################################
        ###################### M and U definitions ###########################
        ######################################################################
        ## Number of operators
        if type(lindbladList).__name__ in ['ndarray']:
            self.num_op = np.shape(lindbladList)[0]
        else:
            self.num_op = np.shape(lindbladList(0))[0]

        ## Adaptative unraveling flag
        self.nonfixedUnraveling = False

        ## Auxiliar definitions
        self.coherent_fields = np.zeros(2*self.num_op, dtype=np.complex128)

        mMatrix_suprt = mMatrix
        uMatrix_suprt = uMatrix
        HMatrix_suprt = HMatrix
        oMatrix_suprt = oMatrix
        TMatrix_suprt = TMatrix
        WMatrix_suprt = WMatrix
        PhiMatrix_suprt = PhiMatrix
        self.mMatrix = mMatrix
        self.uMatrix = uMatrix

        ## Orthogonal degrees of freedom
        if oMatrix == []:
            self.oMatrix = np.eye(2*self.num_op, dtype=np.complex128)
            oMatrix = np.eye(2*self.num_op, dtype=np.complex128)
        else:
            self.oMatrix = oMatrix
        
        ## In case at least one representation is passed as a function
        if (not uMatrix == []) ^ (not mMatrix == []):
            ## Wiseman's rep effiiency H matrix
            if type(HMatrix).__name__ not in ['CPUDispatcher']:
                if HMatrix == []:
                    self.HMatrix = np.eye(self.num_op, dtype=np.complex128)  
                    HMatrix = np.eye(self.num_op, dtype=np.complex128)  
                else:
                    self.HMatrix = HMatrix
            else:
                self.HMatrix = HMatrix

            ## Check if representation matrices are functions or arrays
            if type(uMatrix).__name__ in ['CPUDispatcher'] and type(HMatrix).__name__ in ['CPUDispatcher']:
                self.nonfixedUnraveling = True
                uMatrix_suprt = []
                HMatrix_suprt = []
                @jit(nopython=True)
                def update_defintions(t, rho):
                    return update_defintions_uH(uMatrix(t, rho), HMatrix(t, rho), oMatrix)
                self.update_defintions = update_defintions
            if type(uMatrix).__name__ in ['CPUDispatcher']  and type(HMatrix).__name__ not in ['CPUDispatcher']:
                self.nonfixedUnraveling = True
                uMatrix_suprt = []
                HMatrix_suprt = []
                @jit(nopython=True)
                def update_defintions(t, rho):
                    return update_defintions_uH(uMatrix(t, rho), HMatrix, oMatrix)
                self.update_defintions = update_defintions
            if type(HMatrix).__name__ in ['CPUDispatcher']  and type(uMatrix).__name__ not in ['CPUDispatcher']:
                self.nonfixedUnraveling = True
                uMatrix_suprt = []
                HMatrix_suprt = []
                @jit(nopython=True)
                def update_defintions(t, rho):
                    return update_defintions_uH(uMatrix, HMatrix(t, rho), oMatrix)
                self.update_defintions = update_defintions
            if type(mMatrix).__name__ in ['CPUDispatcher'] :
                self.nonfixedUnraveling = True
                mMatrix_suprt = []
                @jit(nopython=True)
                def update_defintions(t, rho):
                    return update_defintions_M(mMatrix(t, rho))
                self.update_defintions = update_defintions
        ## If both representations are defined 
        elif not uMatrix == [] and not mMatrix == []:
            raise ValueError('Both U and M representation matrices are defined, this could lead to errors. Please just define one.')
        
        ## Get relevant matrices
        self.U_rep, self.M_rep, self.T_bar_rep = representation(self.num_op, mMatrix_suprt, uMatrix_suprt, HMatrix_suprt, oMatrix_suprt, TMatrix_suprt, PhiMatrix_suprt, WMatrix_suprt)

        ## Adittional definitions 
        self.M_dag = np.conjugate(np.transpose(np.asmatrix(self.M_rep)))
        self.M_M_dag = np.round(np.asmatrix(self.M_rep).dot(self.M_dag),6)
        self.sqrt_M_M_dag= sqrtm(np.identity(self.num_op) - self.M_M_dag)

        #########################################################
        #### M and U conditions 
        #########################################################
        self.inefficient, self.eta_diag = condition_check(self.U_rep, self.M_rep)

        #########################################################
        #### Lindblad operators related definitions  
        #########################################################
        if not (lindbladList == []):
            self.timedepent_lindbladoperators = False
            self.update_lindblad_operators(lindbladList)
            ## Lindblad operators must be a one argument function
            if not (type(lindbladList).__name__  in ['ndarray', 'CPUDispatcher']):
                raise ValueError('Lindblad channels must be passed as a jitted function or a numpy array of ndarray of dtype complex128')

            ## If they are not functions with these characteristics we redefine them as such
            if (type(lindbladList).__name__ == 'ndarray'):
                ## Check Lindblad ops variable type
                lindbladList_types = set(L.dtype.name for L in lindbladList)
                assert all(L_type == 'complex128' for L_type in lindbladList_types), 'Lindblad operators must be passed as a list of contiguousarray ndarray of dtype complex128'

                ## Define a numba compatible function for the Lindblad 
                lindbladList_ = np.array([np.ascontiguousarray(L) for L in lindbladList])

                ### Save original channels
                @jit(nopython=True)
                def original_cList(t, cList=lindbladList_):
                    return np.ascontiguousarray(cList)
                self.original_cList = original_cList
                self.original_lindbladList = lindbladList_
                ### unraveling transformed channels
                if self.nonfixedUnraveling:
                    @jit(nopython=True)
                    def Lindblad_ops(t, rho, cList=lindbladList_):
                        M_rep = update_defintions(t, rho)[0]
                        return np.ascontiguousarray(operators_Mrep(M_rep, cList))
                    self.cList = Lindblad_ops
                else:
                    lindbladList_tmp = operators_Mrep(self.M_rep, lindbladList_)
                    @jit(nopython=True)
                    def Lindblad_ops(t, rho, cList=lindbladList_tmp):
                        return np.ascontiguousarray(cList)
                    self.cList = Lindblad_ops
                    self.lindbladList = lindbladList_tmp

            elif (type(lindbladList).__name__ == 'CPUDispatcher'):
                self.timedepent_lindbladoperators = True
                ## Check Lindblad ops variable type
                lindbladList_types = set(L.dtype.name for L in lindbladList(0))
                assert all(L_type == 'complex128' for L_type in lindbladList_types), 'Lindblad operators must be passed as a function returning a list of contiguousarray ndarray of dtype complex128'
                self.original_cList = lindbladList
                if self.nonfixedUnraveling:
                    @jit(nopython=True)
                    def Lindblad_ops(t, rho):
                        M_rep = update_defintions(t, rho)[0]
                        lindbladList_ = operators_Mrep(M_rep, lindbladList(t))
                        return np.ascontiguousarray(lindbladList_)
                    self.cList = Lindblad_ops
                else:
                    @jit(nopython=True)
                    def Lindblad_ops(t, rho, M_rep=self.M_rep):
                        lindbladList_ = operators_Mrep(M_rep, lindbladList(t))
                        return np.ascontiguousarray(lindbladList_)
                    self.cList = Lindblad_ops

            else:
                raise ValueError('Incompatible Lindblad operators. Remember that these operators must pass as L = np.array([L_1, L_2, ...])')

        #########################################################
        #### Feedback operators related definitions  
        #########################################################
        if not (FList == []):
            ## Feedback operators must be two argument functions, beign its arguments time(t) and state(rho)
            if not (type(FList).__name__  in ['ndarray', 'CPUDispatcher']):
                raise ValueError('Feedback operators  must be passed as a jitted function or a list of ndarray of dtype complex128')

            ## If they are not functions with these characteristics we redefine them as such
            if (type(FList).__name__ == 'ndarray'):
                ## Check feedback ops variable type
                FList_types = set(F.dtype.name for F in FList)
                assert all(F_type == 'complex128' for F_type in FList_types), 'Feedback operators must be passed as a list of contiguousarray ndarray of dtype complex128'

                ## Define a numba compatible function for the feedback ops
                FList = np.array([np.ascontiguousarray(F) for F in FList])
                @jit(nopython=True)
                def feedback_ops(t, rho, FList_=FList):
                    return FList_
                self.FList = feedback_ops

            elif (type(FList).__name__ == 'CPUDispatcher'):
                ## Check feedback ops variable type
                FList_types = set(F.dtype.name for F in FList(0))
                assert all(F_type == 'complex128' for F_type in FList_types), 'Feedback operators must be passed as a function returning a list of contiguousarray ndarray of dtype complex128'

                @jit(nopython=True)
                def feedback_ops(t, rho, FList_=FList):
                    new_flist = np.zeros(np.shape(FList_(t)), dtype=np.complex128)
                    for n_F, F in enumerate(FList_(t)):
                        new_flist[n_F] += F
                    return np.ascontiguousarray(new_flist)
                self.FList = feedback_ops
            else:
                raise ValueError('Incompatible feedback operators. Remember that these operators must pass as F = np.array([f_1, f_2, ...])')

        ## diffusive functions compilation status
        self.diffusiveRhoTrajectory_compilation_status= False
        self.diffusiveRhoEnsemble_compilation_status = False
        self.diffusiveRhoAverage_compilation_status= False

        ## jump functions compilation status
        self.jumpRhoTrajectory_compilation_status= False
        self.jumpRhoEnsemble_compilation_status = False
        self.jumpRhoAverage_compilation_status= False

        ## feedback functions compilation status
        self.feedbackRhoTrajectory_compilation_status= False
        self.feedbackRhoAverage_compilation_status= False

        if (not self.timedepent_lindbladoperators) or (not self.timedepent_hamiltonian) or (not self.nonfixedUnraveling):
        ## diffusive functions compilation status
            self.diffusiveRhoTrajectory_compilation_status= True
            self.diffusiveRhoEnsemble_compilation_status = True
            self.diffusiveRhoAverage_compilation_status= True

            ## jump functions compilation status
            self.jumpRhoTrajectory_compilation_status= True
            self.jumpRhoEnsemble_compilation_status = True
            self.jumpRhoAverage_compilation_status= True

            ## feedback functions compilation status
            self.feedbackRhoTrajectory_compilation_status= True
            self.feedbackRhoAverage_compilation_status= True


    ############################################
    ######  diffusive trajectory functions #####
    ############################################
    def diffusiveRhoTrajectory_td(self, seed=0):
        self.diffusiveRhoTrajectory_compilation_status = True
        print('Compiling diffusiveRhoTrajectory ...')
        return partial(diffusiveRhoTrajectory_td, self.initialStateRho, self.timeList, self.H, self.original_cList, self.cList)(seed)

    def diffusiveRhoTrajectory_tind(self, seed=0):
        return partial(diffusiveRhoTrajectory_, self.initialStateRho, self.timeList, self.drivingH, self.original_lindbladList, self.lindbladList)(seed)

    def diffusiveRhoTrajectory(self, seed=0):
        if (self.timedepent_lindbladoperators) or (self.timedepent_hamiltonian) or (self.nonfixedUnraveling):
            return self.diffusiveRhoTrajectory_td(seed)
        else:
            return self.diffusiveRhoTrajectory_tind(seed)

    def diffusiveRhoEnsemble(self, n_trajectories):
        if ((self.timedepent_lindbladoperators) or (self.timedepent_hamiltonian) or (self.nonfixedUnraveling)) and (not self.diffusiveRhoEnsemble_compilation_status):
            self.diffusiveRhoEnsemble_compilation_status = True
            print('Compiling diffusiveRhoEnsemble ...')
            tmp = self.diffusiveRhoTrajectory_td(0)
            del tmp

        return misc.parallel_run(self.diffusiveRhoTrajectory_tind, np.arange(n_trajectories))

    def diffusiveRhoAverage(self, n_trajectories):
        if ((self.timedepent_lindbladoperators) or (self.timedepent_hamiltonian) or (self.nonfixedUnraveling)) and (not self.diffusiveRhoAverage_compilation_status):
            self.diffusiveRhoAverage_compilation_status = True
            print('Compiling diffusiveRhoAverage ...')
            tmp = self.diffusiveRhoTrajectory_td(0)
            del tmp
            
        all_traj = misc.parallel_run(self.diffusiveRhoTrajectory_tind, np.arange(n_trajectories))

        rho_average = np.zeros(np.shape(self.timeList) + np.shape(self.initialStateRho), dtype=np.complex128)
        for rho_traj in all_traj:
            rho_average = rho_average + (1/n_trajectories)*rho_traj
        return rho_average

    #######################################
    ######  jump trajectory functions #####
    #######################################
    def jumpRhoTrajectory_td(self, coherent_fields, seed=0):
        self.jumpRhoTrajectory_compilation_status = True
        print('Compiling jumpRhoTrajectory ...')
        return partial(jumpRhoTrajectory_td, self.initialStateRho, self.timeList, self.H, self.original_cList, self.eta_diag, self.cList)(coherent_fields, seed)

    def jumpRhoTrajectory_tind(self, coherent_fields, seed=0):
        return partial(jumpRhoTrajectory_, self.initialStateRho, self.timeList, self.drivingH, self.original_lindbladList, self.eta_diag, self.lindbladList)(coherent_fields, seed)

    def jumpRhoTrajectory(self, coherent_fields=[], seed=0):
        if len(coherent_fields) == 0:
            coherent_fields = self.coherent_fields

        if (self.timedepent_lindbladoperators) or (self.timedepent_hamiltonian) or (self.nonfixedUnraveling):
            return self.jumpRhoTrajectory_td(coherent_fields, seed)
        else:
            return self.jumpRhoTrajectory_tind(coherent_fields, seed)

    def jumpRhoEnsemble(self, n_trajectories, coherent_fields=[]):
        if len(coherent_fields) == 0:
            coherent_fields = self.coherent_fields

        if ((self.timedepent_lindbladoperators) or (self.timedepent_hamiltonian) or (self.nonfixedUnraveling)) and (not self.jumpRhoEnsemble_compilation_status):
            self.jumpRhoEnsemble_compilation_status = True
            print('Compiling jumpRhoEnsemble ...')
            tmp = self.jumpRhoTrajectory_td(0)
            del tmp

        return misc.parallel_run(partial(self.jumpRhoTrajectory, coherent_fields), np.arange(n_trajectories))

    def jumpRhoAverage(self, n_trajectories, coherent_fields=[]):
        if len(coherent_fields) == 0:
            coherent_fields = self.coherent_fields

        if ((self.timedepent_lindbladoperators) or (self.timedepent_hamiltonian) or (self.nonfixedUnraveling)) and (not self.jumpRhoAverage_compilation_status):
            self.jumpRhoAverage_compilation_status = True
            print('Compiling jumpRhoAverage ...')
            tmp = self.jumpRhoTrajectory_td(0)
            del tmp

        rho_average = np.zeros(np.shape(self.timeList) + np.shape(self.initialStateRho), dtype=np.complex128)
        all_traj = misc.parallel_run(partial(self.jumpRhoTrajectory, coherent_fields), np.arange(n_trajectories))
        for rho_traj in all_traj:
            rho_average = rho_average + (1/n_trajectories)*rho_traj
        return rho_average

    ############################################
    ######  feedback trajectory functions ######
    ############################################
    def feedbackRhoTrajectory(self, seed=0):
        if not self.feedbackRhoTrajectory_compilation_status:
            self.feedbackRhoTrajectory_compilation_status = True
            print('Compiling feedbackRhoTrajectory ...')
        return partial(feedbackRhoTrajectory_, self.initialStateRho, self.timeList, self.H, self.cList, self.FList)(seed)

    def feedbackRhoAverage(self, n_trajectories):
        if not self.feedbackRhoAverage_compilation_status:
            self.feedbackRhoAverage_compilation_status = True
            print('Compiling feedbackRhoAverage ...')
        rho_average = np.zeros(np.shape(self.timeList) + np.shape(self.initialStateRho), dtype=np.complex128)       
        all_traj = misc.parallel_run(self.feedbackRhoTrajectory, np.arange(n_trajectories))
        for rho_traj in all_traj:
            rho_average = rho_average + (1/n_trajectories)*rho_traj
        return rho_average

    ############################################
    ######  update definitions functions ######
    ############################################
    def update_representation(self, uMatrix = [], HMatrix = [], oMatrix = [], mMatrix = []):
        if (np.shape(uMatrix)[0] != 0) or (np.shape(HMatrix)[0] != 0) or (np.shape(mMatrix)[0] != 0):
            if (np.shape(uMatrix)[0] != 0) and (np.shape(HMatrix)[0] != 0):
                oMatrix = self.oMatrix
            elif (np.shape(uMatrix)[0] != 0) and (np.shape(oMatrix)[0] != 0):
                HMatrix = self.HMatrix
            elif (np.shape(HMatrix)[0] != 0) and (np.shape(oMatrix)[0] != 0):
                uMatrix = self.uMatrix
            elif (np.shape(uMatrix)[0] != 0):
                HMatrix = self.HMatrix
                oMatrix = self.oMatrix
            elif (np.shape(HMatrix)[0] != 0):
                uMatrix = self.uMatrix  
                oMatrix = self.oMatrix
            elif (np.shape(oMatrix)[0] != 0):
                uMatrix = self.uMatrix
                HMatrix = self.HMatrix
            self.U_rep, self.M_rep, self.T_bar_rep = update_defintions_uH(uMatrix, HMatrix, oMatrix)
        elif (mMatrix != []):
            self.U_rep, self.M_rep, self.T_bar_rep = update_defintions_M(mMatrix)
        self.update_lindblad_operators(self.original_lindbladList)

    def update_lindblad_operators(self, lindbladList):
        ## Lindblad operators must be a one argument function
        if not (type(lindbladList).__name__  in ['ndarray', 'CPUDispatcher']):
            raise ValueError('Lindblad channels must be passed as a jitted function or a numpy array of ndarray of dtype complex128')

        ## If they are not functions with these characteristics we redefine them as such
        if (type(lindbladList).__name__ == 'ndarray'):
            ## Check Lindblad ops variable type
            lindbladList_types = set(L.dtype.name for L in lindbladList)
            assert all(L_type == 'complex128' for L_type in lindbladList_types), 'Lindblad operators must be passed as a list of contiguousarray ndarray of dtype complex128'

            ## Define a numba compatible function for the Lindblad 
            lindbladList_ = np.array([np.ascontiguousarray(L) for L in lindbladList])

            ### Save original channels
            @jit(nopython=True)
            def original_cList(t, cList=lindbladList_):
                return np.ascontiguousarray(cList)
            self.original_cList = original_cList
            self.original_lindbladList = lindbladList_
            ### unraveling transformed channels
            if self.nonfixedUnraveling:
                @jit(nopython=True)
                def Lindblad_ops(t, rho, cList=lindbladList_):
                    M_rep = self.update_defintions(t, rho)[0]
                    return np.ascontiguousarray(operators_Mrep(M_rep, cList))
                self.cList = Lindblad_ops
            else:
                lindbladList_tmp = operators_Mrep(self.M_rep, lindbladList_)
                @jit(nopython=True)
                def Lindblad_ops(t, rho, cList=lindbladList_tmp):
                    return np.ascontiguousarray(cList)
                self.cList = Lindblad_ops
                self.lindbladList = lindbladList_tmp

        elif (type(lindbladList).__name__ == 'CPUDispatcher'):
            self.timedepent_lindbladoperators = True
            ## Check Lindblad ops variable type
            lindbladList_types = set(L.dtype.name for L in lindbladList(0))
            assert all(L_type == 'complex128' for L_type in lindbladList_types), 'Lindblad operators must be passed as a function returning a list of contiguousarray ndarray of dtype complex128'
            self.original_cList = lindbladList
            if self.nonfixedUnraveling:
                @jit(nopython=True)
                def Lindblad_ops(t, rho):
                    M_rep = self.update_defintions(t, rho)[0]
                    lindbladList_ = operators_Mrep(M_rep, lindbladList(t))
                    return np.ascontiguousarray(lindbladList_)
                self.cList = Lindblad_ops
            else:
                @jit(nopython=True)
                def Lindblad_ops(t, rho, M_rep=self.M_rep):
                    lindbladList_ = operators_Mrep(M_rep, lindbladList(t))
                    return np.ascontiguousarray(lindbladList_)
                self.cList = Lindblad_ops
        else:
            raise ValueError('Incompatible Lindblad operators. Remember that these operators must pass as L = np.array([L_1, L_2, ...])')

def condition_check(U_rep, M_rep):
    inefficient = False
    num_op = np.shape(M_rep)[0]
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
    return inefficient, np.ascontiguousarray(eta_diag)

def representation(num_op, mMatrix, uMatrix, HMatrix, oMatrix, TMatrix, PhiMatrix, WMatrix):
    #######################################################################     
    ## Unraveling parametrization / Orthogonal matrix taken as the identity 
    ######################################################################
    ### O Matrix
    if oMatrix == []:
        O = np.eye(2*num_op, dtype=np.complex128)
    else:
        O = oMatrix
        
    ### H matrix
    if len(HMatrix) == 0:
        HMatrix = np.eye(num_op, dtype=np.complex128)  
    
    if len(uMatrix)!= 0 and len(mMatrix)!= 0:
        raise ValueError('Both U and M representation matrices are defined, this could lead to errors. Please just define one.')

    if len(uMatrix) == 0 and len(mMatrix) == 0 and len(TMatrix) == 0 and len(PhiMatrix) == 0 and len(WMatrix) == 0:
        uMatrix = np.eye(num_op, dtype=np.complex128)
        U_rep = np.zeros((2*num_op,2*num_op), dtype=np.complex128)
        U_rep[:num_op, :num_op] = 0.5*(HMatrix + np.real(uMatrix))
        U_rep[num_op:, :num_op] = 0.5*(np.imag(uMatrix))
        U_rep[:num_op, num_op:] = 0.5*(np.imag(uMatrix))
        U_rep[num_op:, num_op:] = 0.5*(HMatrix - np.real(uMatrix))
        
        ## T matrix - M rep
        sqrt_U = sqrt_jit(U_rep)

        SQ_U_O = np.ascontiguousarray(np.dot(np.ascontiguousarray(sqrt_U), O))
        for i in range(np.shape(SQ_U_O)[0]):
            for j in range(np.shape(SQ_U_O)[0]):
                SQ_U_O[i,j] = np.round_(SQ_U_O[i,j], 7)

        ## M rep
        M_real = SQ_U_O[0:num_op,:]
        M_imag = SQ_U_O[num_op:2*num_op,:]
        M_rep = np.ascontiguousarray(M_real + 1j*M_imag)

        ## T_bar rep
        T_bar_real = SQ_U_O[:,0:num_op]
        T_bar_imag = SQ_U_O[:,num_op:2*num_op]
        T_bar_rep = np.ascontiguousarray(T_bar_real + 1j*T_bar_imag)

    elif len(uMatrix) == 0 and len(mMatrix) != 0:
        if np.shape(mMatrix) != (num_op,num_op*2):
            raise ValueError("Wrong M-Matrix dimension. Remember that M must have dimension (L,2L)")

        SQ_U_O = np.zeros((2*np.shape(mMatrix)[0],2*np.shape(mMatrix)[1]), dtype=np.complex128)
        SQ_U_O[:np.shape(mMatrix)[0],:] = np.real(mMatrix)
        SQ_U_O[np.shape(mMatrix)[0]:,:] = np.imag(mMatrix)

        ## T_bar rep
        T_bar_rep = np.transpose(np.real(mMatrix) + 1j*np.imag(mMatrix))

        ## U rep
        U_rep = np.dot(SQ_U_O, np.transpose(SQ_U_O))

        ## M rep
        M_rep = mMatrix

    elif len(uMatrix) != 0:
        if np.shape(uMatrix) != (num_op,num_op):
            raise ValueError("Wrong u-Matrix dimension. Remember that u must have dimension (L,L)")
        U_rep = np.zeros((2*num_op,2*num_op), dtype=np.complex128)
        U_rep[:num_op, :num_op] = 0.5*(HMatrix + np.real(uMatrix))
        U_rep[num_op:, :num_op] = 0.5*(np.imag(uMatrix))
        U_rep[:num_op, num_op:] = 0.5*(np.imag(uMatrix))
        U_rep[num_op:, num_op:] = 0.5*(HMatrix - np.real(uMatrix))
        
        ## T matrix - M rep
        sqrt_U = sqrt_jit(U_rep)

        SQ_U_O = np.ascontiguousarray(np.dot(np.ascontiguousarray(sqrt_U), O))
        for i in range(np.shape(SQ_U_O)[0]):
            for j in range(np.shape(SQ_U_O)[0]):
                SQ_U_O[i,j] = np.round_(SQ_U_O[i,j], 7)

        ## M rep
        M_real = SQ_U_O[0:num_op,:]
        M_imag = SQ_U_O[num_op:2*num_op,:]
        M_rep = np.ascontiguousarray(M_real + 1j*M_imag)

        ## T_bar rep
        T_bar_real = SQ_U_O[:,0:num_op]
        T_bar_imag = SQ_U_O[:,num_op:2*num_op]
        T_bar_rep = np.ascontiguousarray(T_bar_real + 1j*T_bar_imag)

    elif len(TMatrix) != 0 and len(PhiMatrix) != 0 and len(WMatrix) != 0:
        SQ_U_O = np.zeros((2*np.shape(TMatrix)[0],2*np.shape(TMatrix)[1]), dtype=np.complex128)
        T_bar_rep = np.zeros((2*np.shape(TMatrix)[0],np.shape(TMatrix)[1]), dtype=np.complex128)
        T_bar_rep[:np.shape(TMatrix)[0],:] = np.dot(np.dot(PhiMatrix[:np.shape(TMatrix)[0],:], np.transpose(WMatrix[:np.shape(TMatrix)[0],:])),TMatrix)
        T_bar_rep[np.shape(TMatrix)[0]:,:] = np.dot(np.dot(PhiMatrix[np.shape(TMatrix)[0]:,:], np.transpose(WMatrix[np.shape(TMatrix)[0]:,:])),TMatrix)
        SQ_U_O[:,:np.shape(TMatrix)[0]] = np.real(T_bar_rep)
        SQ_U_O[:,np.shape(TMatrix)[0]:] = np.imag(T_bar_rep)

        ## M rep
        M_rep = np.conjugate(np.transpose(np.real(T_bar_rep) + 1j*np.imag(T_bar_rep)))

        ## U rep
        U_rep = np.dot(SQ_U_O, np.transpose(SQ_U_O))

    return np.ascontiguousarray(U_rep), np.ascontiguousarray(M_rep), np.ascontiguousarray(T_bar_rep)

@jit(nopython=True)
def update_defintions_uH(uMatrix, HMatrix, oMatrix):
    num_op = np.shape(uMatrix)[0]
    # correlation matrix - U rep
    U_rep = np.zeros((2*num_op,2*num_op), dtype=np.complex128)
    U_rep[:num_op, :num_op] = 0.5*(HMatrix + np.real(uMatrix))
    U_rep[num_op:, :num_op] = 0.5*(np.imag(uMatrix))
    U_rep[:num_op, num_op:] = 0.5*(np.imag(uMatrix))
    U_rep[num_op:, num_op:] = 0.5*(HMatrix - np.real(uMatrix))
    
    ## T matrix - M rep
    sqrt_U = sqrt_jit(U_rep)
    
    SQ_U_O = np.ascontiguousarray(np.dot(np.ascontiguousarray(sqrt_U), oMatrix))
    for i in range(np.shape(SQ_U_O)[0]):
        for j in range(np.shape(SQ_U_O)[0]):
            SQ_U_O[i,j] = np.round_(SQ_U_O[i,j], 7)

    ## M rep
    M_real = SQ_U_O[0:num_op,:]
    M_imag = SQ_U_O[num_op:2*num_op,:]
    M_rep = np.ascontiguousarray(M_real + 1j*M_imag)

    ## T_bar rep
    T_bar_real = SQ_U_O[:,0:num_op]
    T_bar_imag = SQ_U_O[:,num_op:2*num_op]
    T_bar_rep = np.ascontiguousarray(T_bar_real + 1j*T_bar_imag)
    
    return U_rep, M_rep, T_bar_rep

@jit(nopython=True)
def update_defintions_M(mMatrix):
    ## T rep
    SQ_U_O = np.zeros((2*np.shape(mMatrix)[0],2*np.shape(mMatrix)[1]), dtype=np.complex128)
    SQ_U_O[:np.shape(mMatrix)[0],:] = np.real(mMatrix)
    SQ_U_O[np.shape(mMatrix)[0]:,:] = np.imag(mMatrix)

    ## T_bar rep
    T_bar_rep = np.transpose(np.real(mMatrix) + 1j*np.imag(mMatrix))

    ## U rep
    U_rep = np.dot(SQ_U_O, np.transpose(SQ_U_O))

    return U_rep, mMatrix, T_bar_rep

@jit(nopython=True)
def update_defintions_T_bar(TMatrix, PhiMatrix, WMatrix):
    ## T_bar rep
    SQ_U_O = np.zeros((2*np.shape(TMatrix)[0],2*np.shape(TMatrix)[1]), dtype=np.complex128)
    T_bar_Matrix = np.zeros((2*np.shape(TMatrix)[0],np.shape(TMatrix)[1]), dtype=np.complex128)
    T_bar_Matrix[:np.shape(TMatrix)[0],:] = np.dot(np.dot(PhiMatrix[:np.shape(TMatrix)[0],:], WMatrix),TMatrix)
    T_bar_Matrix[np.shape(TMatrix)[0]:,:] = np.dot(np.dot(PhiMatrix[np.shape(TMatrix)[0]:,:], WMatrix),TMatrix)
    SQ_U_O[:,:np.shape(TMatrix)[0]] = np.real(T_bar_Matrix)
    SQ_U_O[:,np.shape(TMatrix)[0]:] = np.imag(T_bar_Matrix)

    ## M rep
    M_rep = np.transpose(np.real(T_bar_Matrix) + 1j*np.imag(T_bar_Matrix))

    ## U rep
    U_rep = np.dot(SQ_U_O, np.transpose(SQ_U_O))

    return U_rep, M_rep, T_bar_Matrix