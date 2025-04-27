"""
***************************************
Project: Quantum trajectory integrator
Author: Diego Veloza Diaz
Email: dvelozad@unal.edu.co
Year: 2022
***************************************
"""

import numpy as np
from scipy.linalg import sqrtm
from numba import objmode, njit
import warnings
from functools import partial

# Import parallel auxiliary function
from qt_unraveling.misc_func import parallel_run

# Import numba optimized functions
from qt_unraveling.usual_operators import operators_Mrep, sqrt_jit

# Trajectory modules
from qt_unraveling.diffusive_trajectory import diffusiveRhoTrajectory_, diffusiveRhoTrajectory_td
from qt_unraveling.feedback_trajectory import feedbackRhoTrajectory_, feedbackRhoTrajectory_delay
from qt_unraveling.jumpy_trajectory import jumpRhoTrajectory_, jumpRhoTrajectory_td

# Import integrators
from qt_unraveling.integrators import custom_rungekutta_integrator, scipy_integrator, vonneumann_operator, standartLindblad_operator, feedbackEvol_operator


class QuantumSystem:
    def __init__(self, drivingH, initialState, timeList, *, lindbladList=None, FList=None, uMatrix=None, mMatrix=None, oMatrix=None, HMatrix=None, TMatrix=None, WMatrix=None, PhiMatrix=None):
        """
        Initialize the system with the provided parameters.
        """
        self.setup_time_interval(timeList)
        self.setup_initial_state(initialState)
        self.setup_hamiltonian(drivingH)
        
        if lindbladList is not None:
            self.setup_unraveling_matrices(lindbladList, FList, uMatrix, mMatrix, oMatrix, HMatrix, TMatrix, WMatrix, PhiMatrix)
            self.setup_diffusive_methods()
            self.setup_jumpy_methods()
            self.setup_feedback_methods()

    def setup_time_interval(self, timeList):
        self.timeList = timeList
        self.t0 = timeList[0]
        self.tmax = timeList[-1]
        self.maxiter = np.shape(timeList)[0]
        self.dt = abs(timeList[1] - timeList[0])

    def setup_initial_state(self, initialState):
        self.dimH = np.shape(initialState)[0]
        if len(np.shape(initialState)) == 1:
            self.initial_state_type = 0
            if np.round(np.linalg.norm(initialState), 5) != 1:
                warnings.warn('Initial state is unnormalized. Normalized state taken instead')
                self.initialStatePsi = (1. / np.linalg.norm(initialState)) * initialState
            else:
                self.initialStatePsi = initialState
            self.initialStateRho = np.asarray(np.transpose(np.asmatrix(self.initialStatePsi)).dot(np.conjugate(np.asmatrix(self.initialStatePsi))))
        elif len(np.shape(initialState)) == 2:
            self.initial_state_type = 1
            self.initialStateRho = np.asarray(initialState)

    def setup_hamiltonian(self, drivingH):
        if not (type(drivingH).__name__ in ['ndarray', 'CPUDispatcher']):
            raise ValueError('System Hamiltonian must be passed as a jitted function or ndarray of dtype complex128')

        self.timedepent_hamiltonian = False
        if type(drivingH).__name__ == 'ndarray':
            if not (drivingH.dtype.name == 'complex128'):
                raise ValueError('System Hamiltonian must be passed as a contiguousarray ndarray of dtype complex128')
            else:
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

    def setup_unraveling_matrices(self, lindbladList, FList, uMatrix, mMatrix, oMatrix, HMatrix, TMatrix, WMatrix, PhiMatrix):
        # Unraveling and Lindblad setup
        self.num_op = np.shape(lindbladList)[0] if type(lindbladList).__name__ in ['ndarray', 'list'] else np.shape(lindbladList(0))[0]
        self.nonfixedUnraveling = False
        self.coherent_fields = np.zeros(2 * self.num_op, dtype=np.complex128)

        self.mMatrix = mMatrix
        self.uMatrix = uMatrix
        self.oMatrix = oMatrix if oMatrix is not None else np.eye(2 * self.num_op, dtype=np.complex128)
        self.HMatrix = HMatrix if HMatrix is not None else np.eye(self.num_op, dtype=np.complex128)

        mMatrix_suprt = mMatrix
        uMatrix_suprt = uMatrix
        HMatrix_suprt = HMatrix
        oMatrix_suprt = oMatrix
        TMatrix_suprt = TMatrix
        WMatrix_suprt = WMatrix
        PhiMatrix_suprt = PhiMatrix

        if (uMatrix is not None) ^ (mMatrix is not None):
            if type(HMatrix).__name__ not in ['CPUDispatcher']:
                if HMatrix is None:
                    self.HMatrix = np.eye(self.num_op, dtype=np.complex128)
                    HMatrix = np.eye(self.num_op, dtype=np.complex128)
                else:
                    self.HMatrix = HMatrix

            if type(uMatrix).__name__ in ['function']:
                raise ValueError('Please redefine uMatrix function as a two parameter numba function uMatrix(t, rho)')

            if type(HMatrix).__name__ in ['function']:
                raise ValueError('Please redefine HMatrix function as a two parameter numba function HMatrix(t, rho)')

            if type(mMatrix).__name__ in ['function']:
                raise ValueError('Please redefine mMatrix function as a two parameter numba function mMatrix(t, rho)')

            if type(uMatrix).__name__ in ['CPUDispatcher'] and type(HMatrix).__name__ in ['CPUDispatcher']:
                self.nonfixedUnraveling = True
                uMatrix_suprt = None
                HMatrix_suprt = None
                self.update_defintions = lambda t, rho: update_defintions_uH(uMatrix(t, rho), HMatrix(t, rho), self.oMatrix)

            if type(uMatrix).__name__ in ['CPUDispatcher'] and type(HMatrix).__name__ not in ['CPUDispatcher']:
                self.nonfixedUnraveling = True
                uMatrix_suprt = None
                HMatrix_suprt = None
                self.update_defintions = lambda t, rho: update_defintions_uH(uMatrix(t, rho), HMatrix, self.oMatrix)

            if type(HMatrix).__name__ in ['CPUDispatcher'] and type(uMatrix).__name__ not in ['CPUDispatcher']:
                self.nonfixedUnraveling = True
                uMatrix_suprt = None
                HMatrix_suprt = None
                self.update_defintions = lambda t, rho: update_defintions_uH(uMatrix, HMatrix(t, rho), self.oMatrix)

            if type(mMatrix).__name__ in ['CPUDispatcher']:
                self.nonfixedUnraveling = True
                mMatrix_suprt = None
                self.update_defintions = lambda t, rho: update_defintions_M(mMatrix(t, rho))

        elif uMatrix is not None and mMatrix is not None:
            raise ValueError('Both U and M representation matrices are defined, this could lead to errors. Please just define one.')

        #self.U_rep, self.M_rep, self.T_bar_rep = representation(self.num_op, mMatrix, uMatrix, self.HMatrix, self.oMatrix, TMatrix, PhiMatrix, WMatrix)
        self.U_rep, self.M_rep, self.T_bar_rep = representation(self.num_op, mMatrix_suprt, uMatrix_suprt, HMatrix_suprt, oMatrix_suprt, TMatrix_suprt, PhiMatrix_suprt, WMatrix_suprt)

        self.inefficient, self.eta_diag = condition_check(self.U_rep, self.M_rep)
        self.setup_lindblad_operators(lindbladList)
        self.setup_feedback_operators(FList)

    def setup_lindblad_operators(self, lindbladList):
        if lindbladList is not None:
            self.timedepent_lindbladoperators = False
            self.M_dag = np.conjugate(np.transpose(np.asmatrix(self.M_rep)))
            self.M_M_dag = np.round(np.asmatrix(self.M_rep).dot(self.M_dag), 6)
            self.sqrt_M_M_dag = sqrtm(np.identity(self.num_op) - self.M_M_dag)
            self.original_obj_lindbladList = lindbladList
            self.update_lindblad_operators(lindbladList)

            if not (type(lindbladList).__name__ in ['ndarray', 'CPUDispatcher']):
                raise ValueError('Lindblad channels must be passed as a jitted function or a numpy array of ndarray of dtype complex128')

            if type(lindbladList).__name__ == 'ndarray':
                lindbladList_types = set(L.dtype.name for L in lindbladList)
                assert all(L_type == 'complex128' for L_type in lindbladList_types), 'Lindblad operators must be passed as a list of contiguousarray ndarray of dtype complex128'

                lindbladList_ = np.array([np.ascontiguousarray(L) for L in lindbladList])
                self.original_cList = lambda t, cList=lindbladList_: np.ascontiguousarray(cList)
                self.original_lindbladList = lindbladList_

                if self.nonfixedUnraveling:
                    self.cList = lambda t, rho: np.ascontiguousarray(operators_Mrep(self.update_defintions(t, rho)[1], lindbladList_))
                else:
                    lindbladList_tmp = operators_Mrep(self.M_rep, lindbladList_)
                    self.cList = lambda t, rho, cList=lindbladList_tmp: np.ascontiguousarray(cList)
                    self.lindbladList = lindbladList_tmp

            elif type(lindbladList).__name__ == 'CPUDispatcher':
                self.timedepent_lindbladoperators = True
                lindbladList_types = set(L.dtype.name for L in lindbladList(0))
                assert all(L_type == 'complex128' for L_type in lindbladList_types), 'Lindblad operators must be passed as a function returning a list of contiguousarray ndarray of dtype complex128'
                self.original_cList = lindbladList

                if self.nonfixedUnraveling:
                    self.cList = lambda t, rho: np.ascontiguousarray(operators_Mrep(self.update_defintions(t, rho)[1], lindbladList(t)))
                else:
                    self.cList = lambda t, rho, M_rep=self.M_rep: np.ascontiguousarray(operators_Mrep(M_rep, lindbladList(t)))

    def setup_feedback_operators(self, FList):
        if FList is not None:
            if not (type(FList).__name__ in ['ndarray', 'CPUDispatcher']):
                raise ValueError('Feedback operators must be passed as a jitted function or a list of ndarray of dtype complex128')

            if type(FList).__name__ == 'ndarray':
                FList_types = set(F.dtype.name for F in FList)
                assert all(F_type == 'complex128' for F_type in FList_types), 'Feedback operators must be passed as a list of contiguousarray ndarray of dtype complex128'
                FList = np.array([np.ascontiguousarray(F) for F in FList])
                self.FList = lambda t, rho, FList_=FList: FList_

            elif type(FList).__name__ == 'CPUDispatcher':
                FList_types = set(F.dtype.name for F in FList(0))
                assert all(F_type == 'complex128' for F_type in FList_types), 'Feedback operators must be passed as a function returning a list of contiguousarray ndarray of dtype complex128'
                self.FList = lambda t, rho, FList_=FList: np.ascontiguousarray([F for F in FList_(t)])

    def setup_diffusive_methods(self):
        self.diffusive_methods = DiffusiveMethods(self)

    def setup_jumpy_methods(self):
        self.jumpy_methods = JumpyMethods(self)

    def setup_feedback_methods(self):
        self.feedback_methods = FeedbackMethods(self)

    def vonneumann_analytical(self, integrator='scipy', method='BDF', rrtol=1e-5, aatol=1e-5, last_point=False):
        """
        Calculate the analytical von Neumann evolution using selected integrator
        
        Parameters:
        integrator (str): 'scipy' or 'runge-kutta' integration method
        method (str): Specific method for scipy integrator
        rrtol (float): Relative error tolerance for scipy solver
        aatol (float): Absolute error tolerance for scipy solver
        last_point (bool): If True, return only the final state
        
        Returns:
        Array of density matrices or single density matrix if last_point=True
        """
        hamiltonian = self.H
        
        # Define the von Neumann operator that's compatible with our integrators
        def op_lind(rho_it, it):
            H_it = hamiltonian(it)
            # Calculate -i[H, rho]
            return -1j * (np.dot(H_it, rho_it) - np.dot(rho_it, H_it))
        
        if integrator == 'scipy':
            return scipy_integrator(op_lind, self.initialStateRho, self.timeList, method=method, 
                                   rrtol=rrtol, aatol=aatol, last_point=last_point)
        elif integrator == 'runge-kutta':
            return custom_rungekutta_integrator(op_lind, self.initialStateRho, self.timeList, last_point=last_point)

    def lindblad_analytical(self, integrator='scipy', method='BDF', rrtol=1e-5, aatol=1e-5, last_point=False):
        """
        Calculate the analytical Lindblad evolution using selected integrator
        
        Parameters:
        integrator (str): 'scipy' or 'runge-kutta' integration method
        method (str): Specific method for scipy integrator
        rrtol (float): Relative error tolerance for scipy solver
        aatol (float): Absolute error tolerance for scipy solver
        last_point (bool): If True, return only the final state
        
        Returns:
        Array of density matrices or single density matrix if last_point=True
        """
        hamiltonian = self.H
        lindblad_ops = self.cList
        
        # Use the pure Python version instead of the Numba one
        from qt_unraveling.integrators import standartLindblad_operator_py
        
        # Define the Lindblad operator for the integrator
        def op_lind(rho_it, it):
            return standartLindblad_operator_py(hamiltonian, lindblad_ops, rho_it, it)
        
        if integrator == 'scipy':
            return scipy_integrator(op_lind, self.initialStateRho, self.timeList, method=method, 
                                   rrtol=rrtol, aatol=aatol, last_point=last_point)
        elif integrator == 'runge-kutta':
            return custom_rungekutta_integrator(op_lind, self.initialStateRho, self.timeList, last_point=last_point)

    def feedback_analytical(self, integrator='scipy', method='BDF', rrtol=1e-5, aatol=1e-5, last_point=False):
        """
        Calculate the analytical feedback evolution using selected integrator
        
        Parameters:
        -----------
        integrator : str
            'scipy' or 'runge-kutta' integration method
        method : str
            Specific method for scipy integrator
        rrtol : float
            Relative error tolerance for scipy solver
        aatol : float
            Absolute error tolerance for scipy solver
        last_point : bool
            If True, return only the final state
            
        Returns:
        --------
        Array of density matrices or single density matrix if last_point=True
        """
        hamiltonian = self.H
        lindblad_ops = self.cList
        original_lindblad_ops = self.original_cList
        feedback_ops = self.FList
        
        # Use the pure Python version of the feedback evolution operator
        from qt_unraveling.integrators import feedbackEvol_operator_py
        
        # Define the feedback operator for the integrator
        def op_lind(rho_it, it):
            return feedbackEvol_operator_py(hamiltonian, original_lindblad_ops, lindblad_ops, feedback_ops, rho_it, it)
        
        if integrator == 'scipy':
            return scipy_integrator(op_lind, self.initialStateRho, self.timeList, method=method, 
                                   rrtol=rrtol, aatol=aatol, last_point=last_point)
        elif integrator == 'runge-kutta':
            return custom_rungekutta_integrator(op_lind, self.initialStateRho, self.timeList, last_point=last_point)

    def update_representation(self, uMatrix=[], HMatrix=[], oMatrix=[], mMatrix=[]):
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
        if not (type(lindbladList).__name__ in ['ndarray', 'CPUDispatcher']):
            raise ValueError('Lindblad channels must be passed as a jitted function or a numpy array of ndarray of dtype complex128')

        if type(lindbladList).__name__ == 'ndarray':
            lindbladList_types = set(L.dtype.name for L in lindbladList)
            assert all(L_type == 'complex128' for L_type in lindbladList_types), 'Lindblad operators must be passed as a list of contiguousarray ndarray of dtype complex128'
            lindbladList_ = np.array([np.ascontiguousarray(L) for L in lindbladList])
            self.original_cList = lambda t, cList=lindbladList_: np.ascontiguousarray(cList)
            self.original_lindbladList = lindbladList_

            if self.nonfixedUnraveling:
                self.cList = lambda t, rho: np.ascontiguousarray(operators_Mrep(self.update_defintions(t, rho)[1], lindbladList_))
            else:
                lindbladList_tmp = operators_Mrep(self.M_rep, lindbladList_)
                self.cList = lambda t, rho, cList=lindbladList_tmp: np.ascontiguousarray(cList)
                self.lindbladList = lindbladList_tmp

        elif type(lindbladList).__name__ == 'CPUDispatcher':
            self.timedepent_lindbladoperators = True
            lindbladList_types = set(L.dtype.name for L in lindbladList(0))
            assert all(L_type == 'complex128' for L_type in lindbladList_types), 'Lindblad operators must be passed as a function returning a list of contiguousarray ndarray of dtype complex128'
            self.original_cList = lindbladList

            if self.nonfixedUnraveling:
                self.cList = lambda t, rho: np.ascontiguousarray(operators_Mrep(self.update_defintions(t, rho)[1], lindbladList(t)))
            else:
                self.cList = lambda t, rho, M_rep=self.M_rep: np.ascontiguousarray(operators_Mrep(M_rep, lindbladList(t)))
        else:
            raise ValueError('Incompatible Lindblad operators. Remember that these operators must pass as L = np.array([L_1, L_2, ...])')


class DiffusiveMethods:
    def __init__(self, system):
        self.system = system
        self.diffusiveRhoTrajectory_compilation_status = False
        self.diffusiveRhoEnsemble_compilation_status = False
        self.diffusiveRhoAverage_compilation_status = False

        if (not system.timedepent_lindbladoperators) and (not system.timedepent_hamiltonian) and (not system.nonfixedUnraveling):
            self.diffusiveRhoTrajectory_compilation_status = True
            self.diffusiveRhoEnsemble_compilation_status = True
            self.diffusiveRhoAverage_compilation_status = True

    def diffusive_rho_trajectory_td(self, method='euler', seed=0, verbose=False):
        """
        Non-Numba implementation for time-dependent operators
        Directly calls the diffusiveRhoTrajectory_td function from the module
        
        Parameters:
        -----------
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
        if not self.diffusiveRhoTrajectory_compilation_status and verbose:
            print('Preparing diffusiveRhoTrajectory (non-compiled) ...')
        self.diffusiveRhoTrajectory_compilation_status = True
        
        # Direct implementation without partial function
        from qt_unraveling.diffusive_trajectory import diffusiveRhoTrajectory_td
        
        return diffusiveRhoTrajectory_td(
            self.system.initialStateRho, 
            self.system.timeList, 
            self.system.H, 
            self.system.original_cList, 
            self.system.cList, 
            method=method, 
            seed=seed,
            verbose=verbose
        )

    def diffusive_rho_trajectory_tind(self, method='euler', seed=0):
        return partial(diffusiveRhoTrajectory_, self.system.initialStateRho, self.system.timeList, self.system.drivingH, self.system.original_lindbladList, self.system.lindbladList)(method, seed)

    def diffusive_rho_trajectory(self, method='euler', seed=0, verbose=False):
        if (self.system.timedepent_lindbladoperators) or (self.system.timedepent_hamiltonian) or (self.system.nonfixedUnraveling):
            return self.diffusive_rho_trajectory_td(method, seed, verbose)
        else:
            return self.diffusive_rho_trajectory_tind(method, seed)

    def diffusive_rho_ensemble(self, n_trajectories, method='euler', verbose=False):
        """
        Calculate an ensemble of quantum trajectories
        
        Parameters:
        -----------
        n_trajectories : int
            Number of trajectories to calculate
        method : str
            Integration method ('euler' or 'milstein')
        verbose : bool
            Whether to print progress information
            
        Returns:
        --------
        list
            List of trajectory arrays
        """
        if ((self.system.timedepent_lindbladoperators) or (self.system.timedepent_hamiltonian) or (self.system.nonfixedUnraveling)) and (not self.diffusiveRhoEnsemble_compilation_status):
            self.diffusiveRhoEnsemble_compilation_status = True
            if verbose:
                print('Preparing diffusiveRhoEnsemble (non-compiled) ...')

        all_traj = []
        for seed in range(n_trajectories):
            if (self.system.timedepent_lindbladoperators) or (self.system.timedepent_hamiltonian) or (self.system.nonfixedUnraveling):
                # Only first trajectory should print verbose output
                traj_verbose = verbose and seed == 0
                traj = self.diffusive_rho_trajectory_td(method=method, seed=seed, verbose=traj_verbose)
            else:
                traj = self.diffusive_rho_trajectory_tind(method=method, seed=seed)
            all_traj.append(traj)
            
        return all_traj

    def diffusive_rho_average(self, n_trajectories, method='euler', verbose=False, parallel=True):
        """
        Calculate the average of quantum trajectories, with option for parallel processing.
        
        Parameters:
        -----------
        n_trajectories : int
            Number of trajectories to average
        method : str
            Integration method ('euler' or 'milstein')
        verbose : bool
            Whether to print progress information
        parallel : bool
            Whether to use parallel processing
            
        Returns:
        --------
        numpy.ndarray
            Average trajectory
        """
        if ((self.system.timedepent_lindbladoperators) or (self.system.timedepent_hamiltonian) or (self.system.nonfixedUnraveling)) and (not self.diffusiveRhoAverage_compilation_status):
            self.diffusiveRhoAverage_compilation_status = True
            if verbose:
                print('Preparing diffusiveRhoAverage (non-compiled) ...')
        
        # Use parallel processing if requested and available
        if parallel:
            from qt_unraveling.misc_func import parallel_run
            
            # Create a wrapper function for parallel processing
            def traj_wrapper(seed):
                if (self.system.timedepent_lindbladoperators) or (self.system.timedepent_hamiltonian) or (self.system.nonfixedUnraveling):
                    # Only first trajectory should print verbose output
                    traj_verbose = verbose and seed == 0
                    return self.diffusive_rho_trajectory_td(method=method, seed=seed, verbose=traj_verbose)
                else:
                    return self.diffusive_rho_trajectory_tind(method=method, seed=seed)
            
            # Run the trajectories in parallel
            seed_list = np.arange(n_trajectories)
            
            # Show progress bar if verbose
            trajectory_list = parallel_run(traj_wrapper, seed_list, tqdm_bar=verbose)
            
            # Calculate the average
            time_shape = np.shape(self.system.timeList)
            state_shape = np.shape(self.system.initialStateRho)
            rho_average = np.zeros(time_shape + state_shape, dtype=np.complex128)
            
            for traj in trajectory_list:
                rho_average += traj / n_trajectories
                
            return rho_average
        else:
            # Original sequential implementation
            all_traj = self.diffusive_rho_ensemble(n_trajectories, method, verbose=verbose)
            
            rho_average = np.zeros(np.shape(self.system.timeList) + np.shape(self.system.initialStateRho), dtype=np.complex128)
            for rho_traj in all_traj:
                rho_average = rho_average + (1 / n_trajectories) * rho_traj
                
            return rho_average


class JumpyMethods:
    def __init__(self, system):
        self.system = system
        self.jumpRhoTrajectory_compilation_status = False
        self.jumpRhoEnsemble_compilation_status = False
        self.jumpRhoAverage_compilation_status = False
        self.coherent_field_check = True

        if (not system.timedepent_lindbladoperators) and (not system.timedepent_hamiltonian) and (not system.nonfixedUnraveling):
            self.jumpRhoTrajectory_compilation_status = True
            self.jumpRhoEnsemble_compilation_status = True
            self.jumpRhoAverage_compilation_status = True

    def jump_rho_trajectory_td(self, coherent_fields, seed=0, verbose=False, parallel=False):
        """
        Non-Numba implementation for time-dependent operators
        Directly calls the jumpRhoTrajectory_td function from the module
        
        Parameters:
        -----------
        coherent_fields : numpy.ndarray
            Coherent field amplitudes
        seed : int
            Random seed for stochastic integration
        verbose : bool
            Whether to print progress information
        parallel : bool
            Whether this function is being called in parallel mode
            
        Returns:
        --------
        numpy.ndarray
            Evolution of the density matrix
        """
        if not self.jumpRhoTrajectory_compilation_status and verbose:
            print('Preparing jumpRhoTrajectory (non-compiled) ...')
        self.jumpRhoTrajectory_compilation_status = True
        
        # Direct implementation without partial function to avoid type errors
        from qt_unraveling.jumpy_trajectory import jumpRhoTrajectory_td
        
        return jumpRhoTrajectory_td(
            self.system.initialStateRho, 
            self.system.timeList, 
            self.system.H, 
            self.system.original_cList, 
            self.system.eta_diag, 
            self.system.cList,
            coherent_fields,
            seed,
            verbose
        )

    def jump_rho_trajectory_tind(self, coherent_fields, seed=0, parallel=True):
        """
        Time-independent implementation of jump trajectory
        
        Parameters:
        -----------
        coherent_fields : numpy.ndarray
            Coherent field amplitudes
        seed : int
            Random seed for stochastic integration
        parallel : bool
            Whether the function is being called in parallel mode
            
        Returns:
        --------
        numpy.ndarray
            Evolution of the density matrix
        """
        from qt_unraveling.jumpy_trajectory import jumpRhoTrajectory_, jumpRhoTrajectory_py
        
        # When running in parallel or if there might be type issues, use the pure Python version
        if parallel:
            return jumpRhoTrajectory_py(
                self.system.initialStateRho,
                self.system.timeList,
                self.system.drivingH,
                self.system.original_lindbladList,
                self.system.eta_diag,
                self.system.lindbladList,
                coherent_fields,
                seed
            )
        else:
            # Try the Numba-optimized version for non-parallel execution
            try:
                return partial(jumpRhoTrajectory_, 
                    self.system.initialStateRho, 
                    self.system.timeList, 
                    self.system.drivingH, 
                    self.system.original_lindbladList, 
                    self.system.eta_diag, 
                    self.system.lindbladList)(coherent_fields, seed)
            except Exception as e:
                # If Numba version fails, fall back to Python implementation
                print(f"Warning: Numba-optimized implementation failed with error: {e}")
                print("Using pure Python implementation as fallback")
                return jumpRhoTrajectory_py(
                    self.system.initialStateRho,
                    self.system.timeList,
                    self.system.drivingH,
                    self.system.original_lindbladList,
                    self.system.eta_diag,
                    self.system.lindbladList,
                    coherent_fields,
                    seed
                )

    def jump_rho_trajectory(self, coherent_fields=[], coherent_field_check=True, seed=0, verbose=False, parallel=False):
        """
        Calculate a quantum jump trajectory
        
        Parameters:
        -----------
        coherent_fields : numpy.ndarray
            Coherent field amplitudes
        coherent_field_check : bool
            Whether to check coherent fields compatibility
        seed : int
            Random seed for stochastic integration
        verbose : bool
            Whether to print progress information
        parallel : bool
            Whether this function is being called in parallel mode
            
        Returns:
        --------
        numpy.ndarray
            Evolution of the density matrix
        """
        if len(coherent_fields) > 0 and coherent_fields is not None:
            if np.shape(coherent_fields)[0] != 2 * self.system.num_op:
                if verbose:
                    print(f'Warning: The number of coherent fields {np.shape(coherent_fields)[0]} should be twice the number of Lindblad operators {self.system.num_op}.')
                    print('Attempting to adjust coherent fields...')
                # Try to adjust the coherent fields
                if np.shape(coherent_fields)[0] < 2 * self.system.num_op:
                    # Pad with zeros
                    coherent_fields = np.pad(coherent_fields, (0, 2 * self.system.num_op - np.shape(coherent_fields)[0]), 
                                           mode='constant', constant_values=0)
                else:
                    # Truncate
                    coherent_fields = coherent_fields[:2 * self.system.num_op]

        if coherent_field_check:
            if np.all(coherent_fields == 0) and self.system.nonfixedUnraveling:
                warnings.warn('This library version does not support null coherent fields in addition to adaptative unraveling. Regular x-quadrature unraveling selected')
                num_op = self.system.num_op
                oMatrix = self.system.oMatrix
                self.system.update_defintions = lambda t, rho: update_defintions_uH(np.eye(num_op), np.eye(num_op), oMatrix)
                original_obj_lindbladList = self.system.original_obj_lindbladList

                if type(self.system.original_obj_lindbladList).__name__ == 'ndarray':
                    self.system.cList = lambda t, rho, cList=original_obj_lindbladList: np.ascontiguousarray(operators_Mrep(self.system.update_defintions(t, rho)[0], cList))
                elif type(self.system.original_obj_lindbladList).__name__ == 'CPUDispatcher':
                    self.system.cList = lambda t, rho: np.ascontiguousarray(operators_Mrep(self.system.update_defintions(t, rho)[0], original_obj_lindbladList(t)))

        if len(coherent_fields) == 0:
            coherent_fields = self.system.coherent_fields

        if (self.system.timedepent_lindbladoperators) or (self.system.timedepent_hamiltonian) or (self.system.nonfixedUnraveling):
            return self.jump_rho_trajectory_td(coherent_fields, seed, verbose, parallel)  # Added parallel parameter here
        else:
            return self.jump_rho_trajectory_tind(coherent_fields, seed, parallel)

    def jump_rho_ensemble(self, n_trajectories, coherent_fields=[], verbose=False, parallel=True):
        """
        Calculate an ensemble of quantum jump trajectories
        
        Parameters:
        -----------
        n_trajectories : int
            Number of trajectories to calculate
        coherent_fields : numpy.ndarray
            Coherent field amplitudes
        verbose : bool
            Whether to print progress information
        parallel : bool
            Whether to use parallel processing
            
        Returns:
        --------
        list
            List of trajectory arrays
        """
        if len(coherent_fields) == 0:
            coherent_fields = self.system.coherent_fields

        if ((self.system.timedepent_lindbladoperators) or (self.system.timedepent_hamiltonian) or (self.system.nonfixedUnraveling)) and (not self.jumpRhoEnsemble_compilation_status):
            self.jumpRhoEnsemble_compilation_status = True
            if verbose:
                print('Preparing jumpRhoEnsemble ...')
                # Initialize with a single run to ensure compilation
                try:
                    tmp = self.jump_rho_trajectory(coherent_fields=coherent_fields, coherent_field_check=True, seed=0, verbose=verbose)
                    del tmp
                except Exception as e:
                    print(f"Warning: Initial trajectory compilation failed with error: {e}")
                    print("Continuing with ensemble calculation...")
        
        # Use parallel processing if requested
        if parallel:
            from qt_unraveling.misc_func import parallel_run
            
            # Create a wrapper function for parallel processing
            def traj_wrapper(seed):
                # Only first trajectory should print verbose output
                traj_verbose = verbose and seed == 0
                return self.jump_rho_trajectory(
                    coherent_fields=coherent_fields, 
                    coherent_field_check=False, 
                    seed=seed, 
                    verbose=traj_verbose,
                    parallel=True
                )
            
            # Run trajectories in parallel
            return parallel_run(traj_wrapper, np.arange(n_trajectories), tqdm_bar=verbose)
        else:
            # Sequential implementation
            all_traj = []
            for seed in range(n_trajectories):
                traj_verbose = verbose and seed == 0
                traj = self.jump_rho_trajectory(
                    coherent_fields=coherent_fields, 
                    coherent_field_check=False, 
                    seed=seed, 
                    verbose=traj_verbose
                )
                all_traj.append(traj)
            return all_traj

    def jump_rho_average(self, n_trajectories, coherent_fields=[], verbose=False, parallel=True):
        """
        Calculate the average of quantum jump trajectories
        
        Parameters:
        -----------
        n_trajectories : int
            Number of trajectories to average
        coherent_fields : numpy.ndarray
            Coherent field amplitudes
        verbose : bool
            Whether to print progress information
        parallel : bool
            Whether to use parallel processing
            
        Returns:
        --------
        numpy.ndarray
            Average trajectory
        """
        if len(coherent_fields) == 0:
            coherent_fields = self.system.coherent_fields

        if ((self.system.timedepent_lindbladoperators) or (self.system.timedepent_hamiltonian) or (self.system.nonfixedUnraveling)) and (not self.jumpRhoAverage_compilation_status):
            self.jumpRhoAverage_compilation_status = True
            if verbose:
                print('Preparing jumpRhoAverage ...')
                # Initialize with a single run to ensure compilation
                try:
                    tmp = self.jump_rho_trajectory(coherent_fields=coherent_fields, coherent_field_check=True, seed=0, verbose=verbose)
                    del tmp
                except Exception as e:
                    print(f"Warning: Initial trajectory compilation failed with error: {e}")
                    print("Continuing with average calculation...")

        # Get all trajectories using the ensemble method
        all_traj = self.jump_rho_ensemble(
            n_trajectories, 
            coherent_fields=coherent_fields, 
            verbose=verbose,
            parallel=parallel
        )
        
        # Calculate average
        rho_average = np.zeros(np.shape(self.system.timeList) + np.shape(self.system.initialStateRho), dtype=np.complex128)
        for rho_traj in all_traj:
            rho_average = rho_average + (1 / n_trajectories) * rho_traj
            
        return rho_average


class FeedbackMethods:
    def __init__(self, system):
        self.system = system
        self.feedbackRhoTrajectory_compilation_status = False
        self.feedbackRhoAverage_compilation_status = False

        if (not system.timedepent_lindbladoperators) and (not system.timedepent_hamiltonian) and (not system.nonfixedUnraveling):
            self.feedbackRhoTrajectory_compilation_status = True
            self.feedbackRhoAverage_compilation_status = True

    def feedback_rho_trajectory(self, seed=0):
        if not self.feedbackRhoTrajectory_compilation_status:
            self.feedbackRhoTrajectory_compilation_status = True
            print('Compiling feedbackRhoTrajectory ...')
        return partial(feedbackRhoTrajectory_, self.system.initialStateRho, self.system.timeList, self.system.H, self.system.cList, self.system.FList)(seed)

    def feedback_rho_average(self, n_trajectories):
        if not self.feedbackRhoAverage_compilation_status:
            self.feedbackRhoAverage_compilation_status = True
            print('Compiling feedbackRhoAverage ...')
            tmp = self.feedback_rho_trajectory(0)
            del tmp

        rho_average = np.zeros(np.shape(self.system.timeList) + np.shape(self.system.initialStateRho), dtype=np.complex128)
        all_traj = parallel_run(self.feedback_rho_trajectory, np.arange(n_trajectories))
        for rho_traj in all_traj:
            rho_average = rho_average + (1 / n_trajectories) * rho_traj
        return rho_average

    def feedback_rho_trajectory_delay(self, tau, seed=0):
        if not self.feedbackRhoTrajectory_compilation_status:
            self.feedbackRhoTrajectory_compilation_status = True
            print('Compiling feedbackRhoTrajectory ...')
        return partial(feedbackRhoTrajectory_delay, self.system.initialStateRho, self.system.timeList, self.system.H, self.system.original_lindbladList, self.system.cList, self.system.FList, tau)(seed)

    def feedback_rho_average_delay(self, n_trajectories, tau):
        if not self.feedbackRhoAverage_compilation_status:
            self.feedbackRhoAverage_compilation_status = True
            print('Compiling feedbackRhoAverage ...')
            tmp = self.feedback_rho_trajectory_delay(tau=0, seed=0)
            del tmp

        rho_average = np.zeros(np.shape(self.system.timeList) + np.shape(self.system.initialStateRho), dtype=np.complex128)
        all_traj = parallel_run(partial(self.feedback_rho_trajectory_delay, tau), np.arange(n_trajectories))
        for rho_traj in all_traj:
            rho_average = rho_average + (1 / n_trajectories) * rho_traj
        return rho_average


def condition_check(U_rep, M_rep):
    inefficient = False
    num_op = np.shape(M_rep)[0]
    MMT = np.round(np.asmatrix(M_rep).dot(np.conjugate(np.transpose(np.asmatrix(M_rep)))), 6)
    diagMMT = np.zeros((num_op, num_op), dtype=np.complex128)
    diagVec = np.zeros(num_op, dtype=np.complex128)
    for i in range(num_op):
        diagMMT[i, i] = MMT[i, i]
        diagVec[i] = MMT[i, i]

    eta_diag = diagVec
    if ((MMT - diagMMT) != 0).any():
        print('MM^dag=', repr(MMT))
        raise ValueError('Invalid M matrix. Remember that M must be defined in such a way to ensure M.M^dag to be diagonal')

    if (np.round(np.real(diagMMT), 10) > 1).any() or (np.round(np.real(diagMMT), 8) < 0).any() or (np.round(np.imag(diagMMT), 8) != 0).any():
        print('MM^dag=', repr(MMT))
        raise ValueError('Invalid M matrix. Remember that M must be defined in such a way to ensure 0 <= M.M^dag <= 1')
    elif (np.round(np.real(diagVec), 8) < 1).any():
        inefficient = True

    U11 = np.round(U_rep[0:num_op, 0:num_op], 10)
    U22 = np.round(U_rep[num_op:2 * num_op, num_op:2 * num_op], 10)
    UtU = U11 + U22
    diagUtU = np.zeros((num_op, num_op), dtype=np.complex128)
    for i in range(num_op):
        diagUtU[i, i] = UtU[i, i]

    if ((UtU - diagUtU) != 0).any():
        print(UtU)
        raise ValueError('Invalid U matrix. Remember that U must be defined in such a way to ensure U11 + U22 to be diagonal')

    if (np.round(np.real(diagUtU), 10) > 1).any() or (np.round(np.real(diagUtU), 8) < 0).any() or (np.round(np.imag(diagUtU), 8) != 0).any():
        raise ValueError('Invalid U matrix. Remember that U must be defined in such a way to ensure 0 <= U11 + U22 <= 1')
    return inefficient, np.ascontiguousarray(eta_diag)


def sqrt_jit_local(M):
    """
    Local implementation of matrix square root for use in representation function.
    This avoids dependency issues with the imported sqrt_jit.
    """
    from scipy.linalg import sqrtm
    return sqrtm(M)

def representation(num_op, mMatrix, uMatrix, HMatrix, oMatrix, TMatrix, PhiMatrix, WMatrix):
    """
    Generate representation matrices for the quantum system.
    """
    if oMatrix is None:
        O = np.eye(2 * num_op, dtype=np.complex128)
    else:
        O = oMatrix

    if HMatrix is None:
        HMatrix = np.eye(num_op, dtype=np.complex128)

    if uMatrix is not None and mMatrix is not None:
        raise ValueError('Both U and M representation matrices are defined, this could lead to errors. Please just define one.')

    if uMatrix is None and mMatrix is None and TMatrix is None and PhiMatrix is None and WMatrix is None:
        uMatrix = np.eye(num_op, dtype=np.complex128)
        U_rep = np.zeros((2 * num_op, 2 * num_op), dtype=np.complex128)
        U_rep[:num_op, :num_op] = 0.5 * (HMatrix + np.real(uMatrix))
        U_rep[num_op:, :num_op] = 0.5 * (np.imag(uMatrix))
        U_rep[:num_op, num_op:] = 0.5 * (np.imag(uMatrix))
        U_rep[num_op:, num_op:] = 0.5 * (HMatrix - np.real(uMatrix))

        # Use local implementation instead of imported one
        sqrt_U = sqrt_jit_local(U_rep)
        SQ_U_O = np.ascontiguousarray(np.dot(np.ascontiguousarray(sqrt_U), O))
        for i in range(np.shape(SQ_U_O)[0]):
            for j in range(np.shape(SQ_U_O)[0]):
                SQ_U_O[i, j] = np.round(SQ_U_O[i, j], 7)  # Changed from np.round_ to np.round

        M_real = SQ_U_O[0:num_op, :]
        M_imag = SQ_U_O[num_op:2 * num_op, :]
        M_rep = np.ascontiguousarray(M_real + 1j * M_imag)

        T_bar_real = SQ_U_O[:, 0:num_op]
        T_bar_imag = SQ_U_O[:, num_op:2 * num_op]
        T_bar_rep = np.ascontiguousarray(T_bar_real + 1j * T_bar_imag)

    elif uMatrix is None and mMatrix is not None:
        if np.shape(mMatrix) != (num_op, num_op * 2):
            raise ValueError("Wrong M-Matrix dimension. Remember that M must have dimension (L, 2L)")
        SQ_U_O = np.zeros((2 * np.shape(mMatrix)[0], 2 * np.shape(mMatrix)[0]), dtype=np.complex128)
        SQ_U_O[:np.shape(mMatrix)[0], :] = np.real(mMatrix)
        SQ_U_O[np.shape(mMatrix)[0]:, :] = np.imag(mMatrix)

        T_bar_rep = np.transpose(np.real(mMatrix) + 1j * np.imag(mMatrix))
        U_rep = np.dot(SQ_U_O, np.transpose(SQ_U_O))
        M_rep = mMatrix

    elif uMatrix is not None:
        if np.shape(uMatrix) != (num_op, num_op):
            raise ValueError(f"Wrong u-Matrix dimension. Remember that u must have dimension ({num_op}, {num_op})")
        U_rep = np.zeros((2 * num_op, 2 * num_op), dtype=np.complex128)
        U_rep[:num_op, :num_op] = 0.5 * (HMatrix + np.real(uMatrix))
        U_rep[num_op:, :num_op] = 0.5 * (np.imag(uMatrix))
        U_rep[:num_op, num_op:] = 0.5 * (np.imag(uMatrix))
        U_rep[num_op:, num_op:] = 0.5 * (HMatrix - np.real(uMatrix))

        # Use local implementation instead of imported one
        sqrt_U = sqrt_jit_local(U_rep)
        SQ_U_O = np.ascontiguousarray(np.dot(np.ascontiguousarray(sqrt_U), O))
        for i in range(np.shape(SQ_U_O)[0]):
            for j in range(np.shape(SQ_U_O)[0]):
                SQ_U_O[i, j] = np.round(SQ_U_O[i, j], 7)  # Changed from np.round_ to np.round

        M_real = SQ_U_O[0:num_op, :]
        M_imag = SQ_U_O[num_op:2 * num_op, :]
        M_rep = np.ascontiguousarray(M_real + 1j * M_imag)

        T_bar_real = SQ_U_O[:, 0:num_op]
        T_bar_imag = SQ_U_O[:, num_op:2 * num_op]
        T_bar_rep = np.ascontiguousarray(T_bar_real + 1j * T_bar_imag)

    elif TMatrix is not None and PhiMatrix is not None and WMatrix is not None:
        SQ_U_O = np.zeros((2 * np.shape(TMatrix)[0], 2 * np.shape(TMatrix)[1]), dtype=np.complex128)
        T_bar_rep = np.zeros((2 * np.shape(TMatrix)[0], np.shape(TMatrix)[1]), dtype=np.complex128)
        T_bar_rep[:np.shape(TMatrix)[0], :] = np.dot(np.dot(PhiMatrix[:np.shape(TMatrix)[0], :], np.transpose(WMatrix[:np.shape(TMatrix)[0], :])), TMatrix)
        T_bar_rep[np.shape(TMatrix)[0]:, :] = np.dot(np.dot(PhiMatrix[np.shape(TMatrix)[0]:, :], np.transpose(WMatrix[np.shape(TMatrix)[0]:, :])), TMatrix)
        SQ_U_O[:, :np.shape(TMatrix)[0]] = np.real(T_bar_rep)
        SQ_U_O[:, np.shape(TMatrix)[0]:] = np.imag(T_bar_rep)

        M_rep = np.conjugate(np.transpose(np.real(T_bar_rep) + 1j * np.imag(T_bar_rep)))
        U_rep = np.dot(SQ_U_O, np.transpose(SQ_U_O))

    return np.ascontiguousarray(U_rep), np.ascontiguousarray(M_rep), np.ascontiguousarray(T_bar_rep)


def update_defintions_uH(uMatrix, HMatrix, oMatrix):
    """
    Non-Numba implementation for computing U_rep, M_rep, and T_bar_rep matrices
    
    This pure Python version avoids Numba typing issues with complex data
    """
    num_op = uMatrix.shape[0]
    U_rep = np.zeros((2 * num_op, 2 * num_op), dtype=np.complex128)
    U_rep[:num_op, :num_op] = 0.5 * (HMatrix + np.real(uMatrix))
    U_rep[num_op:, :num_op] = 0.5 * np.imag(uMatrix)
    U_rep[:num_op, num_op:] = 0.5 * np.imag(uMatrix)
    U_rep[num_op:, num_op:] = 0.5 * (HMatrix - np.real(uMatrix))

    # Use scipy's sqrtm instead of numba's sqrt_jit
    from scipy.linalg import sqrtm
    sqrt_U = sqrtm(U_rep)
    
    SQ_U_O = np.dot(sqrt_U, oMatrix)
    
    # Create M_rep directly without intermediates
    M_rep = SQ_U_O[:num_op].copy() + 1j * SQ_U_O[num_op:2*num_op].copy()
    
    # Create T_bar_rep directly
    T_bar_rep = np.zeros((2*num_op, num_op), dtype=np.complex128)
    T_bar_rep[:, :] = SQ_U_O[:, :num_op] + 1j * SQ_U_O[:, num_op:2*num_op]
    
    return np.ascontiguousarray(U_rep), np.ascontiguousarray(M_rep), np.ascontiguousarray(T_bar_rep)

def update_defintions_M(mMatrix):
    """
    Non-Numba implementation for computing matrices from an M-matrix
    """
    num_op = mMatrix.shape[0]
    SQ_U_O = np.zeros((2 * num_op, 2 * mMatrix.shape[1]), dtype=np.complex128)
    SQ_U_O[:num_op, :] = np.real(mMatrix)
    SQ_U_O[num_op:, :] = np.imag(mMatrix)

    T_bar_rep = np.transpose(np.real(mMatrix) + 1j * np.imag(mMatrix))
    U_rep = np.dot(SQ_U_O, np.transpose(SQ_U_O))

    return np.ascontiguousarray(U_rep), np.ascontiguousarray(mMatrix), np.ascontiguousarray(T_bar_rep)

def update_defintions_T_bar(TMatrix, PhiMatrix, WMatrix):
    """
    Non-Numba implementation for computing matrices from T, Phi, W matrices
    """
    num_op = TMatrix.shape[0]
    SQ_U_O = np.zeros((2 * num_op, 2 * TMatrix.shape[1]), dtype=np.complex128)
    T_bar_Matrix = np.zeros((2 * num_op, TMatrix.shape[1]), dtype=np.complex128)
    
    # Perform dot products once and store
    PhiW_upper = np.dot(PhiMatrix[:num_op, :], np.transpose(WMatrix[:num_op, :]))
    PhiW_lower = np.dot(PhiMatrix[num_op:, :], np.transpose(WMatrix[num_op:, :]))
    
    # Use stored dot products
    T_bar_Matrix[:num_op, :] = np.dot(PhiW_upper, TMatrix)
    T_bar_Matrix[num_op:, :] = np.dot(PhiW_lower, TMatrix)
    
    SQ_U_O[:, :num_op] = np.real(T_bar_Matrix)
    SQ_U_O[:, num_op:] = np.imag(T_bar_Matrix)

    M_rep = np.transpose(np.real(T_bar_Matrix) + 1j * np.imag(T_bar_Matrix))
    U_rep = np.dot(SQ_U_O, np.transpose(SQ_U_O))

    return np.ascontiguousarray(U_rep), np.ascontiguousarray(M_rep), np.ascontiguousarray(T_bar_Matrix)
