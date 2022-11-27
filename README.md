# Overview
The purpose of this project is to build a library that allows us to implement Quantum Trajectories, an approach to open Markovian systems with an emphasis on the use of different unravelings. The implementation is done for Mathematica, Julia, and python. 

Our python implementation is based on the complete unraveling parametrization developed in [[1]](https://arxiv.org/abs/1102.3073), including a Quantum jump unraveling scheme in principle compatible with this same parametrization. Additionally, it is included the Markovian feedback scheme studied in [[2]](https://arxiv.org/abs/1102.3098).

# Dependencies
We recommend to install the package [anaconda](https://www.anaconda.com/products/individual), however, it is not required. You can run the following to get the necessary packages:

```python
pip install numpy
pip install scipy
```
# Installing
You can install the library by running the code:

```python
pip install qt_unraveling
```
# Getting started
Our python implementation is intended to generate a Class object called "System". A System class object call has to be done in the following way:  
```python
import qt_trajectories as qtr

test = qtr.System(drivingH, initialState, timeList, lindbladList, uMatrix, HMatrix, mMatrix,  FList, amp)
```
where ```drivingH``` corresponds to the system's Hamiltonian given as a one-parameter python function returning a complex numpy array, even if the system is time-independent, for example:

```python
def H0(t):
    return (1/2) * np.array([[0,1],[1,0]], dtype=np.complex128)
```
With ```initialState``` you can specify the initial state of the system, this can be a pure state or a mixed state. For the former case this parameter must be a complex numpy vector, for the later case it must be a complex numpy matrix: 
```python
## Pure initial state
psi0 = np.sqrt(1/5)*np.array([1,2], dtype = np.complex128)

## Mixed initial state
rho0 = np.array([[0.5,0.2],[0.1,0.5]], dtype = np.complex128)
```
The last required paramter is ```timeList``` and must be also a numpy array:
```python
t = np.linspace(t0, tf, np.int32(time_steps)) 
```
Given these three parameters we can obtain the Von Neumann evolution of the system by using the method ```VonNeumannAnalitical()```:
```python
test = qtr.System(H0, rho0, t) 
anali = test.VonNeumannAnalitical()
```
If the system has dimension 2, we can get a visualition by calling rhoBlochrep:
```python
test.rhoBlochrep(anali, 'Analitical', '--')
```
![alt text](./examples/example_graphs/vn_test.png)
With the ```lindbladList``` parameter we can include a set of Linblad operators as a function returning an array that contains each Lindblad operator expressed as a numpy array:
```python
def L():
    return [gamma*np.array([[0,1],[0,0]])]
```
In general, these operators can be time-dependent, in that case the we can change the above definitons by ```def L(t)```. It must take into account that they must be bounded operators and their decay rates ![formula](https://render.githubusercontent.com/render/math?math=\gamma) must be positive. The analitical evolution dictated by the Lindblad equation can be obtained by calling the function ```lindbladAnalitical()```:
 ```python
test = qtr.System(H0, rho0, t, L) 
rhoUana = test.lindbladAnalitical()
```
## Unravelings
As stated, our python implementation is based on the M and U unravelings parametrization developed by H. Wiseman et al. Details about each parametrization can be found in [[1]](https://arxiv.org/abs/1102.3073). The unraveling parametrization is specified by ```uMatrix``` or ```mMatrix```. Only one type of parametrization is allowed due to the possible conflicts that could arise. If the class receives as parameters both matrices simultaneously you should get the following warning:
 ```python
'Both U and M representation matrices are defined, this could lead to errors. Please just define one.'
```
In the case you want to use the U-representation, ```uMatrix``` must be a (L x L) numpy array, where L is the number of Lindblad operators. Similarly, for the M-representation, ```mMatrix``` must be a (L x 2L) numpy array:
 ```python
 ## U-representation
u_matrix = np.identity(len(L(0)), dtype = np.complex128)
test_Urep = qtr.System(H0, rho0, t, L, uMatrix = u_matrix) 

## M-representation
m_matrix = np.array([1,0])
test_Mrep = qtr.System(H0, rho0, t, L, mMatrix = m_matrix) 
```
You can also define an adaptive unraveling by defining ```uMatrix```, ```HMatrix```, or ```mMatrix``` as a two parameter function. To define the unraveling <img src="https://render.githubusercontent.com/render/math?math=u%20%3D%20%3C%5Csigma_%7B-%7D%3E%20%2F%20%3C%5Csigma_%7B%2B%7D%3E"> we write
 ```python
 ## U-representation
def u_matrix(t, rho):
    return np.array([[-np.trace(np.dot(rho,sigmam))/np.trace(np.dot(rho,sigmap))]])
```

Given a representation, you can check the corresponding matrix for the other representation by calling the class variables ```U_rep``` and ```M_rep```, for example:
 ```python
test_Urep.U_rep
#### output: array([1])

test_Urep.M_rep
#### output: array([1,0])
```
The System class is equipped with several quantum jump, diffusive and Markovian feedback oriented methods, each one of them returning a numpy array of the same length as t as follows:

### Quantum jumps
An individual trajectory can be obtained by calling ```jumpPsiTrajectory()``` or  ```jumpRhoTrajectory()```, this function will return an array of pure states for each time step:
 ```python
ind_traj_rho = test.jumpRhoTrajectory()
## returns [rho_t0, ...., rho_tf] 

ind_traj_psi = test.jumpPsiTrajectory()
## returns [psi_t0, ...., psi_tf] 
```
Additionally, the functions ```jumpRhoEnsemble(N)``` and ```jumpPsiEnsemble(N)``` will return an emsemble of N different trajectories given as density matrices or vector states respectivaly:
 ```python
ensemble_rho = test.jumpRhoEnsemble(N)
## returns [[rho_1_t0, ..., rho_1_tf], ...., [rho_N_0, ..., rho_N_tf]] 

ensemble_psi = test.jumpPsiEnsemble(N)
## returns [[psi_1_t0, ..., psi_1_tf], ...., [psi_N_0, ..., psi_N_tf]] 
```
In the other hand, the average quantum trajectory is given by the function ```jumpRhoAverage(n_trajectories)```, where ```n_trajectories``` corresponds to the number of different trajectories to calculate:
 ```python
rho_qjump = test.jumpRhoAverage(n_trajectories = 250)

## Plot
test.rhoBlochrep(rhoUana, 'Analitical', '-')
test.rhoBlochrep(rho_qjump, 'Qjump', '-')
```
![alt text](./examples/example_graphs/qjump_test.png)
Finally, this Qjumps implementation is compatible with each unraveling parametrization, but by default this option is deactivated. To activate the unraveling parametrization you have to set ```unraveling = True```:
 ```python
ensemble = test.jumpRhoEnsemble(N, unraveling = True)
rhoU = test.jumpRhoAverage(n_trajectories = 250, unraveling = True)
```
Additionally, you can set the amplitude of the external local oscillator (LO) via the ```amp``` parameter. 
### Diffusive unraveling
All difussive methods take the same form as for the Qjump case, so if you want to get an individual trajectory you can call:
 ```python
ind_traj_rho = test.diffusiveRhoTrajectory()
## returns [rho_t0, ...., rho_tf] 

ind_traj_psi = test.diffusivePsiTrajectory()
## returns [psi_t0, ...., psi_tf] 
```
To obtain an emsemble of N different trajectories given as density matrices or vector states you can call respectivaly:
 ```python
ensemble_rho = test.diffusiveRhoEnsemble(N)
## returns [[rho_1_t0, ..., rho_1_tf], ...., [rho_N_0, ..., rho_N_tf]] 

ensemble_psi = test.diffusivePsiEnsemble(N)
## returns [[psi_1_t0, ..., psi_1_tf], ...., [psi_N_0, ..., psi_N_tf]] 
```
Finally, the average quantum trajectory is given by the function ```diffusiveRhoAverage(n_trajectories)```:
 ```python
rho_diff = test.diffusiveRhoAverage(n_trajectories = 250)

## Plot
test.rhoBlochrep(rhoUana, 'Analitical', '-')
test.rhoBlochrep(rho_diff, 'Diffusive', '-')
```
![alt text](./examples/example_graphs/diffusive_test.png)
### Markovian feedback
To make use of feedback methods you have to include a one parameter function returning an array of numpy arrays for each feedback operator through the ```FList``` parameter:
 ```python
def F(t):
    return [np.array([[0,1],[1,0]], dtype = np.complex128), 
            np.array([[0,1j],[-1j,0]], dtype = np.complex128)]

test_Mrep = qtr.System(H0, rho0, t, L, mMatrix = m_matrix, FList = F)
```
Remenber that for each Lindblad operator defined in ```L(t)``` you must pass two Feedback matrices, this even for the case of an Homodyne type detection; assuming the detection of the x-quadrature you should define:
 ```python
def F(t):
    return [np.array([[0,1],[1,0]], dtype = np.complex128), 
            np.zeros((2,2), dtype = np.complex128)]
```
All methods take the same form as for the Qjump and diffusive cases: To get an individual trajectory you can call:
 ```python
ind_traj_rho = test.feedeRhoTrajectory()
## returns [rho_t0, ...., rho_tf] 
```
To obtain an emsemble of N different trajectories you can call:
 ```python
ensemble_rho = test.feedRhoEnsemble(N)
## returns [[rho_1_t0, ..., rho_1_tf], ...., [rho_N_0, ..., rho_N_tf]] 
```
Finally, the average quantum trajectory is given by the function ```feedRhoAverage(n_trajectories)```:
 ```python
average_feed = test.feedRhoAverage(n_trajectories = 250, method_ = "euler", parallelfor = True)
anali_feed = test.feedAnaliticalUncond(3, 1e-12, 1e-12)

## Plot
test.rhoBlochrep(average_feed, 'Diffusive', '-')
test.rhoBlochrep(anali_feed, 'Analitical', '-')

```
![alt text](./examples/example_graphs/feed_test.png)