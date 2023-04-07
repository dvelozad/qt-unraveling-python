# Overview
The purpose of this project is to build a library to implement and analyse Quantum Trajectories, an approach to open Markovian systems with an emphasis on the use of different unravelings. 

Our python implementation is based on the complete unraveling parametrization developed in [[1]](https://arxiv.org/abs/1102.3073), including a Quantum jump unraveling scheme in principle compatible with this same parametrization. Additionally, it is included the Markovian feedback scheme studied in [[2]](https://arxiv.org/abs/1102.3098).

# Installing
You can install the library via pip:

```python
pip install qt_unraveling
```
# Getting started
This library is based in a Class object called "System". A System class object call has to be done in the following way:  
```python
## Import library
import qt_trajectories as qtr

## System object
test = qtr.System(drivingH, initialState, timeList)
```
This library heavely uses the Numba library, as such many of the methods of the library expect jitted functions. In this example, ```drivingH``` corresponds to the system's Hamiltonian given as a numpy array for the time-independent case, or a one-parameter jitted python function returning a complex numpy array for the time-dependet case, i.e:

```python
## time-dependent case
@njit
def H0(t):
    return (1/2) * np.sin(0.1*t) *np.array([[0,1],[1,0]], dtype=np.complex128) 

## time-independent case
H0 = (1/2) * np.array([[0,1],[1,0]], dtype=np.complex128) # or equivalently -> drivingH = qtr.sigmax
```
With the argument ```initialState``` you can specify the initial state of the system, this can be a pure state or a mixed state. For the former case this parameter must be a complex numpy vector, for the later case it must be a complex numpy matrix: 
```python
## Pure initial state
psi0 = np.sqrt(1/5)*np.array([1,2], dtype = np.complex128)

## Mixed initial state
rho0 = np.array([[0.5,0.2],[0.1,0.5]], dtype = np.complex128)
```
The last required argument is ```timeList``` and must be also a numpy array:
```python
## time-span
time_list = np.linspace(t0, tf, np.int32(time_steps)) 
```
Given these three parameters we can obtain the Von Neumann evolution of the system by using the method ```VonNeumannAnalitical()```:
```python
## System object
qtr_test = qtr.System(H0, rho0, time_list) 

## Von neumann time evolution
von_neu_evolution = qtr_test.VonNeumannAnalitical()
```
If the system has dimension 2, we can get a visualition by calling ```qtr.misc.rhoBlochcomp_plot```:
```python
qtr.misc.rhoBlochcomp_plot(von_neu_evolution, qtr_test.timeList, label='traj')
```
![image](https://drive.google.com/uc?export=view&id=1zpToQQATgKsXR9loDo3qXwiGJ_y0UN3d)
With the ```lindbladList``` argument we can include a set of Linblad operators as a jitted function returning an array that contains each Lindblad operator expressed as a numpy array:
```python
## time-dependent Lindblad ops
@njit
def L(t):
    num_op = 1
    L_list = np.zeros((num_op,2,2), dtype=np.complex128)
    L_list[0] = np.sqrt(gamma)*np.kron(qtr.sigmam, np.eye(2, dtype = np.complex128))
    return L_list

## time-independent Lindblad ops
L = np.sqrt(gamma)*np.array([qtr.sigmam], dtype=np.complex128)
```
In general, these operators can be time-dependent. We must take into account that they must be bounded operators and their decay rates ![formula](https://render.githubusercontent.com/render/math?math=\gamma) must be positive. The analitical evolution dictated by the Lindblad equation can be obtained by calling the function ```lindbladAnalitical()```:
 ```python
 ## System object
qtr_test = qtr.System(H0, rho0, time_list, L) 

## Lindblad time evolution
lindblad_evolution = qtr_test.lindbladAnalitical()

## Diffusive plot
fig, ax = qtr.misc.figure()
qtr.misc.rhoBlochcomp_plot(lindblad_evol, qtr_test.timeList, label='Analitical', line='-', ax=ax)
```
![image](https://drive.google.com/uc?export=view&id=1da0JhLKxvH27uEu08MpI4-0npUbbwEuQ)
## Unravelings
As stated, our python implementation is based on the M and U unravelings parametrization developed by H. Wiseman et al. Details about each parametrization can be found in [[1]](https://arxiv.org/abs/1102.3073). The unraveling parametrization is specified by ```uMatrix``` or ```mMatrix```. Only one type of parametrization is allowed due to the possible conflicts that could arise. If the class receives both matrices simultaneously you should get the following warning:
 ```python
'Both U and M representation matrices are defined, this could lead to errors. Please just define one.'
```
In the case you want to use the U-representation, ```uMatrix``` must be a (L x L) numpy array, where L is the number of Lindblad operators. Similarly, for the M-representation, ```mMatrix``` must be a (L x 2L) numpy array:
 ```python
 ## U-representation
u_matrix = np.eye(np.shape(L)[0], dtype=np.complex128)
qtr.test_Urep = qtr.System(H0, rho0, time_list, L, uMatrix = u_matrix) 

## M-representation
m_matrix = np.array([1,0], dtype=np.complex128)
qtr.test_Mrep = qtr.System(H0, rho0, time_list, L, mMatrix = m_matrix) 
```
You also have the ability of defining the U unraveling parametriztion extra degrees of freedom: The efficiency matrix H and the orthogonal matrix O. These can be defined as follows:
 ```python
## U-representation
u_matrix = np.eye(np.shape(L)[0], dtype=np.complex128)

## Efficiency freedom
h_matrix = 0.5*np.eye(np.shape(L)[0], dtype=np.complex128)

## Orthogonal freedom
theta = 0
oMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

## System object
qtr.test_Urep = qtr.System(H0, rho0, time_list, L, uMatrix = u_matrix, oMatrix = oMatrix, HMatrix = h_matrix) 
```
We can also define an adaptive unraveling by defining ```uMatrix```, ```HMatrix```, or ```mMatrix``` as a two parameter jitted function. To define the unraveling <img src="https://render.githubusercontent.com/render/math?math=u%20%3D%20%3C%5Csigma_%7B-%7D%3E%20%2F%20%3C%5Csigma_%7B%2B%7D%3E"> we write
 ```python
# You have the option to define an adaptive unraveling. This is done by defining it in term of a function of t and the state
## Adapative unraveling
@jit(nopython=True)
def u_matrix(t, rho):
    num_op = 1
    uM = np.zeros((num_op,num_op), dtype=np.complex128)
    uM[0,0] = -np.trace(np.dot(rho,qtr.sigmam))/np.trace(np.dot(rho,qtr.sigmap))
    return uM
```

Given a representation, you can check the corresponding matrix for the other representation by calling the class variables ```U_rep``` and ```M_rep```, for example:
 ```python
qtr_test_Urep.U_rep
#### output: array([1])

qtr_test_Urep.M_rep
#### output: array([1,0])
```
The ```System``` class is equipped with several quantum jump, diffusive and Markovian feedback oriented methods, each one of them returning a numpy array of the same length as ```time_list```.

### Quantum jumps
An individual trajectory can be obtained by calling ```jumpRhoTrajectory()```, this function will return an array of quantum states for each time step:
 ```python
ind_traj_rho = qtr_test.jumpRhoTrajectory()
## returns [rho_t0, ...., rho_tf] 
```
Additionally, the method ```jumpRhoEnsemble(N)``` will return an emsemble of ```N``` different trajectories given as density matrices:
 ```python
ensemble_rho = qtr_test.jumpRhoEnsemble(N)
## returns [[rho_1_t0, ..., rho_1_tf], ...., [rho_N_0, ..., rho_N_tf]] 
```
In the other hand, the average quantum trajectory is given by the function ```jumpRhoAverage(n_trajectories)```, where ```n_trajectories``` corresponds to the number of different trajectories to calculate
 ```python
average_rho = jumpRhoAverage(n_trajectories)
## returns [[rho_1_t0, ..., rho_1_tf], ...., [rho_N_0, ..., rho_N_tf]] 
```
A typical implementation of a qjump trajectory looks like this:
 ```python
## Average conditional Qjump evoltuion
rho_qjump = qtr_test.jumpRhoAverage(n_trajectories=1)

## Qjump plot
fig, ax = qtr.misc.figure()
qtr.misc.rhoBlochcomp_plot(rho_qjump, qtr_test.timeList, label='QJump', line='-', ax=ax)
qtr.misc.rhoBlochcomp_plot(lindblad_evolution, qtr_test.timeList, label='Analitical', line='-', ax=ax)
```
![image](https://drive.google.com/uc?export=view&id=1UJ4dy_yAGEibG2NZxqBJV31mEu0A8mVF)
Finally, this Qjumps implementation is compatible with each unraveling parametrization, but by default this option is deactivated. The optional argument ```coherent_fields``` corresponds to a numpy array where we can specify the amplitude of the coherent fields added to archive the desired -dyne type measuarement. 
 ```python
## Average conditional Qjump evoltuion
coherent_fields = np.array([amplitude_x, amplitude_y], dtype=np.complex128)
rho_qjump = qtr_test.jumpRhoAverage(n_trajectories=1, coherent_fields=coherent_fields)
```
As these coherent fields approach infinity, the qjump trajectory generated approaches the diffusive scheme, as described in the Qjump and diffusive schemes connection section. 

### Diffusive unraveling
All difussive methods take the same form as for the Qjump case, so if you want to get an individual trajectory you can call:
 ```python
ind_traj_rho = qtr_test.diffusiveRhoTrajectory()
## returns [rho_t0, ...., rho_tf] 
```
To obtain an emsemble of N different trajectories given as density matrices or vector states you can call respectivaly:
 ```python
ensemble_rho = qtr_test.diffusiveRhoEnsemble(N)
## returns [[rho_1_t0, ..., rho_1_tf], ...., [rho_N_0, ..., rho_N_tf]] 
```
Finally, the average quantum trajectory is given by the function ```diffusiveRhoAverage(n_trajectories)```:
 ```python
rho_diff = qtr_test.diffusiveRhoAverage(n_trajectories = 1)

## Diffusive plot
fig, ax = qtr.misc.figure()
qtr.misc.rhoBlochcomp_plot(rho_diff, qtr_test.timeList, label='Diffusive', line='-', ax=ax)
qtr.misc.rhoBlochcomp_plot(lindblad_evol, qtr_test.timeList, label='Analitical', line='-', ax=ax)
```
![image](https://drive.google.com/uc?export=view&id=1hsw6nMxG9JPwDWw_mxnBxJrHXVwHB3RS)
We also have the possibility of representing the trajectory in the Bloch sphere as follows
 ```python
## Bloch sphere
qtr.misc.rhoBlochSphere([lindblad_evol, rho_diff])
```
![image](https://drive.google.com/uc?export=view&id=17Dm3UcYsYXyZ6wAkpbnGPwZbu6w-jeiG)
### Qjump and diffusive schemes connection
As mentioned before, from the Qjump scheme you can get to the diffusive scheme using the ```coherent_fields``` argument
 ```python
## Qjump trajectory conditional evoltuion
amplitude_x, amplitude_y = 5, 5
coherent_fields = np.array([amplitude_x, amplitude_y], dtype=np.complex128)
rho_qjump = qtr_test.jumpRhoAverage(n_trajectories=1, coherent_fields=coherent_fields)

## Diffusive conditional evoltuion
rho_diff = qtr_test.diffusiveRhoAverage(n_trajectories = 1)
```
Both trajectories look that like
![image](https://drive.google.com/uc?export=view&id=1JbO908CXVmbSTSPZgPW0tirDuIS2iiqR)
![image](https://drive.google.com/uc?export=view&id=1CBG86cSnS5LvMWfO8G3ihX9XvC1oFtsJ)

You can tweak the amplitudes as you see fit.
### Markovian quantum feedback
Our Markovian quantum feedback implementeation is based on [[2]](https://arxiv.org/abs/1102.3098). As it is shown in previous sections, to make use of the feedback methods we have to include a jitted function returning an array of numpy arrays for each feedback operator or a numpy array conteining each feedback operator:
 ```python
 ## time-depenedt feedback operators
@jit(nopython=True)
def F_func(t):
    num_op = 2
    F_list = np.zeros((num_op,2,2), dtype=np.complex128)
    F_list[0] = np.sin(0.5*t)*qtr.sigmay
    F_list[1] = np.sin(t)*qtr.sigmaz
    return F_list

## time-independent Lindblad ops
F_list = np.sqrt(gamma)*np.array([qtr.sigmay, qtr.sigmaz], dtype=np.complex128)

## System definition
qtr_test = qtr.System(H0, initialState, timelist, lindbladList = L, FList = F_func, uMatrix=u_matrix)
# qtr_test = qtr.System(H0, initialState, timelist, lindbladList = L, FList = F_list, uMatrix=u_matrix)
```
Remember that for each Lindblad operator defined in ```lindbladList``` you must pass two Feedback matrices, this even for the case of an Homodyne type detection; assuming the detection of the x-quadrature you should define:
 ```python
@jit(nopython=True)
def F_func(t):
    num_op = 2
    F_list = np.zeros((num_op,2,2), dtype=np.complex128)
    F_list[0] = np.sin(0.5*t)*qtr.sigmay
    F_list[1] = np.zeros((2,2), dtype = np.complex128)
    return F_list
```
All methods take the same form as for the Qjump and diffusive cases: To get an individual trajectory you can call
 ```python
## single feedback trajectory
ind_traj_rho = qtr_test.feedeRhoTrajectory()
# returns [rho_t0, ...., rho_tf] 

## plot trajectory
qtr.misc.rhoBlochcomp_plot(feedback_traj, timelist, line='-')
```
![image](https://drive.google.com/uc?export=view&id=1HM0v_h6CVriLexpiTg4XIIQI79Y7UR6F)

To obtain an emsemble of N different trajectories you can call
 ```python
## feedback emsemble
ensemble_rho = qtr_test.feedRhoEnsemble(N)
# returns [[rho_1_t0, ..., rho_1_tf], ...., [rho_N_0, ..., rho_N_tf]] 
```
The average quantum trajectory is given by the function ```feedRhoAverage(n_trajectories)```:
 ```python
 ## Average conditional feedrbac evoltuion
average_feed = qtr_test.feedRhoAverage(n_trajectories = 1000)
```
And the analitical evolution in the presence of feedback is given by
 ```python
 ## Anlitical feedback evolution
feedback_evol = qtr_test.feedbackAnalitical()

## Feedback plot
fig, ax = qtr.misc.figure()
qtr.misc.rhoBlochcomp_plot(feedback_evol, timelist, label='analitical', ax=ax, line='-')
qtr.misc.rhoBlochcomp_plot(average_feedback_traj, timelist, label='average', ax=ax, line='--')
```
![image](https://drive.google.com/uc?export=view&id=1wyG7ORvQkdxmuePVKBq6MfonZjL-nLfN)