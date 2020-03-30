import ray
from qutip.states import ket2dm
from itertools import product,chain
from qutip.metrics import tracedist
from qutip import Qobj
from qutip import *
from random import sample
import cupy as cp
import numpy as np 
from matplotlib import pyplot as plt


@ray.remote
def energy_trace_batch(energy_states,dims,strt,num):
    """
    For a givens portion of the states specified by the strt and num variables
    calculates trace distance and energy distance and returns values of cartesian prod
    
    NOTE: the ptrace turns the state directly into the density matrix without need for calling
    Ket2dm speeds up and stops issue of memory buffer overwrite etc
    
    """
    
    
    energys = []
    traces = []
    
    dim = list(range(dims))
    
    for pair1,pair2  in product(energy_states[strt:strt+num], energy_states) :
        
        if pair1!=pair2:
            
            energys.append(abs(pair1[0]-pair2[0]))
                        
            substate1 = pair1[1].ptrace(dim)
            substate2 = pair2[1].ptrace(dim)
            
            trace_distance_val = tracedist(substate1,substate2)
            traces.append(trace_distance_val)

    return energys,traces
        
    
    

def energy_trace_compare_p(h,fraction,dims, random_sample = True, proc=4):
    
    energys, states = h.eigenstates()
    
    num_energys = len(energys)
    
    random_indices = sample(range(num_energys),int(num_energys*fraction))
    
    energy_states = [(energys[ind],states[ind]) for ind in random_indices]
    energy_states_id = ray.put(energy_states)
    
    n = int(num_energys/proc)
    
    result_ids = [energy_trace_batch.remote(energy_states_id,dims,n*i,n) for i in range(proc)]
    
    results = ray.get(result_ids)

    xs = [val[0] for val in results]
    ys = [val[1] for val in results]
    xs = list(chain(*xs))
    ys = list(chain(*ys))

    
    return xs,ys




def get_equilibrated_dens_op(energys, states, init_state:Qobj):
    return sum([abs(state.overlap(init_state)**2)*state*state.dag() for state in states])


def eff_dim(dens_oper:Qobj):
    """
    Returns effective dimension of mixed state:
    Via formula 1/Tr(rho^2)
    
    """
    dens_oper_sq = dens_oper**2
    return 1/(dens_oper_sq.tr())

def phase(t,E):
    return np.exp(-1j*E*t)


@ray.remote
def time_step(coef,eigenstates,eigenenergies,func,times,strt,num):
    
    result =[]
    
    for t in times[strt:strt+num]:
        state = sum([phase(energy,t)*coef[ind]*eigenstates[ind] for ind,energy in enumerate(eigenenergies)])
        result.append(func(state))
    
    return result



def simulate(energys,states,init,t_start,t_end,steps,ret_func = lambda x:x,Proc=4):
    
    
    
    coef = [init.overlap(state) for state in states]
    times = np.linspace(t_start,t_end,steps)
    
    energys_id = ray.put(energys)
    states_id = ray.put(states)
    coef_id = ray.put(coef)
    times_id = ray.put(times)
    
    num = int(steps/Proc)
    
    
    result = [time_step.remote(coef_id,states_id,energys_id, ret_func,times_id,num*i,num) for i in range(Proc)]
    
    results = ray.get(result)
    
    return list(chain(*results))
    
    
    
def equilibration_analyser_p(energys,states, init_state:Qobj, start,stop, steps:int, trace=[0],Proc=4): 
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    equilibrated_dens_op = get_equilibrated_dens_op(energys,states,init_state)
    effective_dimension = eff_dim(equilibrated_dens_op)
    
    #now we have the actual effective dimension trace over as cant do it before or messes up
    
    equilibrated_dens_op = equilibrated_dens_op.ptrace(trace)
    
    
    bound = 0.5*np.sqrt(2**len(trace)**2/effective_dimension)
    
    trace_dist_compare = lambda state: tracedist(equilibrated_dens_op,state.ptrace(trace))
    
    trace_distances = simulate(energys,states,init_state,start,stop,steps, ret_func= trace_dist_compare)
    
    times = [start+step*(stop-start)/steps for step in range(steps)]
    
    bound_line = [bound]*steps
    
    ax.plot(times,trace_distances,label="Trace-Distance")
    ax.plot(times,bound_line,label="Bound-Distance")
    plt.title(f"System: effective dimension {effective_dimension:.2f} and bound {bound:.2f} ")
    ax.set_xlabel(r"Time /$\hbar$s")
    ax.set_ylabel(r"$TrDist(\rho(t),\omega$)")
    plt.legend()
    plt.show()  
    
    
    
    



    
    
    
    