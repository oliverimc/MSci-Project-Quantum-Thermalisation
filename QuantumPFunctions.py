import ray
from qutip.states import ket2dm
from itertools import product,chain
from qutip.metrics import tracedist
from qutip import Qobj
from qutip import *
from random import sample
import cupy as cp
from numpy import all as allc 
import numpy as np 

@ray.remote
def energy_trace_batch(energy_states,dims,strt,num):
    
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
    
    energy_states = [(energys[ind],ket2dm(states[ind])) for ind in random_indices]
    energy_states_id = ray.put(energy_states)
    
    n = int(num_energys/proc)
    
    result_ids = [energy_trace_batch.remote(energy_states_id,dims,n*i,n) for i in range(proc)]
    
    results = ray.get(result_ids)

    xs = [val[0] for val in results]
    ys = [val[1] for val in results]
    xs = list(chain(*xs))
    ys = list(chain(*ys))

    
    return xs,ys
    
    
    
    