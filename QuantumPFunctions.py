import ray
from qutip.states import ket2dm
from itertools import product
from qutip.metrics import tracedist
from qutip import Qobj
from qutip import *
from random import sample
import cupy as cp
from numpy import all as allc 

@ray.remote
def energy_trace_batch(energy_state_chunk, energy_states,dims):
    
    energys = []
    traces = []
    
    dim = list(range(dims))
    
    for pair1,pair2  in product(energy_state_chunk, energy_states) :
        
        if pair1!=pair2:
            
            energys.append(abs(pair1[0]-pair2[0]))
            substate1 = pair1[1].ptrace(dim)
            substate2 = pair2[1].ptrace(dim)
            trace_distance_val = tracedist(substate1,substate2)
            traces.append(trace_distance_val)

    return energys,traces
        
    
    

def energy_trace_compare_p(h,fraction,dims, random_sample = True):
        
    energys, states = h.eigenstates()
    
    num_energys = len(energys)
    
    random_indices = sample(range(num_energys),int(num_energys*fraction))
    
    energy_states = [(energys[ind],states[ind]) for ind in random_indices]
    
    n = int(num_energys/4)
    
    energy_states_chunks = [energy_states[i * n:(i + 1) * n] for i in range((len(energy_states) + n - 1) // n )]  
    
    result_ids = [energy_trace_batch.remote(chunk,energy_states,dims) for chunk in energy_states_chunks]
    
    results = ray.get(result_ids)

    xs = [val[0] for val in results]
    ys = [val[1] for val in results]

    
    return xs,ys
    
    
    
    