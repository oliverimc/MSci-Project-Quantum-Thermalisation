from qutip import sigmax, sigmay, sigmaz, identity, tensor
from qutip.mesolve import mesolve
from qutip.essolve import essolve
from qutip import Qobj
from qutip.random_objects import rand_unitary
from qutip.metrics import tracedist
from qutip.states import basis,ket2dm
from qutip import Options

from numpy import linspace,array,sqrt,zeros
from numpy.random import normal
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import matplotlib.patches as mpatches
from random import sample
import numpy as np

import sys
from math import sqrt
from time import time
from itertools import product
from collections import Counter

from functools import reduce



#   |____                         ___
#   |___         ---------          |
#   |   |        ---------          | 
#   |   |                        ___|___
# hbar ==1   


#purity = Tr(rho^2) 1->1/d where 1/d is maximally mixed 

sigma = [sigmax(),sigmay(),sigmaz()]
sqrt2 = sqrt(2)

#TODO FIX MEMORY ERROR by replacing sum(list) with iteration sum
#TODO watch system equilibrate

def get_sig_dif_states(H):
    """
    Takes a Hamiltonian calcualtes its eigenstates and values
    then removes any degneracies (chucks away states) and makes sure they are not 
    within a small distance of each other
    """
    
    val,states = H.eigenstates()
    pairs = zip(val,states)
    non_degen_pairs =[]
    e_seen =[] # what energy values have we seen so far we know to chuck away because seen before => degenerate
    
    for e,s in pairs:
        if e in e_seen:
            pass
        
        else:
            e_seen.append(e)
            non_degen_pairs.append((e,s))

    prev = non_degen_pairs[0][0]

    sig_different_pairs =[]

    for energy, state in non_degen_pairs:
        if(abs(energy-prev)>0.5):
            sig_different_pairs.append((energy,state))
            prev = energy

    
    return sig_different_pairs
    
def Heisenberg1dRingGen(Jx,Jy,Jz,N):
    
    def return_func(n,m,i,j):
        if (i==j):
            if(abs(n-m)==1):
                return -1/2*[Jx,Jy,Jz][i]
            if((n==(N-1) and m==0) or (n==0 and m==(N-1))):
                return -1/2*[Jx,Jy,Jz][i]
        return 0
    
    return return_func

def Heisenberg1dChainGen(Jx,Jy,Jz,N):

    def return_func(n,m,i,j):
        if (i==j):
            if(abs(n-m)==1):
                return -1/2*[Jx,Jy,Jz][i]
        return 0
    
    return return_func

def random_hamiltonian(n,m,i,j):
    return complex(random(),random())/sqrt(2)
    


def basis_vectors(n):
    """
    Generator that yields the basis vectors
    with correct tensor dimensional form 
    """

    single_vectors = [basis(2,0),basis(2,1)]
    for selection in product(*([single_vectors]*n)):
        yield tensor(*selection)

def eff_dim(dens_oper:Qobj):
    """
    Returns effective dimension of mixed state:
    Via formula 1/Tr(rho^2)
    
    """
    dens_oper_sq = dens_oper**2
    return 1/(dens_oper_sq.tr())

def purity(dens_oper:Qobj):
    return (dens_oper**2).tr()

def gen_random_state(n):
    """
    Generate a random state
    of an n particle system.
    => dimension 2^n"
    """
    state = sum(complex(normal(),normal())*vector for vector in basis_vectors(n))
    return state.unit()

def make_hermitian(h):
    return 0.5*(h + h.dag())



def ket2dmR(state):
    n = len(state.dims[0])
    return Qobj(np.array(state)@np.array(state).T.conjugate(), dims = [[2]*n,[2]*n])



def max_seperation(h:Qobj):
    energies = h.eigenenergies()
    return max([abs(e1-e2) for e1,e2 in product(energies,energies)])

def get_equilibrated_dens_op(hamiltonian:Qobj, init_state:Qobj):
    energys,states = hamiltonian.eigenstates()
    return sum((abs(state.overlap(init_state)**2)*state*state.dag() for state in states))


def get_thermal_state(states):
    prob = 1/len(states)
    return prob*sum(state*state.dag() for state in states)

def get_ran_unit_norm_oper(n,_dims):
    return make_hermitian(rand_unitary(2**n, dims = _dims)) 


def hamiltonian_spin_interaction_component(alpha, n, m, i, j, N):
    """
    Generates a component of the hamiltonian due to a two spins interacting. 
    
    Params: n,m two particles which the interaction is between.
           i,j two particles spin components which interaction is between (0|x, 1|y, 2|z)
           N   number of particles in the system
    
    Returns: Tensor product of the N matrices (quantum obj)
    """

    if n==m:
        raise ValueError("Cannot have spin interaction with itself n==m")
    
    big_index = max(n,m)  #find out which of n and m comes first in the tensor product 
    small_index = min(n,m)

    #To construct: split product into three sections  I⊗ I⊗ ....⊗ |   σ_i   |  ⊗ ..... ⊗ | σ_j  | ⊗ I ⊗ I .....I 
    #                                                 <------1st---->          <-----2nd----->      <--------3rd ---->   
    
    fst_section = [identity(2)]*(small_index) 
    sec_section = [identity(2)]*(big_index-small_index-1)
    thrd_section = [identity(2)]*(N-big_index-1)
    operators = fst_section+[sigma[i]]+sec_section+[sigma[j]]+thrd_section

    return alpha*tensor(operators)

def hamiltonian_spin_on_site_component(beta, n, i, N):
    """
    Generates a component of the hamiltonian due to a spin interacting with an enviromental bias. 
    
    Params: n particle which is interacting.
           i particle spin component which is interacting (0|x, 1|y, 2|z)
           N   number of particles in the system
    
    Returns: Tensor product of the N matrices (quantum obj)
    """
    fst = [identity(2)]*n    
    sec = [identity(2)]*(N-n-1)

    return beta*tensor(fst+[sigma[i]]+sec)

def hamiltonian(alpha,beta,N):
    
    """
    Creates the hamiltonian for a given alpha function that specifies the structure of the system.

    Params: alpha - specifies the structure of the system and how a given particle interacts with others and the enviroment
            beta - specifes how each individual atom interacts with an external field/interaction
            N - number of particles 

    Returns: Hamiltonian Matrix
    """


    spin_components = (hamiltonian_spin_interaction_component(alpha(n,m,i,j),n,m,i,j,N) for n,m,i,j in product(range(N),range(N),range(3),range(3)) if n!=m )
    interaction_components = (hamiltonian_spin_on_site_component(beta(n,i),n,i,N) for n,i in product(range(N),range(3)))
            
    return sum(interaction_components)+sum(spin_components)
    


def energy_trace_comp_2d(h:Qobj, fraction, d):
    
    energys, states = h.eigenstates()
    
    num_energys = len(energys)
    
    random_indices = sample(range(num_energys),int(num_energys*fraction))
    
    energy_states = [(energys[ind],states[ind]) for ind in random_indices]
   
    x_vals =[]
    y_vals =[]
    
    dims = list(range(d))

    for pair1,pair2 in tqdm(product(energy_states,energy_states)):
       
        if(pair1!=pair2):
            energy_difference = abs(pair1[0]-pair2[0])
            substate1 = pair1[1].ptrace(dims)
            substate2 = pair2[1].ptrace(dims)
            trace_distance_val = tracedist(substate1,substate2)
            x_vals.append(energy_difference)
            y_vals.append(trace_distance_val) 

    return x_vals,y_vals 
    
def energy_trace_relative_n():
    
    difference = []
    n_range = range(5,13)
    
    for n in tqdm(n_range):
        difference.append(0.0)
        
        alpha1 = Heisenberg1dChainGen(-1,1/2,0,n)
        beta = lambda n,i :0.1
        h = hamiltonian(alpha1,beta,n) 
        h = h + 0.005*get_ran_unit_norm_oper(n,h.dims)
        
        energys, states = h.eigenstates()
        energy_pairs = sorted([(energy,state) for energy,state in zip(energys,states)], key = lambda x : x[0])
        band = energy_pairs[:int(len(energy_pairs)/10)]
        
        reduced_band = [(pair[0],pair[1].ptrace([0])) for pair in band]
        counter =0.0
       
        
        for energy_pair1 , energy_pair2 in product(reduced_band,reduced_band):
            if energy_pair1!=energy_pair2:
                difference[-1]+=tracedist(energy_pair1[1],energy_pair2[1])
                counter+=1
        
        difference[-1]/=(counter) #look at this seems to be where the problem lies
   
    plt.plot(n_range,difference)
    plt.xlabel("System spin number")
    plt.ylabel("Average Trace Distance")
    plt.show()


def energy_trace_fixed_n():
    
    difference = []
    n_range = range(5,13)
    
    for n in tqdm(n_range):
        difference.append(0.0)
        
        alpha1 = Heisenberg1dChainGen(-1,1/2,0,n)
        beta = lambda n,i :0.1
        h = hamiltonian(alpha1,beta,n) 
        h = h + 0.005*get_ran_unit_norm_oper(n,h.dims)
        
        energys, states = h.eigenstates()
        energy_pairs = sorted([(energy,state) for energy,state in zip(energys,states)], key = lambda x : x[0])
        band = energy_pairs[:20]
        
        reduced_band = [(pair[0],pair[1].ptrace([0])) for pair in band]
        counter =0.0
        

        for energy_pair1 , energy_pair2 in product(reduced_band,reduced_band):
            if energy_pair1!=energy_pair2:
                difference[-1]+=tracedist(energy_pair1[1],energy_pair2[1])
                counter+=1.0
        
        
        difference[-1]/=counter
            
                        
   
    plt.plot(n_range,difference)
    plt.xlabel("System spin number")
    plt.ylabel("Average Trace Distance")
    plt.show()



def equilibration_analyser(hamiltonian:Qobj, init_state:Qobj, time:int, steps:int, trace=[0]): 
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) # two rows, one column, first plot
    
    times = linspace(0,time,steps)
    results = mesolve(hamiltonian,init_state,times,[],[],options=Options(nsteps=1e6))
    
    equilibrated_dens_op = get_equilibrated_dens_op(hamiltonian,init_state)
    effective_dimension = eff_dim(equilibrated_dens_op)
    bound = 0.5*sqrt((2**len(trace))**2/effective_dimension)
    
    trace_distances = [tracedist(equilibrated_dens_op.ptrace(trace),state.ptrace(trace)) for state in results.states]
    bound_line = [bound for state in results.states]
    
    ax.plot(times,trace_distances,label="Trace-Distance")
    ax.plot(times,bound_line,label="Bound-Distance")
    plt.title(f"System: effective dimension {effective_dimension:.2f} and bound {bound:.2f} ")
    ax.set_xlabel(r"Time /$\hbar$s")
    ax.set_ylabel(r"$TrDist(\rho(t),\omega$)")
    plt.legend()
    plt.savefig("TestOld")
    
    

def energy_band_plot(hamiltonian,title_text):
    energys = hamiltonian.eigenenergies()
    energys_count = Counter(energys)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) # two rows, one column, first plot

    max_energy = max(energys)
    min_energy = min(energys)
    degeneracy = False
    
    text_shift = (max_energy-min_energy)/100
    
    for energy, degen in energys_count.items():
        color_val = 'b' if degen ==1 else "r"
        
        if degen >1 :
            ax.plot(linspace(0,9.3,20),[energy for i in linspace(0,9.3,20)], color = color_val)
            ax.text(9.4,energy-text_shift,f"Degen: {degen} fold")
            degeneracy = True
        else:
            ax.plot(range(10),[energy for i in range(10)], color = color_val)


    
    degen_patch = mpatches.Patch(color='red', label='Degenerate level')
    norm_patch = mpatches.Patch(color='blue', label='Non-Degenerate level')
    patches = [degen_patch, norm_patch] if degeneracy else [norm_patch]
    plt.legend(handles=patches, loc= "center left")
    ax.set_ylabel("Normalized energy value")
    
    ax.axes.get_xaxis().set_visible(False)
    ax.set_xlim([-5,15])
    plt.title(title_text)
    plt.show()


def thermalisation_analyser(equilibrated_dens_op , band, trace=[0]): 
    
    #TODO ADD IN GRAPH like equilibration between current state and thermal sate ! with bound plus tonys bound superimposed!
    #Todo add a ray version of get equilibrated dens_op!! aand biuuld better ecosysmte to interface ie states, maybe in new doc
    
    dR = len(band)

    equilibrated_sub_op = equilibrated_dens_op.ptrace(trace)
    deff = eff_dim(equilibrated_dens_op)
   
    thermal_state = get_thermal_state(band)
    thermal_sub_state = thermal_state.ptrace(trace)

    epsilon  = (1/dR)*sum(tracedist(thermal_sub_state, state.ptrace(trace)) for state in band)
    actual_distance = tracedist(thermal_sub_state,equilibrated_sub_op)
   
    bound =sqrt(epsilon*(dR-deff)/deff)

    print(f"Actual value {actual_distance:.3f} with bound predicting {bound} with epsilon {epsilon:.3f}, dR {dR:.3f} , deff {deff:.3f}")