from qutip import sigmax, sigmay, sigmaz, identity, tensor
from qutip.mesolve import mesolve
from qutip.essolve import essolve
from qutip import Qobj
from qutip.metrics import tracedist
from qutip.states import basis,ket2dm
from qutip import Options

from numpy import linspace,array,sqrt,zeros
from numpy.random import normal
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

import sys

from time import time
from itertools import product



#   |____                         ___
#   |___         ---------          |
#   |   |        ---------          | 
#   |   |                        ___|___
# hbar ==1   


#purity = Tr(rho^2) 1->1/d where 1/d is maximally mixed 

sigma = [sigmax(),sigmay(),sigmaz()]


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
    return normal()


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

def gen_random_state(n):
    """
    Generate a random state
    of an n particle system.
    => dimension 2^n"
    """
    state = sum([complex(normal(),normal())*vector for vector in basis_vectors(n)])
    return state.unit()

def get_equilibrated_dens_op(hamiltonian:Qobj, init_state:Qobj):
    energys,states = hamiltonian.eigenstates()
    return sum([abs(state.overlap(init_state)**2)*state*state.dag() for state in states])
   

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
    
    spin_components =[]

    for n,m,i,j in product(range(N),range(N),range(3),range(3)):
        if n!=m:
            spin_components.append(hamiltonian_spin_interaction_component(alpha(n,m,i,j),n,m,i,j,N))

    for n,i in product(range(N),range(3)):
        spin_components.append(hamiltonian_spin_on_site_component(beta(n,i),n,i,N))
     
    return sum(spin_components)


def energy_trace_comp_3d(energy_pairs):
   
    E =[]
    Eprime =[]
    trace_dist =[]
    
    
    for energy_pair1, energy_pair2 in tqdm(product(energy_pairs,energy_pairs)):
        
        
        e1,s1 = energy_pair1
        e2,s2 = energy_pair2

        if e1!=e2 and s1!=s2:

            density_op1 = ket2dm(s1)
            density_op2 = ket2dm(s2)
            
            red_dens1 = density_op1.ptrace(0)
            red_dens2 = density_op2.ptrace(0)
            
            E.append(e1)
            Eprime.append(e2)
            trace_dist.append(tracedist(red_dens1,red_dens2))
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    tri = mtri.Triangulation(E, Eprime)
    
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.set_xlabel("E Value")
    ax.set_ylabel("E' Value")
    ax.set_zlabel("Trace distance")
    ax.set_title(f"Trace distance of energy density operators: {len(E)} pairs")
    ax.plot_trisurf(E,Eprime,trace_dist)
    
    plt.show()

def energy_trace_comp_heat(energy_pairs):
   #does it work??
   #TODO look at lower energy eg lowest 10% by value E <0
    E =[]
    Eprime =[]
    trace_dist ={}
    
    for energy_pair1, energy_pair2 in tqdm(product(energy_pairs,energy_pairs)):
        
        e1,s1 = energy_pair1
        e2,s2 = energy_pair2

        if e1!=e2 and s1!=s2:

            density_op1 = ket2dm(s1)
            density_op2 = ket2dm(s2)
            
            red_dens1 = density_op1.ptrace(0)
            red_dens2 = density_op2.ptrace(0)
            
            E.append(e1)
            Eprime.append(e2)
            trace_dist[str(e1)+' '+str(e2)]=tracedist(red_dens1,red_dens2)
    
    data = zeros((len(E),len(E)))
    
    for x,ex in enumerate(E):
        for y,ey in enumerate(Eprime):
            if ex!=ey:
                data[x][y]=trace_dist[str(ex)+' '+str(ey)]
            else:
                data[x][y]=0
    
    plt.imshow(data,interpolation ="nearest")
    plt.colorbar()
    plt.show()

def energy_trace_comp_2d(H:Qobj, fraction, d, energy_diff =100):
    energys, states = H.eigenstates()
   
    x_vals =[]
    y_vals =[]
    energy_pairs = sorted([(energy,state) for energy,state in zip(energys,states)], key = lambda x : x[0])
    energy_pairs_range = energy_pairs[:int(len(energy_pairs)*fraction)]
    
    dims = list(range(d))

    for pair1,pair2 in tqdm(product(energy_pairs_range,energy_pairs_range)):
        if(pair1!=pair2):
            energy_difference = abs(pair1[0]-pair2[0])
            if(energy_difference<energy_diff):
                substate1 = ket2dm(pair1[1]).ptrace(dims)
                substate2 = ket2dm(pair2[1]).ptrace(dims)
                trace_distance_val = tracedist(substate1,substate2)
                x_vals.append(energy_difference)
                y_vals.append(trace_distance_val) 

    return x_vals,y_vals 
    






def equilibration_analyser(hamiltonian:Qobj, init_state:Qobj, time:int,steps:int, trace=[0]): 
    
    times = linspace(0,time,steps)
    results = mesolve(hamiltonian,init_state,times,[],[],options=Options(nsteps=1e6))
    
    equilibrated_dens_op = get_equilibrated_dens_op(hamiltonian,init_state)
    effective_dimension = eff_dim(equilibrated_dens_op)
    bound = 0.5*sqrt(2**len(trace)**2/effective_dimension)
    
    trace_distances = [tracedist(equilibrated_dens_op.ptrace(trace),state.ptrace(trace)) for state in results.states]
    bound_line = [bound for state in results.states]
    
    plt.plot(times,trace_distances,label="Trace-Distance")
    plt.plot(times,bound_line,label="Bound-Distance")
    plt.title(f"System: effective dimension {effective_dimension:.2f} and bound {bound:.2f} ")
    plt.xlabel(r"Time /$\hbar$s")
    plt.ylabel(r"$TrDist(\rho(t),\omega$)")
    plt.legend()
    plt.show()  
    

n=9

state1 = tensor([basis(2,0)]*n)
alpha1 = Heisenberg1dChainGen(-1,1/2,0,n)
beta = lambda n,i :0
h = hamiltonian(alpha1,beta,n) + 0.05*hamiltonian(random_hamiltonian,beta,n)
markers = ['.','o','v','^', '>', '<','8','s','+']

for d in range(0,n,2):
    xs,ys = energy_trace_comp_2d(h,0.2,d+1)
    plt.scatter(xs,ys,s=5,marker=markers[d], label =str(d+1))
    

plt.xlabel("Energy Difference")
plt.ylabel("Trace Distance")
plt.legend()
plt.show()

 










""" state2 = tensor([basis(2,0)]*n)
alpha2 = Heisenberg1dRingGen(-1,1,1,n)
H2 = hamiltonian(random_hamiltonian,lambda n,i: 0, n)

state3 = tensor([basis(2,0)]*n)
alpha3 = Heisenberg1dChainGen(-1,0,1,n)
H3 = hamiltonian(alpha3,lambda n,i: 0, n)



#for energy in get_sig_dif_states(H):
#    print(energy[0])
#    print(energy[1])


#energy_trace_comp_heat(get_sig_dif_states((H)))
equilibration_analyser(H1,state1,50,200)
equilibration_analyser(H2,state2,50,200)
equilibration_analyser(H3,state3,50,200) """



