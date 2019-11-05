from qutip import sigmax, sigmay, sigmaz, identity, tensor
from qutip.mesolve import mesolve
from qutip.essolve import essolve
from qutip import Qobj
from qutip.metrics import tracedist
from qutip.states import basis,ket2dm
from qutip import Options

from numpy import linspace,array,sqrt
from numpy.random import normal
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

from time import time
from itertools import product



#   |____                         ___
#   |___         ---------          |
#   |   |        ---------          | 
#   |   |                        ___|___
# hbar ==1   


sigma = [sigmax(),sigmay(),sigmaz()]


#TODO FIX MEMORY ERROR by replacing sum(list) with iteration sum
#TODO watch system equilibrate


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


def energy_trace_dist_compare(hamiltonian:Qobj):
    #Todo look into what is going wrong

    E =[]
    Eprime =[]
    trace_dist =[]
    
    energies, states = hamiltonian.eigenstates()
    
    for energy_pair1, energy_pair2 in tqdm(product(zip(energies,states),zip(energies,states))):
        
        e1,s1 = energy_pair1
        e2,s2 = energy_pair2
        
        if e1!=e2:
            density_op1 = ket2dm(s1)
            density_op2 = ket2dm(s2)
            red_dens1 = density_op1.ptrace(0)
            red_dens2 = density_op2.ptrace(0)
            E.append(e1)
            Eprime.append(e2)
            trace_dist.append(tracedist(red_dens1,red_dens2))
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    #tri = mtri.Triangulation(E, Eprime)
    
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.set_xlabel("E Value")
    ax.set_ylabel("E' Value")
    ax.set_zlabel("Trace distance")
    ax.set_title(f"Trace distance of energy density operators: {len(E)} pairs")
    ax.plot_trisurf(E,Eprime,trace_dist)
    
    plt.show()

def equilibration_analyser(hamiltonian:Qobj, init_state:Qobj, time:int,steps:int, trace=[0]): 
    #TODO THINK OVER WITH PAPER
    times = linspace(0,time,steps)
    results = mesolve(hamiltonian,init_state,times,[],[],options=Options(nsteps=1e6))
    equilibrated_state = get_equilibrated_dens_op(hamiltonian,init_state)
    effective_dimension = eff_dim(equilibrated_state)
    bound = 0.5*sqrt(2**len(trace)**2/effective_dimension)
    trace_distances = [tracedist(equilibrated_state.ptrace(trace),state.ptrace(trace)) for state in results.states]
    plt.plot(times,trace_distances)
    plt.title(f"System with effective dimension {effective_dimension:.2f} and bound {bound} ")
    plt.xlabel("Time / hbar")
    plt.ylabel("Trace-distance rho_eq - rho")
    plt.show()  
    


n=7
state:Qobj = gen_random_state(n)
alpha = Heisenberg1dRingGen(0,0,1,n)
H = hamiltonian(random_hamiltonian,lambda n,i: 0, n)


equilibration_analyser(H,state,5,100)



