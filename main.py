from qutip import sigmax, sigmay, sigmaz, identity, tensor
from qutip.mesolve import mesolve
from qutip.essolve import essolve
from qutip import Qobj
from qutip.metrics import tracedist
from qutip.states import basis,ket2dm

from numpy import linspace
from itertools import product
from time import time
from matplotlib import pyplot as plt

#   |____                         ___
#   |___         ---------          |
#   |   |        ---------          | 
#   |   |                        ___|___
# hbar ==1   


sigma = [sigmax(),sigmay(),sigmaz()]


#TODO FIX MEMORY ERROR by replacing sum(list) with iteration sum
#TODO watch system equilibrate
#TODO effective dimension = 1/prob(En)**2 -> large for equilibration

def alpha(n,m,i,j):
    
    if (m-n)==1 and i==j:
        return [0,2,1][i]
    else:
        return 0
    
def eff_dim(dens_oper):
    """
    Returns effective dimension of mixed state:
    Via formula 1/Tr(rho^2)
    
    """
    return 1/((dens_oper**2).tr())

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


"""
#example way to evolve a state with qutip can also use essolve 
H = hamiltonian(alpha,lambda n,i: 0, 3)
psi0 = tensor(basis(2,1),basis(2,0),basis(2,0))
times = linspace(0,1,10)
result = mesolve(H,psi0,times,[],[])

print(result.states)
"""

psi:Qobj = tensor(basis(2,0),basis(2,0))


psi0 = tensor(basis(2,0),basis(2,1))
psi1 = tensor(basis(2,0),basis(2,0))

p1 = ket2dm(psi0)
p2 = ket2dm(psi1)

print(tracedist(psi1,psi0))

