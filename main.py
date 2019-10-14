from qutip import sigmax, sigmay, sigmaz, identity
from qutip import tensor
from numpy import array, zeros
from itertools import product


sigma = [sigmax(),sigmay(),sigmaz()]



def alpha(n,m,i,j):
    if (m-n)==1:
        return 1
    else:
        return 0
    


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

def hamiltonian_spin_bias_component(beta, n, i, N):
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

    for n,i in product(range(N),range(N)):
        spin_components.append(hamiltonian_spin_bias_component(beta(n,i),n,i,N))
     
    return sum(spin_components)
    
    

print(hamiltonian(alpha,lambda n,i: 0, 2))


