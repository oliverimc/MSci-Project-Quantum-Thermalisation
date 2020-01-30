from Quantum_functions import *
from qutip.random_objects import rand_unitary



#ETH theorem test near energys give near energy densitys 

n = 8
state1 = tensor([basis(2,0)]*n)
alpha1 = Heisenberg1dChainGen(-1,1/2,0,n)
beta = lambda n,i :0.1
H1 = hamiltonian(alpha1,beta,n)
perturbation: Qobj = make_hermitian(rand_unitary(2**n, dims = H1.dims))
perturbation = perturbation/perturbation.norm()
epsilon = 0.05
H1 = H1 + perturbation*epsilon


markers = ['.','o','v','^', '>', '<','8','s','+']


"""
state2 = tensor([basis(2,0)]*n)
alpha2 = Heisenberg1dRingGen(-1,1,1,n)
H2 = hamiltonian(random_hamiltonian,lambda n,i: 0, n) #?investigatge

state3 = tensor([basis(2,0)]*n)
alpha3 = Heisenberg1dChainGen(-1,0,1,n)
H3 = hamiltonian(alpha3,lambda n,i: 0, n)


#Test equilibration bounds with different hamiltonians 
equilibration_analyser(H1,state1,50,200)
equilibration_analyser(H2,state2,50,200)
equilibration_analyser(H3,state3,50,200)
"""


for d in range(0,3):
    xs,ys = energy_trace_comp_2d(H1,0.3,d+1)
    plt.scatter(xs,ys,s=5,marker=markers[d], label =str(d+1))
    

plt.xlabel("Energy Difference")
plt.ylabel("Trace Distance")
plt.legend()
plt.show()
