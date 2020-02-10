from Quantum_functions import *





n = 8
state1 = tensor([basis(2,0)]*n)

alpha1 = Heisenberg1dRingGen(1,1,1,n)
alpha0 = lambda n,m,i,j: 0
beta1 = lambda n,i :1
beta0 = lambda n,i :0

self_interaction = hamiltonian(alpha1,beta0,n)
self_interaction = self_interaction/self_interaction.norm()
external_interaction = hamiltonian(alpha0,beta1,n)
external_interaction = external_interaction/external_interaction.norm()


perturbation: Qobj = make_hermitian(rand_unitary(2**n, dims = self_interaction.dims))
perturbation = perturbation/perturbation.norm()
epsilon = 0.3

H1 = epsilon*self_interaction + external_interaction + epsilon**2*perturbation

markers = ['.','o','v','^', '>', '<','8','s','+']

#PUT THIS INTO JUPITER NOTEBOOK!

"""

#find h1 from report just make same effective dimension
state2 = tensor([basis(2,0)]*n)
H2 = hamiltonian(random_hamiltonian,lambda n,i: 0, n) #?investigatge

state3 = tensor([basis(2,0)]*n)
alpha3 = Heisenberg1dChainGen(-1,0,1,n)
H3 = hamiltonian(alpha3,lambda n,i: 0, n)


#Test equilibration bounds with different hamiltonians 
equilibration_analyser(H1,state1,50,200)
equilibration_analyser(H2,state2,50,200)
equilibration_analyser(H3,state3,50,200)


#Band of energies average trace dist as n increases relative band incr wit n fixed-> stays same
#energy_trace_fixed_n()
#energy_trace_relative_n()


#ETH test different cases
n = 8
state1 = tensor([basis(2,0)]*n)

alpha1 = Heisenberg1dRingGen(1,1,1,n)
alpha0 = lambda n,m,i,j: 0
beta1 = lambda n,i :1
beta0 = lambda n,i :0

self_interaction = hamiltonian(alpha1,beta0,n)
self_interaction = self_interaction/self_interaction.norm()
external_interaction = hamiltonian(alpha0,beta1,n)
external_interaction = external_interaction/external_interaction.norm()


perturbation: Qobj = make_hermitian(rand_unitary(2**n, dims = self_interaction.dims))
perturbation = perturbation/perturbation.norm()
epsilon = 0.3

H1 = epsilon*self_interaction + external_interaction + epsilon**2*perturbation
xs,ys = energy_trace_comp_2d(H1,0.3,1)
plt.scatter(xs,ys,s=5,marker=markers[d], label =str(1))
plt.xlabel("Energy Difference")
plt.ylabel("Trace Distance")
plt.legend()
plt.show()

"""


#look into purity tr(rho^2)
#look into purely interaction
#try 2 spins as well for abo
#look into balance between interaction and external
#look into analyticaly showing this with purely analytical haar measure eg average for band  =<trB(psi><psi)>>psi member of subspace = trb(maxim mixed)


#ETH theorem test near energys give near energy densitys 
for d in range(0,1):
    xs,ys = energy_trace_comp_2d(H1,0.3,d+1)
    plt.scatter(xs,ys,s=5,marker=markers[d], label =str(d+1))
    

plt.xlabel("Energy Difference")
plt.ylabel("Trace Distance")
plt.legend()
plt.show()
