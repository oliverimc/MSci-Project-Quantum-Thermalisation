from Quantum_functions import *
from qutip.random_objects import rand_unitary



#ETH theorem test near energys give near energy densitys 


state1 = tensor([basis(2,0)]*n)
alpha1 = Heisenberg1dChainGen(-1,1/2,0,n)
beta = lambda n,i :0.1
perturbation = rand_unitary(n)
h = hamiltonian(alpha1,beta,n)
markers = ['.','o','v','^', '>', '<','8','s','+']

for d in range(0,n,2):
    xs,ys = energy_trace_comp_2d(h,0.3,d+1)
    plt.scatter(xs,ys,s=5,marker=markers[d], label =str(d+1))
    

plt.xlabel("Energy Difference")
plt.ylabel("Trace Distance")
plt.legend()
plt.show()
