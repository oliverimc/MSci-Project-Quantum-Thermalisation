import ray
from QuantumLib import *
from sys import argv
from os import environ
from time import time
from qutip import tensor
from qutip.states import basis
 


n = int(argv[1])
processors = int(argv[2])
mem = int(argv[2])
epsilon = 0.1
gig = 1024*1024*1024

ray.init(memory = mem*gig, object_store_memory=0.8*float(mem)*gig)

start_t = time()


state1 = tensor([basis(2,0)]*n)

beta1 = lambda n, m: 1
beta0 = lambda n, m: 0

alpha1 = Heisenberg1dRingGen(-1, 0.5, 0.1, n)

H1 = hamiltonian(alpha1, beta1, n)
H1=H1/H1.norm()

assert(H1.isherm)


energys1, states1 = H1.eigenstates()

print(f"Finished Setup Now Starting Ray Section: after {time()-start_t}s")

equilibration_analyser_p(energys1, states1, state1, 1e7, 200, f"ExtInt-{n}", _proc=processors)




end_t = time()

print(f"Completed {n} spins with {processors} processors taking {end_t-start_t} seconds")