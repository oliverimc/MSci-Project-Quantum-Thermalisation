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


Pertubation = random_herm_oper([[2]*n,[2]*n],n)

H1 = Pertubation
H1 = H1/H1.norm()

assert(H1.isherm)


energys1, states1 = H1.eigenstates()

print(f"Finished Setup Now Starting Ray Section: after {time()-start_t}s")

equilibration_analyser_p(energys1, states1, state1, 1e7, 200, f"Pert-{n}", _proc=processors)




end_t = time()

print(f"Completed {n} spins with {processors} processors taking {end_t-start_t} seconds")