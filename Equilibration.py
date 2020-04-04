import ray
from QuantumLib import *
from sys import argv
from os import environ
from time import time
from qutip import tensor
from qutip.states import basis
 


n = int(argv[1])
processors = int(argv[2])
epsilon = 0.1

environ["OMP_NUM_THREADS"] = str(processors) # export OMP_NUM_THREADS=4
environ["OPENBLAS_NUM_THREADS"] = str(processors) # export OPENBLAS_NUM_THREADS=4 
environ["MKL_NUM_THREADS"] = str(processors) # export MKL_NUM_THREADS=6
environ["VECLIB_MAXIMUM_THREADS"] =str(processors) # export VECLIB_MAXIMUM_THREADS=4
environ["NUMEXPR_NUM_THREADS"] = str(processors) # export NUMEXPR_NUM_THREADS=6


ray.init()

start_t = time()


state1 = tensor([basis(2,0)]*n)

beta1 = lambda n, m: 1
beta0 = lambda n, m: 0

alpha1 = Heisenberg1dRingGen(-1, 0.5, 0.1, n)

H1 = hamiltonian(alpha1, beta0, n)
H1=H1/H1.norm()

H2 = hamiltonian(alpha1, beta1, n)
H2 = H2/H2.norm()

H3 = hamiltonian(alpha1, beta0, n)
H3 = H3/H3.norm()

Pertubation = random_herm_oper(H3.dims,n)
Pertubation = Pertubation/Pertubation.norm()

H3 = H3 + Pertubation
H3 = H3/H3.norm()

H4 = Pertubation


assert(H1.isherm)
assert(H2.isherm)
assert(H3.isherm)
assert(H4.isherm)

energys1, states1 = H1.eigenstates()
energys2, states2 = H2.eigenstates()
energys3, states3 = H3.eigenstates()
energys4, states4 = H4.eigenstates()


equilibration_analyser_p(energys1, states1, state1, 1e5, 200, f"H1-{n}", _proc=processors)
equilibration_analyser_p(energys2, states2, state1, 1e5, 200, f"H2-{n}", _proc=processors)
equilibration_analyser_p(energys3, states3, state1, 1e5, 200, f"H3-{n}", _proc=processors)
equilibration_analyser_p(energys4, states4, state1, 1e5, 200, f"H4-{n}", _proc=processors)



end_t = time()

print(f"Completed {n} spins with {processors} processors taking {end_t-start_t} seconds")