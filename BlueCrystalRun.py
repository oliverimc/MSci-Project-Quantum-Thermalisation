import ray
from time import time 
from matplotlib import pyplot as plt
from numpy.linalg import norm
from Quantum_functions import *
import numpy as np
from QuantumPFunctions import *
from qutip import *
from sys import argv
import os


n = 10
processors = int(argv[1])



os.environ["OMP_NUM_THREADS"] = str(processors) # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = str(processors) # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = str(processors) # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] =str(processors) # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(processors) # export NUMEXPR_NUM_THREADS=6


ray.init()


state1 = tensor([basis(2,0)]*n)

alpha1 = Heisenberg1dRingGen(1,1,1,n)
alpha0 = lambda n,m,i,j: 0
beta1 = lambda n,i : [0,0,1][i]
beta0 = lambda n,i :0

self_interaction = hamiltonian(alpha1,beta0,n)
self_interaction = self_interaction/self_interaction.norm()
external_interaction = hamiltonian(alpha0,beta1,n)
external_interaction = external_interaction/external_interaction.norm()


perturbation: Qobj = make_hermitian(rand_unitary(2**n, dims = self_interaction.dims))
perturbation = perturbation/perturbation.norm()
epsilon = 0.3

H1 = epsilon*self_interaction + external_interaction + epsilon**2*perturbation



start = time()
xs,ys = energy_trace_compare_p(H1,1,1,proc =processors)
end = time()


print(len(xs))

plt.scatter(xs,ys,s=1)
plt.savefig("Ten_SPIN")

print(f"Finished executing completed in {end-start}s")
