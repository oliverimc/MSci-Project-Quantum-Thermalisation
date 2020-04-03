import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from numpy import linspace, exp, sqrt, dot
import ray
from qutip import Qobj, sigmax, sigmay, sigmaz, identity, tensor
from qutip.metrics import tracedist
from qutip.states import basis
from qutip.random_objects import rand_unitary_haar
from scipy.sparse import csr_matrix

from itertools import chain, product
from time import time
sigma = [sigmax(),sigmay(),sigmaz()]

def Heisenberg1dRingGen(Jx, Jy, Jz, N):
    
    def return_func(n,m,i,j):
        if (i==j):
            if(abs(n-m)==1):
                return -1/2*[Jx,Jy,Jz][i]
            if((n==(N-1) and m==0) or (n==0 and m==(N-1))):
                return -1/2*[Jx, Jy, Jz][i]
        return 0
    
    return return_func


def Heisenberg1dChainGen(Jx, Jy, Jz, N):

    def return_func(n,m,i,j):
        if (i==j):
            if(abs(n-m)==1):
                return -1/2*[Jx, Jy, Jz][i]
        return 0
    
    return return_func


def hamiltonian_spin_interaction_component(alpha, n, m, i, j, N):
    """
    Generates a component of the hamiltonian due to a two spins interacting. 
    
    Params: n,m two particles which the interaction is between.
           i,j two particles spin components which interaction is between (0|x, 1|y, 2|z)
           N   number of particles in the system
    
    Returns: Tensor product of the N matrices (quantum obj)
    """

    if n == m:
        raise ValueError("Cannot have spin interaction with itself n==m")
    
    big_index = max(n, m)  #find out which of n and m comes first in the tensor product 
    small_index = min(n, m)

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


def hamiltonian(alpha, beta, N):
    
    """
    Creates the hamiltonian for a given alpha function that specifies the structure of the system.

    Params: alpha - specifies the structure of the system and how a given particle interacts with others and the enviroment
            beta - specifes how each individual atom interacts with an external field/interaction
            N - number of particles 

    Returns: Hamiltonian Matrix
    """


    spin_components = (hamiltonian_spin_interaction_component(alpha(n,m,i,j),n,m,i,j,N) for n,m,i,j in product(range(N), range(N), range(3), range(3)) if n != m)
    interaction_components = (hamiltonian_spin_on_site_component(beta(n,i),n,i,N) for n,i in product(range(N), range(3)))
            
    return sum(interaction_components)+sum(spin_components)
    

def basis_vectors(n):
    """
    Generator that yields the basis vectors
    with correct tensor dimensional form 
    """

    single_vectors = [basis(2,0) ,basis(2,1)]
    for selection in product(*([single_vectors]*n)):
        yield tensor(*selection)


def get_thermal_state(states):
    prob = 1/len(states)
    return prob*sum(state*state.dag() for state in states)


def eff_dim(dens_oper):
    """
    Returns effective dimension of mixed state:
    Via formula 1/Tr(rho^2)

    """
    dens_oper_sq = dens_oper**2
    return 1/(dens_oper_sq.tr())

def make_hermitian(h):
    return 0.5*(h + h.dag())


def random_herm_oper(_dims, n):
    return make_hermitian(rand_unitary_haar(2**n, dims =_dims))


def ket2dmR(state):
    
    return csr_matrix(state.data)@csr_matrix(state.data).transpose().conjugate()


@ray.remote
def eq_terms(states, coefs, start, end):
    
    return sum(abs(coefs[i]**2)*ket2dmR(states[i]) for i in range(start, end))


def get_equilibrated_dens_op_P(eigstates, coefs, n , proc=4):

    number = len(eigstates)//proc
  
    s_id = ray.put(eigstates)
    c_id = ray.put(coefs)

    results = [eq_terms.remote(s_id, c_id, i*number, (i+1)*number) for i in range(proc-1)]
    results.append(eq_terms.remote(s_id, c_id, (proc-1)*number, len(eigstates)))

    results_val = ray.get(results)
    
    return Qobj(sum(results_val), dims=[[2]*n, [2]*n])
    

def phase(t, E):
   
    return exp(-1j*E*t)

@ray.remote
def time_step(coef, eigstates, eigenenergies, func, times, strt, stop):
    
    result = []
    
    for time in times[strt:stop]:
        state = sum(phase(energy, time)*coef[ind]*eigstates[ind] for ind, energy in enumerate(eigenenergies))
        result.append(func(state))
    
    return result


def simulate(energys, eigstates, coef, t_start, t_end, steps, ret_func=lambda x: x, proc=4):
    
    times = linspace(t_start, t_end, steps)

    energys_id = ray.put(energys)
    states_id = ray.put(eigstates)
    coef_id = ray.put(coef)
    times_id = ray.put(times)

    num = steps//proc

    result = [time_step.remote(coef_id, states_id, energys_id, ret_func, times_id, num*i, num*(i+1)) for i in range(proc-1)]
    result.append(time_step.remote(coef_id, states_id, energys_id, ret_func, times_id, (proc-1)*num, steps))
    
    results = ray.get(result)
    
    return list(chain(*results))
    
    
def equilibration_analyser_p(energys, eigstates, init_state, start, stop, steps, name, trace=[0], _proc=4):
    
    start =time()
    coef = [init_state.overlap(state) for state in eigstates]

    n= len(init_state.dims[0])
    subsys_trace = trace
    bath_trace = [ dim for dim in range(n) if dim not in subsys_trace]
    
    print("Basics")
    print(time()-start)
                               
    equilibrated_dens_op = get_equilibrated_dens_op_P(eigstates, coef, n, proc=_proc)
    
    print("EqDensOp")
    print(time()-start)
    
    effective_dimension_sys = eff_dim(equilibrated_dens_op)
    effective_dimension_bath = eff_dim(equilibrated_dens_op.ptrace(bath_trace))
    #now we have the actual effective dimension trace over as can't do it before or messes up
    
    equilibrated_dens_op = equilibrated_dens_op.ptrace(subsys_trace)
    
    print("Tracing")
    print(time()-start)
    
    bound_loose = 0.5*sqrt((2**len(subsys_trace))**2/effective_dimension_sys)
    bound_tight = 0.5*sqrt(2**len(subsys_trace)/effective_dimension_bath)

    print("Bounds")
    print(time()-start)
    
    trace_dist_compare = lambda state: tracedist(equilibrated_dens_op, state.ptrace(trace))
    trace_distances = simulate(energys, eigstates, coef, start, stop, steps, ret_func=trace_dist_compare, proc=_proc)
    
    print("Simulation")
    print(time()-start)
    
    times = [start+step*(stop-start)/steps for step in range(steps)]
    
    bound_line_loose = [bound_loose]*steps
    bound_line_tight = [bound_tight]*steps
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(times, trace_distances, label="Trace-Distance")
    print("Plot1")
    print(time()-start)
    
    ax.plot(times, bound_line_loose, label="Bound-Distance (loose)")
    ax.plot(times, bound_line_tight, label="Bound-Distance (tight)")
    print("Plot2")
    print(time()-start)
    plt.title(f"System: effective dimension {effective_dimension_sys:.2f}. Bound) loose:{bound_loose:.2f} tight:{bound_tight:.2f}")
    ax.set_xlabel(r"Time /$\hbar$s")
    ax.set_ylabel(r"$TrDist(\rho(t),\omega$)")
    plt.legend()
    plt.savefig(name)
    print("Finished")
    print(time()-start)
    
    
    
    

