import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
import base64

import random
import time
import gc
import os
global _globals
_globals = {
	'debug':None,
	'dist':None,
	'swanlab':None,
	'device':None,
	'bufsize':None,
	'gamma':None,
	'tau':None,
	'num_envs':None,
}
args = None

BEEP = lambda msg: os.system(f'say "{msg}"') # tested only on macOS

def get_device(prefer_cuda=True):
	"""
	Get the best available device with CUDA fallback to CPU.
	
	Args:
		prefer_cuda (bool): Whether to prefer CUDA if available
		
	Returns:
		str: 'cuda' if CUDA is available and working, 'cpu' otherwise
	"""
	if not prefer_cuda:
		return 'cpu'
		
	try:
		if torch.cuda.is_available():
			# Test CUDA functionality
			test_tensor = torch.tensor([1.0], device='cuda')
			test_result = test_tensor + 1.0
			del test_tensor, test_result
			torch.cuda.empty_cache()
			return 'cuda'
	except Exception as e:
		print(f"CUDA test failed, falling back to CPU: {e}")
		
	return 'cpu'

# set seed
def set_seed(seed: int):
	gc.collect()
	try:
		torch.cuda.empty_cache()
		torch.cuda.manual_seed(seed)
	except RuntimeError:
		# CUDA not available, skip CUDA-specific operations
		pass
	
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

from qiskit.quantum_info import Statevector, random_statevector, random_clifford

def list2sv(lst):
	cmpx = [complex(re, im) for re, im in zip(lst, lst[len(lst)//2:])]
	return Statevector(cmpx)

def np2tableau(lst):
	# print(lst)
	return stim.Tableau.from_numpy(x2x=lst[0], x2z=lst[1], z2x=lst[2], z2z=lst[3], x_signs=lst[4], z_signs=lst[5])
	# return stim.Tableau.from_state_vector(np.array(cmpx, dtype=np.complex64), endian='little')

def make_random(dims, seed, dist='general') -> Statevector:
	if dist == 'logical':
		return random_logical_statevector(dims, seed)
	elif dist == 'clifford':
		sv = Statevector([1.]+[0.]*(dims-1))
		return sv.evolve(random_clifford(int(np.log2(dims)+1e-6), seed))
	else:
		return random_statevector(dims, seed)
	
############# ignore ###############
def random_logical_statevector(dims, seed) -> Statevector:
	if isinstance(seed, np.random.Generator):
		# print('YESS')
		gen = seed
	else:
		gen = np.random.default_rng(seed)
	bitvec = gen.integers(dims)
	# print(seed, bitvec, dims)
	res = np.zeros(dims)
	res[bitvec] = 1.
	res = res/np.linalg.norm(res)
	res = Statevector(res)
	# print(res)
	return res

def prepare_testbench(num_exp: int, qubits: int, testfile: str, seed: int, overwrite=True, dist='general') -> list:
	basis_sz = 2**qubits
	test_set = []
	# generate random target states
	if testfile in ['', 'none'] or not os.path.exists(testfile) or overwrite:
		fname = testfile
		if testfile in ['', 'none']: fname = f'random-test.{qubits}'
		f = open(fname, 'w')
		for i in range(num_exp):
			random_test = make_random(basis_sz, seed=(seed+i if seed else None), dist=dist).data
			f.write(str(random_test.tolist())+' # randomly generated\n')
			
			real, imag = random_test.real.astype(np.float32), random_test.imag.astype(np.float32)
			random_test = np.concatenate((real, imag))
			test_set.append(random_test.tolist())
		f.close()

	else:
		with open(testfile) as f:
			for i, state in enumerate(f):
				if len(test_set) >= num_exp: break
				l = np.array(eval(state.split('#')[0].rstrip()))
				if len(l) == basis_sz:
					l /= np.linalg.norm(l)
					real, imag = l.real.astype(np.float32), l.imag.astype(np.float32)
					l = np.concatenate((real, imag))
					test_set.append(l)
				else:
					raise ValueError(f'{i+1}th state in file is not of the right size, expected size {basis_sz}, found size {len(l)}'); return None

	return test_set

############ UNSEEDED (uses system noise internally) ################
def prepare_testbench_tableau(num_exp: int, qubits: int, testfile: str, seed: int, overwrite=False, dist='clifford', **kwargs) -> list:
	test_set = []

	def tableau2str(tabl: stim.Tableau):
		return '[' + ', '.join(map(lambda s: f'"{str(s)}"', tabl.to_stabilizers(canonicalize=True))) + ']'
	def str2tableau(s: str):
		return stim.Tableau.from_stabilizers(list(map(stim.PauliString, eval(s))))
	# generate random target states/all states of a particular size
	if testfile in ['', 'none'] or not os.path.exists(testfile) or overwrite:
		fname = testfile
		if testfile in ['', 'none']: fname = f'random-test-{qubits}q'
		f = open(fname, 'w')
		print(num_exp, flush=True)
		if num_exp == -1:
			j = 0
			# generate all possible tableaus (iterate)
			for tabl in stim.Tableau.iter_all(qubits, unsigned=True):
				j +=1
				print(j, flush=True)
				f.write(tableau2str(tabl)+' # randomly generated\n')
				test_set.append(tabl)
		else:
			for i in range(num_exp):
				random_test = make_random_tableau(qubits, dist=dist, **kwargs)
				f.write(tableau2str(random_test)+' # randomly generated\n')
				test_set.append(random_test)
		f.close()
	else:
		with open(testfile) as f:
			for i, state in enumerate(f):
				if len(test_set) >= num_exp: break
				try:
					tableau = str2tableau(state.split('#')[0].rstrip())
					test_set.append(tableau)
				except Exception as interrupt:
					print(interrupt)
					raise ValueError(f'(Most probably) {i+1}th state in file is not of the right size, expected size {qubits}')
	return test_set

import inspect
def debug():
	# print current line
	print(f'{inspect.currentframe().f_back.f_lineno}')

flags = set()
def print_once(msg, use):
	if use in flags: return
	print(msg, flush=True)
	flags.add(use)

# random clifford state
import stim

def make_random_tableau(nqbits: int, dist='clifford', **kwargs) -> stim.Tableau:
	# print("duhh, passed is", nqbits)
	if dist == 'clifford':
		return stim.Tableau.random(nqbits)
	elif dist.startswith('clifford-brickwork'):
		depth = kwargs.get('depth', 5)
		print(f'Generating random brickwork clifford state with {nqbits} qubits and depth {depth}', flush=True)
		return random_brickwork_clifford(nqbits, depth)
	else:
		print(f'Invalid distribution ({dist}) for random tableau generation')

def random_brickwork_clifford(qubits, depth):
	tab = stim.Tableau(qubits)
	gts = []
	tgts = []
	for i in range(depth):
		for j in range(i%2, qubits, 2):
			rand_gate = stim.Tableau.random(2)
			gts.append(rand_gate)
			tgts.append([j, (j+1)%qubits])
			tab.append(rand_gate, [j, (j+1)%qubits])
	return tab#, gts, tgts

# fidelity calculations for clifford states
def decomposition(P: stim.PauliString, tableau: stim.Tableau):
	"""
	implements the decomposition lemma from Aaronson and Gottesman 2004
	Given a full tableau T = (S, D) and a pauli P E G_n, we can decompose P as P = a d s, where a E {+1, -1, i, -i}, d E D, s E S.
	"""
	# compute the number of qubits n
	n = len(P)

	# compute S and D, the sets of stabilizers and destabilizers of the tableau
	# S = [tableau.z_output(i) for i in range(n)]
	S = tableau.to_stabilizers()
	D = [tableau.x_output(i) for i in range(n)]

	# compute the 2n anticommutators of P with the stabilizers and destabilizers
	d = [not P.commutes(si) for si in S]
	s = [not P.commutes(di) for di in D]

	# compute the actual elements by multiplying the stabs and destabs
	d_pauli = stim.PauliString(n) # identity on n qubits
	s_pauli = stim.PauliString(n) # identity on n qubits
	for i in range(n):
		if d[i]:
			d_pauli *= D[i]
		if s[i]:
			s_pauli *= S[i]
	ds_pauli = d_pauli * s_pauli
	a = P.sign / ds_pauli.sign

	return a, d_pauli, s_pauli, d, s

def L1_distance(tableau1: stim.Tableau, tableau2: stim.Tableau):
	stabs1 = tableau1.to_stabilizers(canonicalize=True)
	stabs2 = tableau2.to_stabilizers(canonicalize=True)
	return 0

def tableau2array(tableau: stim.Tableau):
	stabs = map(str, tableau.to_stabilizers(canonicalize=True))
	mapping = {'_':'00', 'Z':'01', 'X':'10', 'Y':'11', '+':'0', '-':'1'}
	stabs = [''.join(mapping[s] for s in stab) for stab in stabs]
	numer, denom = 0, 0 # computing the jacard distance to target
	for i, stab in enumerate(stabs):
		target_string = '0' + '00'*i + '01' + '00'*(len(stabs)-i-1)
		# print('stab', stab)
		# print('tstr', target_string)
		for a, b in zip(stab, target_string):
			denom += a == '1' or b == '1'
			numer += a != b
	l1_distance_to_target = numer / denom
	# print('-' * 20)
	final_bitstring = ''.join(stabs)
	return np.array(list(map(int, final_bitstring)), dtype=np.uint8), l1_distance_to_target

def fidelity(tableau1: stim.Tableau, tableau2: stim.Tableau, logscale=False, debug=False):
	"""
	compute the fidelity tr (rho2 rho1) between the two tableaus tableau1 and tableau2.
	we compute the stabilizer union U and return the fidelity 2 ** (n - |U|) or 0.
	"""
	n = len(tableau1)

	# new tableau U. since we will update this, we keep track of stabilizers and destabilizers ourselves
	U_stab = tableau1.to_stabilizers()
	U_destab = [tableau1.x_output(i) for i in range(n)]
	U_marked_destabilizers_idxs = []

	for s_prime in tableau2.to_stabilizers():
		# go over the stabilizers of tableau2, seeing if we need to modify U
		# if debug: print('====== iteration ' + str(i) + '======')
		# if debug: print('stabilizers', U_stab)
		# if debug: print('destabilizers', U_destab)
		# if debug: print('new state', s_prime)
		U_tableau = stim.Tableau.from_conjugated_generators(xs=U_destab, zs=U_stab)
		a, _, _, d_list, s_list = decomposition(s_prime, U_tableau)
		# if debug: print(d_list, s_list, U_marked_destabilizers_idxs)
		
		unmarked_destabilizer_idx = n
		for j in range(n):
			if d_list[j] and j not in U_marked_destabilizers_idxs:
				unmarked_destabilizer_idx = j
				break
		
		if unmarked_destabilizer_idx == n:
			# if debug:
			# 	print(f'{a=}')
			# 	print(U_marked_destabilizers_idxs)
			# 	print(d_list)
			assert a == 1 or a == -1
			if a == -1:
				return 0.0
			continue

		# if s' has d terms but all of them marked, even here no need to modify U
			# assert a == 1 or a == -1
			# if a == -1:
			# 	return 0.0, [], [], []
			# continue
		
		# otherwise, we need to modify U
		U_destab[unmarked_destabilizer_idx] = s_prime
		U_marked_destabilizers_idxs.append(unmarked_destabilizer_idx)
		
		# fixing up the commutation algebra
		# assert U_tableau.z_output(unmarked_destabilizer_idx) == U_stab[unmarked_destabilizer_idx]
		# fix = U_tableau.z_output(unmarked_destabilizer_idx).copy()
		fix = U_stab[unmarked_destabilizer_idx]
		for j in range(n):
			if j == unmarked_destabilizer_idx: continue
			# for stabilizers
			if d_list[j]:
				U_stab[j] *= fix
			# for destabilizers
			if s_list[j]:
				U_destab[j] *= fix
	assert len(U_stab) == n
	if logscale:
		return -len(U_marked_destabilizers_idxs)
	return 2 ** -len(U_marked_destabilizers_idxs)

# computing  the entanglement entropy of a state
def binaryMatrix(zStabilizers):
    """
        - Purpose: Construct the binary matrix representing the stabilizer states.
        - Inputs:
            - zStabilizers (array): The result of conjugating the Z generators on the initial state.
        Outputs:
            - binaryMatrix (array of size (N, 2N)): An array that describes the location of the stabilizers in the tableau representation.
    """
    N = len(zStabilizers)
    binaryMatrix = np.zeros((N,2*N))
    r = 0 # Row number
    for row in zStabilizers:
        c = 0 # Column number
        for i in row:
            if i == 3: # Pauli Z
                binaryMatrix[r,N + c] = 1
            if i == 2: # Pauli Y
                binaryMatrix[r,N + c] = 1
                binaryMatrix[r,c] = 1
            if i == 1: # Pauli X
                binaryMatrix[r,c] = 1
            c += 1
        r += 1

    return binaryMatrix

def convert_cutmat_to_rowlist(cutmat):
   
    N,_ = np.shape(cutmat)
    rows = []
    for i in range(N):
        binary_string = ''.join(map(str, cutmat[i,:]))
        res = int(binary_string, 2)                      
        rows.append(int(res))
    return rows  

def getCutStabilizers(binaryMatrix, keeparr):
    """
        - Purpose: Return only the part of the binary matrix that corresponds to the qubits we want to consider for a bipartition.
        - Inputs:
            - binaryMatrix (array of size (N, 2N)): The binary matrix for the stabilizer generators.
            - keeparr : qubit indices to be kept IN (0,1,...,N-1)
        - Outputs:
            - cutMatrix (array of size (N, 2N)): The binary matrix for the cut  
    """
    N = len(binaryMatrix)
    cutMatrix = np.zeros((N,2*N))
    for j in keeparr:
        cutMatrix[:,j] = binaryMatrix[:,j]
        cutMatrix[:,j+N] = binaryMatrix[:,j+N]
    return cutMatrix

def gf2_rank(rows):
    """
    Find rank of a matrix over GF2.

    The rows of the matrix are given as nonnegative integers, thought
    of as bit-strings.

    This function modifies the input list. Use gf2_rank(rows.copy())
    instead of gf2_rank(rows) to avoid modifying rows.
    """
    rank = 0
    while rows:
        pivot_row = rows.pop()
        if pivot_row:
            rank += 1
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
    return rank

def ent_entr(tabl: stim.Tableau):
	stabs = np.array([tabl.z_output(i) for i in range(len(tabl))])
	binmat = binaryMatrix(stabs)
	keep_qubits = range(int(len(tabl)/2))
	cutmat = getCutStabilizers(binmat, keep_qubits)
	rows = convert_cutmat_to_rowlist(np.array(cutmat,	dtype=int))
	rank = gf2_rank(rows.copy())
	sA = (rank - int(len(keep_qubits)))
	return sA

## ALL BELOW IS UNUSED ##

# adapted from the mitdeeplearning library
class PeriodicPlotter:
	def __init__(self, sec, filename, xlabel='', ylabel='', scale=None):
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.sec = sec
		self.scale = scale
		self.filename = filename

		self.tic = time.time()

	def plot(self, data):
		if time.time() - self.tic <= self.sec:
			return
		
		plt.cla()
		
		if self.scale is None:
			plt.plot(data)
		elif self.scale == 'semilogx':
			plt.semilogx(data)
		elif self.scale == 'semilogy':
			plt.semilogy(data)
		elif self.scale == 'loglog':
			plt.loglog(data)
		else:
			raise ValueError("unrecognized parameter scale {}".format(self.scale))

		plt.xlabel(self.xlabel); plt.ylabel(self.ylabel)
		plt.savefig(self.filename)
		self.tic = time.time()

# action-selection strategies
class GreedyStrategy:
	def __call__(self, model, state):
		with torch.no_grad():
			return torch.argmax(model(state)).item()

class EpsGreedyStrategy:
	def __init__(self, epsilon):
		self.epsilon = self.init_epsilon = epsilon
		self.exploratory_action_taken = None

	def update_eps(self):
		return self.epsilon
	
	def update(self):
		return self.update_eps()

	def __call__(self, model, state, update=False, sample=False):
		
		self.exploratory_action_taken = False
		with torch.no_grad():
			q_values = model(state)

		self.exploratory_action_taken = random.randrange(1000)/1000 < self.epsilon + 1e-12
		
		if sample:
			sample_idxs = np.random.choice(len(q_values), (len(q_values)//2,), replace=False)
		else:
			sample_idxs = torch.arange(len(q_values))
		
		ans = np.random.choice(sample_idxs) if self.exploratory_action_taken else torch.argmax(q_values[sample_idxs]).item()
		
		if (update): 
			self.update_eps()
		
		return ans

class ExpDecEpsGreedyStrategy(EpsGreedyStrategy):
	def __init__(self, init_epsilon=1.0, min_epsilon=0.01, decay_steps=1000):
		super().__init__(init_epsilon)
		self.min_epsilon = min_epsilon
		self.decay_steps = decay_steps
		self.decay_factor = (self.init_epsilon/self.min_epsilon)**(1/decay_steps)
		self.t = 0

	def update_eps(self):
		if self.t >= self.decay_steps:
			self.epsilon = self.min_epsilon # stay static at self.min_epsilon
		else:
			self.epsilon /= self.decay_factor
			self.t += 1
		return self.epsilon

class LinDecEpsGreedyStrategy(EpsGreedyStrategy):
	def __init__(self, init_epsilon=1.0, min_epsilon=0.01, decay_steps=1000):
		super().__init__(init_epsilon)
		self.min_epsilon = min_epsilon
		self.decay_steps = decay_steps
		self.decay_factor = (self.init_epsilon - self.min_epsilon)/self.decay_steps # ** to * that's all
		self.t = 0

	def update_eps(self):
		if self.t >= self.decay_steps:
			self.epsilon = self.min_epsilon # stay static at self.min_epsilon
		else:
			self.epsilon -= self.decay_factor # / to - that's all
			self.t += 1
		return self.epsilon


def test(strategy):
	if strategy == 'eps-greedy':
			s = EpsGreedyStrategy(0.3)
			plt.title('Epsilon-Greedy epsilon value')
	elif strategy == 'exp-eps-greedy':
			s = ExpDecEpsGreedyStrategy(1, 0.01, 20)
			plt.title('Exp-Decay-Epsilon-Greedy epsilon value')
	elif strategy == 'lin-eps-greedy':
			s = LinDecEpsGreedyStrategy(1, 0.01, 20)
			plt.title('Lin-Decay-Epsilon-Greedy epsilon value')

	plt.plot([s.update_eps() for _ in range(50)])
	plt.xticks(rotation=45)
	plt.show()

# testing script
STRATEGY_TEST=False
RESULTS_DIR = 'utils_test'
if __name__ == '__main__' and STRATEGY_TEST:
	dir = os.path.join(RESULTS_DIR, 'exploit-explore')
	test('eps-greedy')
	test('exp-eps-greedy')
	test('lin-eps-greedy')






























# models/techniques
from functools import reduce

class SimpleValueRLNet(nn.Module):
	def __init__(self, nin: int, nhid: tuple[int], nout: int, activ) -> None:
		super().__init__()
		combined = [nin] + list(nhid) + [nout]
		self.layers = nn.ModuleList(nn.Linear(nin_i, nout_i) for _, (nin_i, nout_i) in enumerate(zip(combined, combined[1:])))
		for layer in self.layers: nn.init.xavier_normal_(layer.weight); nn.init.normal_(layer.bias, std=0.01)
		nn.init.zeros_(self.layers[-1].bias)
		self.activ_fn = activ
	
	def __call__(self, X) -> torch.Tensor:
		if not isinstance(X, torch.Tensor):
			X = torch.tensor(X, dtype=torch.float32, requires_grad=False)
		X = reduce(lambda X, layer: self.activ_fn(layer(X)), self.layers[:-1], X)
		return self.layers[-1](X)

# stats generators
# -- for now written in the main code itself, for each agent

# data smoothing
def smoothen(data: list, window=20):
	if len(data) < window: return data
	smoothed = []
	for i in range(window, len(data)-window+1):
		smoothed.append(sum(data[i-window:i+window+1])/(2*window+1))
	return smoothed

# gaussian smoothing
def gaussian_smoothen(data: list, window=20):
	if len(data) < window: return data
	smoothed = []
	for i in range(len(data)-window+1):
		smoothed.append(sum(data[j]*np.exp(-(j-i)**2) for j in range(i-window, i+window+1))/(2*window+1)) # doing it in n2 instead of nlogn, oof
	return smoothed


# ================================
# QUANTUM OPERATION TIMING CONSTANTS FOR TIMELINE VISUALIZATION
# ================================
# Time constants (in microseconds) - centralized control
SINGLE_QUBIT_TIME = 20.0    # Single-qubit gate duration (H, S)
TWO_QUBIT_TIME = 10.0       # Two-qubit gate duration (CX)  
MEASUREMENT_TIME = 100.0    # Measurement duration

# Zone movement cost matrix (in microseconds)
# Zones: 0=readout, 1=single qubit gate, 2=entanglement
MOVEMENT_COSTS = np.array([
    [0.0,   400.0, 400.0],  # From readout to [readout, single qubit, entanglement]
    [400.0, 0.0,   100.0],  # From single qubit to [readout, single qubit, entanglement]  
    [400.0, 100.0, 50.0]     # From entanglement to [readout, single qubit, entanglement]
])
# ================================


def get_inverse_gate_name(gate_name):
    """Get the inverse gate name for visualization purposes without target qubits, in lowercase letters."""
    # Extract gate type and return inverse without qubit info
    if gate_name.startswith('h('):
        return 'h'  # H is self-inverse
    elif gate_name.startswith('hsdgh('):
        # hsdgh inverse is hsh
        return 'hsh'
    elif gate_name.startswith('hsh('):
        # hsh inverse is hsdgh  
        return 'hsdgh'
    elif gate_name.startswith('sdg('):
        # sdg inverse is s
        return 's'
    elif gate_name.startswith('s('):
        # s inverse is sdg
        return 'sdg'
    elif gate_name.startswith('x('):
        return 'x'  # X is self-inverse
    elif gate_name.startswith('y('):
        return 'y'  # Y is self-inverse
    elif gate_name.startswith('z('):
        return 'z'  # Z is self-inverse
    elif gate_name.startswith('cnot('):
        return 'cx'  # CNOT is self-inverse, call it cx
    elif gate_name.startswith('cx('):
        return 'cx'  # CX is self-inverse
    elif gate_name.startswith('t('):
        return 'tdg'  # T inverse is TDG
    elif gate_name.startswith('tdg('):
        return 't'  # TDG inverse is T
    else:
        # For any other gates, just lowercase and remove parentheses
        return gate_name.split('(')[0].lower()

def format_gate_name(gate_name):
    """Convert gate names to lowercase letters without target qubits, CNOT -> cx"""
    if gate_name.startswith('h('):
        return 'h'
    elif gate_name.startswith('hsdgh('):
        return 'hsdgh'
    elif gate_name.startswith('hsh('):
        return 'hsh'
    elif gate_name.startswith('sdg('):
        return 'sdg'
    elif gate_name.startswith('s('):
        return 's'
    elif gate_name.startswith('x('):
        return 'x'
    elif gate_name.startswith('y('):
        return 'y'
    elif gate_name.startswith('z('):
        return 'z'
    elif gate_name.startswith('cnot('):
        return 'cx'
    elif gate_name.startswith('cx('):
        return 'cx'
    elif gate_name.startswith('t('):
        return 't'
    elif gate_name.startswith('tdg('):
        return 'tdg'
    else:
        # For any other gates, just lowercase and remove parentheses
        return gate_name.split('(')[0].lower()

# def plot_timeline(action_sequence, env, save_path=None, reverse=False):
#     """ Plot execution timeline showing when each qubit is busy with operations.
    
#     Creates a Gantt chart visualization adapted from the JAX-based environment logic
#     to work with the circuit environment structure.
    
#     Args:
#         action_sequence: List of action indices taken in the circuit
#         env: The circuit Environment object containing gate and target information
#         save_path: Optional path to save the plot (if None, returns base64 string)
#         reverse: If True, reverse the gate sequence and show inverse gates
        
#     Returns:
#         Base64 encoded image string for display or save path if provided
#     """
#     # Handle reverse mode
#     if reverse:
#         action_sequence = list(reversed(action_sequence))
    
#     n_total = env.qubits
#     final_time = len(action_sequence)
    
#     # Initialize timeline tracking
#     qubit_free_time = np.zeros(n_total)
#     current_zones = np.ones(n_total, dtype=np.uint8)  # All qubits start in single qubit gate zone (zone 1)
    
#     # Store operations for visualization: list of (qubit_idx, start_time, end_time, operation_name, operation_type)
#     operations = []
    
#     def simulate_single_action(step):
#         nonlocal qubit_free_time, current_zones, operations
        
#         # Get action info
#         action = action_sequence[step]
#         gate_name = env.gates[action]
#         # Use inverse gate name if in reverse mode
#         if reverse:
#             gate_name = get_inverse_gate_name(gate_name)
#         targets = env.targets[action]  # List of qubit indices for this gate
        
#         # Determine operation parameters based on gate type
#         is_single_qubit = len(targets) == 1
        
#         if is_single_qubit:
#             target_zones = np.array([1, -1])  # Single qubit gate zone, second qubit unused
#             operation_time = SINGLE_QUBIT_TIME
#             op_type = 'single_qubit'
#         else:
#             target_zones = np.array([2, 2])   # Both qubits to entanglement zone
#             operation_time = TWO_QUBIT_TIME
#             op_type = 'two_qubit'
        
#         # Calculate movement times for involved qubits
#         qubit1_idx = targets[0]
#         current_zone1 = current_zones[qubit1_idx]
#         target_zone1 = target_zones[0]
#         movement_time1 = MOVEMENT_COSTS[current_zone1, target_zone1]
        
#         if not is_single_qubit:
#             # Two-qubit operation
#             qubit2_idx = targets[1]
#             current_zone2 = current_zones[qubit2_idx]
#             target_zone2 = target_zones[1]
#             movement_time2 = MOVEMENT_COSTS[current_zone2, target_zone2]
            
#             # Calculate start time (when both qubits are free + movement time)
#             earliest_start_time = max(qubit_free_time[qubit1_idx] + movement_time1, 
#                                     qubit_free_time[qubit2_idx] + movement_time2)
#             operation_end_time = earliest_start_time + operation_time
            
#             # Add movement operations if needed
#             if movement_time1 > 0:
#                 move_type1 = 'inter_zone_movement' if current_zone1 != target_zone1 else 'intra_zone_movement'
#                 operations.append((qubit1_idx, qubit_free_time[qubit1_idx], 
#                                  qubit_free_time[qubit1_idx] + movement_time1, 
#                                  f'Move {current_zone1}→{target_zone1}', move_type1))
#             if movement_time2 > 0:
#                 move_type2 = 'inter_zone_movement' if current_zone2 != target_zone2 else 'intra_zone_movement'
#                 operations.append((qubit2_idx, qubit_free_time[qubit2_idx], 
#                                  qubit_free_time[qubit2_idx] + movement_time2, 
#                                  f'Move {current_zone2}→{target_zone2}', move_type2))
            
#             # Add gate operation for both qubits
#             operations.append((qubit1_idx, earliest_start_time, operation_end_time, gate_name, op_type))
#             operations.append((qubit2_idx, earliest_start_time, operation_end_time, gate_name, op_type))
            
#             # Update timeline
#             qubit_free_time[qubit1_idx] = operation_end_time
#             qubit_free_time[qubit2_idx] = operation_end_time
#             current_zones[qubit1_idx] = target_zone1
#             current_zones[qubit2_idx] = target_zone2
            
#         else:
#             # Single qubit operation
#             earliest_start_time = qubit_free_time[qubit1_idx] + movement_time1
#             operation_end_time = earliest_start_time + operation_time
            
#             # Add movement operation if needed
#             if movement_time1 > 0:
#                 move_type1 = 'inter_zone_movement' if current_zone1 != target_zone1 else 'intra_zone_movement'
#                 operations.append((qubit1_idx, qubit_free_time[qubit1_idx], 
#                                  qubit_free_time[qubit1_idx] + movement_time1, 
#                                  f'Move {current_zone1}→{target_zone1}', move_type1))
            
#             # Add gate operation
#             operations.append((qubit1_idx, earliest_start_time, operation_end_time, gate_name, op_type))
            
#             # Update timeline
#             qubit_free_time[qubit1_idx] = operation_end_time
#             current_zones[qubit1_idx] = target_zone1
    
#     # Simulate all actions
#     for step in range(final_time):
#         simulate_single_action(step)
    
#     # Calculate total execution time (when the last qubit finishes)
#     total_execution_time = max([end_time for _, _, end_time, _, _ in operations]) if operations else 0
    
#     # Add idle time blocks for all unoccupied periods
#     for qubit_idx in range(n_total):
#         # Get all operations for this qubit and sort by start time
#         qubit_operations = [(start_time, end_time) for q_idx, start_time, end_time, _, _ in operations if q_idx == qubit_idx]
#         qubit_operations.sort()
        
#         current_time = 0.0
#         for start_time, end_time in qubit_operations:
#             # Add idle time before this operation if there's a gap
#             if current_time < start_time:
#                 operations.append((qubit_idx, current_time, start_time, 'idle', 'idle'))
#             current_time = end_time
        
#         # Add idle time from last operation to total execution time
#         if current_time < total_execution_time:
#             operations.append((qubit_idx, current_time, total_execution_time, 'idle', 'idle'))
    
#     # Create the plot with wider aspect ratio for better gate visibility
#     plt.close('all')
#     fig, ax = plt.subplots(figsize=(20, max(4, n_total * 0.6)))
    
#     # Color mapping for different operation types
#     colors = {
#         'single_qubit': '#3498db',        # Blue
#         'two_qubit': '#e74c3c',           # Red
#         'inter_zone_movement': '#f1c40f', # Yellow (between different zones)
#         'intra_zone_movement': '#2ecc71', # Green (within same zone)
#         'idle': '#808080'                 # Gray (idle time)
#     }
    
#     # Plot operations as rectangles
#     for qubit_idx, start_time, end_time, op_name, op_type in operations:
#         duration = end_time - start_time
#         rect = patches.Rectangle(
#             (start_time, qubit_idx - 0.4),  # (x, y) bottom-left corner
#             duration,                       # width (duration)
#             0.8,                           # height (qubit lane height)
#             linewidth=1,                   # Thin black boundary
#             edgecolor='black',             # Black edge color
#             facecolor=colors[op_type],
#             alpha=0.8                      # Slightly less transparent
#         )
#         ax.add_patch(rect)
        
#         # Add operation label if the rectangle is wide enough
#         if duration > 8:  # Lower threshold to show labels on shorter gate operations
#             font_size = 12 if duration > 50 else 10  # Larger font for better readability
#             ax.text(start_time + duration/2, qubit_idx, op_name, 
#                    ha='center', va='center', fontsize=font_size, fontweight='bold')
    
#     # Calculate total execution time (when the last qubit finishes)
#     total_execution_time = max([end_time for _, _, end_time, _, _ in operations]) if operations else 0
    
#     # Customize the plot
#     ax.set_xlabel('Time (microseconds)', fontsize=14)
#     ax.set_ylabel('Qubit Index', fontsize=14)
#     ax.set_title(f'Quantum Circuit Execution Timeline (Total Time: {total_execution_time:.1f}μs)', 
#                 fontsize=16, fontweight='bold')
    
#     # Set axis limits and ticks
#     max_time = max([end_time for _, _, end_time, _, _ in operations]) if operations else 0
#     ax.set_xlim(0, max_time * 1.05)
#     ax.set_ylim(-0.5, n_total - 0.5)
#     ax.set_yticks(range(n_total))
#     ax.set_yticklabels([f'Qubit {i}' for i in range(n_total)])
    
#     # Invert y-axis so Qubit 0 is on top
#     ax.invert_yaxis()
    
#     # Add legend
#     legend_elements = [
#         patches.Patch(color=colors['single_qubit'], label='Single Qubit Gate'),
#         patches.Patch(color=colors['two_qubit'], label='Two Qubit Gate'), 
#         patches.Patch(color=colors['inter_zone_movement'], label='Inter-Zone Movement'),
#         patches.Patch(color=colors['intra_zone_movement'], label='Intra-Zone Movement'),
#         patches.Patch(color=colors['idle'], label='Idle')
#     ]
#     ax.legend(handles=legend_elements, loc='upper right')
    
#     # Add grid for better readability
#     ax.grid(True, alpha=0.3)
    
#     # Tight layout
#     plt.tight_layout()
    
#     # Save or return base64
#     if save_path:
#         plt.savefig(save_path, dpi=400, bbox_inches='tight')
#         print(f"Timeline plot saved to: {save_path}")
#         plt.close(fig)
#         return save_path
#     else:
#         # Return base64 for display
#         img_buffer = io.BytesIO()
#         fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
#         img_buffer.seek(0)
#         img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
#         img_buffer.close()
#         plt.close(fig)
#         return img_base64


def plot_timeline_qiskit(qiskit_circuit, save_path=None, title_suffix=""):
    """ Plot execution timeline for a Qiskit circuit showing when each qubit is busy with operations.
    
    Creates a Gantt chart visualization for Qiskit circuits, similar to plot_timeline but for
    benchmark circuits that are already in Qiskit format.
    
    Args:
        qiskit_circuit: Qiskit QuantumCircuit object
        save_path: Optional path to save the plot (if None, returns base64 string)
        title_suffix: Additional text to add to the plot title
        
    Returns:
        Base64 encoded image string for display or save path if provided
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import io
    import base64
    import numpy as np
    
    n_total = qiskit_circuit.num_qubits
    
    # Initialize timeline tracking
    qubit_free_time = np.zeros(n_total)
    current_zones = np.ones(n_total, dtype=np.uint8)  # All qubits start in single qubit gate zone (zone 1)
    
    # Store operations for visualization: list of (qubit_idx, start_time, end_time, operation_name, operation_type)
    operations = []
    
    # Process each instruction in the Qiskit circuit
    for instruction_idx, (instruction, qubits, clbits) in enumerate(qiskit_circuit.data):
        gate_name = instruction.name
        qubit_indices = [qiskit_circuit.find_bit(q).index for q in qubits]
        
        # Skip SWAP gates completely - ignore them in timeline calculation
        if gate_name.lower() == 'swap':
            continue
        
        # Determine operation parameters based on gate type
        is_single_qubit = len(qubit_indices) == 1
        
        if is_single_qubit:
            operation_time = SINGLE_QUBIT_TIME
            op_type = 'single_qubit'
            
            qubit_idx = qubit_indices[0]
            current_zone = current_zones[qubit_idx]
            target_zone = 1  # Single qubit gates in single qubit gate zone
            movement_time = MOVEMENT_COSTS[current_zone, target_zone]
            
            # Calculate timing
            earliest_start_time = qubit_free_time[qubit_idx] + movement_time
            operation_end_time = earliest_start_time + operation_time
            
            # Add movement operation if needed
            if movement_time > 0:
                move_type = 'inter_zone_movement' if current_zone != target_zone else 'intra_zone_movement'
                operations.append((qubit_idx, qubit_free_time[qubit_idx], 
                                 qubit_free_time[qubit_idx] + movement_time, 
                                 f'Move {current_zone}→{target_zone}', move_type))
            
            # Add gate operation
            operations.append((qubit_idx, earliest_start_time, operation_end_time, gate_name, op_type))
            
            # Update timeline
            qubit_free_time[qubit_idx] = operation_end_time
            current_zones[qubit_idx] = target_zone
            
        else:
            # Two-qubit operation (CNOT, CZ, etc.) - both qubits to entanglement zone
            qubit1_idx = qubit_indices[0]
            qubit2_idx = qubit_indices[1]
            
            operation_time = TWO_QUBIT_TIME
            op_type = 'two_qubit'
            
            current_zone1 = current_zones[qubit1_idx]
            current_zone2 = current_zones[qubit2_idx]
            target_zone1 = 2  # entanglement zone
            target_zone2 = 2  # entanglement zone
            
            movement_time1 = MOVEMENT_COSTS[current_zone1, target_zone1]
            movement_time2 = MOVEMENT_COSTS[current_zone2, target_zone2]
            
            # Calculate start time (when both qubits are free and moved to entangling zone)
            earliest_start_time = max(qubit_free_time[qubit1_idx] + movement_time1, 
                                    qubit_free_time[qubit2_idx] + movement_time2)
            operation_end_time = earliest_start_time + operation_time
            
            # Add movement operations if needed
            if movement_time1 > 0:
                move_type1 = 'inter_zone_movement' if current_zone1 != target_zone1 else 'intra_zone_movement'
                operations.append((qubit1_idx, qubit_free_time[qubit1_idx], 
                                 qubit_free_time[qubit1_idx] + movement_time1, 
                                 f'Move {current_zone1}→{target_zone1}', move_type1))
            if movement_time2 > 0:
                move_type2 = 'inter_zone_movement' if current_zone2 != target_zone2 else 'intra_zone_movement'
                operations.append((qubit2_idx, qubit_free_time[qubit2_idx], 
                                 qubit_free_time[qubit2_idx] + movement_time2, 
                                 f'Move {current_zone2}→{target_zone2}', move_type2))
            
            # Add gate operation for both qubits
            operations.append((qubit1_idx, earliest_start_time, operation_end_time, gate_name, op_type))
            operations.append((qubit2_idx, earliest_start_time, operation_end_time, gate_name, op_type))
            
            # Update timeline and zones
            qubit_free_time[qubit1_idx] = operation_end_time
            qubit_free_time[qubit2_idx] = operation_end_time
            current_zones[qubit1_idx] = target_zone1
            current_zones[qubit2_idx] = target_zone2
    
    # Calculate total execution time (when the last qubit finishes)
    total_execution_time = max([end_time for _, _, end_time, _, _ in operations]) if operations else 0
    
    # Add idle time blocks for all unoccupied periods
    for qubit_idx in range(n_total):
        # Get all operations for this qubit and sort by start time
        qubit_operations = [(start_time, end_time) for q_idx, start_time, end_time, _, _ in operations if q_idx == qubit_idx]
        qubit_operations.sort()
        
        current_time = 0.0
        for start_time, end_time in qubit_operations:
            # Add idle time before this operation if there's a gap
            if current_time < start_time:
                operations.append((qubit_idx, current_time, start_time, 'idle', 'idle'))
            current_time = end_time
        
        # Add idle time from last operation to total execution time
        if current_time < total_execution_time:
            operations.append((qubit_idx, current_time, total_execution_time, 'idle', 'idle'))
    
    # Create the plot with wider aspect ratio for better gate visibility
    plt.close('all')
    fig, ax = plt.subplots(figsize=(20, max(4, n_total * 0.6)))
    
    # Color mapping for different operation types
    colors = {
        'single_qubit': '#3498db',        # Blue
        'two_qubit': '#e74c3c',           # Red
        'inter_zone_movement': '#f1c40f', # Yellow (between different zones)
        'intra_zone_movement': '#2ecc71', # Green (within same zone)
        'idle': '#808080'                 # Gray (idle time)
    }
    
    # Plot operations as rectangles
    for qubit_idx, start_time, end_time, op_name, op_type in operations:
        duration = end_time - start_time
        rect = patches.Rectangle(
            (start_time, qubit_idx - 0.4),  # (x, y) bottom-left corner
            duration,                       # width (duration)
            0.8,                           # height (qubit lane height)
            linewidth=1,                   # Thin black boundary
            edgecolor='black',             # Black edge color
            facecolor=colors[op_type],
            alpha=0.8                      # Slightly less transparent
        )
        ax.add_patch(rect)
        
        # Add operation label if the rectangle is wide enough
        if duration > 8:  # Lower threshold to show labels on shorter gate operations
            font_size = 12 if duration > 50 else 10  # Larger font for better readability
            ax.text(start_time + duration/2, qubit_idx, op_name, 
                   ha='center', va='center', fontsize=font_size, fontweight='bold')
    
    # Calculate total execution time (when the last qubit finishes)
    total_execution_time = max([end_time for _, _, end_time, _, _ in operations]) if operations else 0
    
    # Customize the plot
    ax.set_xlabel('Time (microseconds)', fontsize=14)
    ax.set_ylabel('Qubit Index', fontsize=14)
    title = f'Qiskit Circuit Execution Timeline{title_suffix} (Total Time: {total_execution_time:.1f}μs)'
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Set axis limits and ticks
    max_time = max([end_time for _, _, end_time, _, _ in operations]) if operations else 0
    ax.set_xlim(0, max_time * 1.05)
    ax.set_ylim(-0.5, n_total - 0.5)
    ax.set_yticks(range(n_total))
    ax.set_yticklabels([f'Qubit {i}' for i in range(n_total)])
    
    # Invert y-axis so Qubit 0 is on top
    ax.invert_yaxis()
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['single_qubit'], label='Single Qubit Gate'),
        patches.Patch(color=colors['two_qubit'], label='Two Qubit Gate'), 
        patches.Patch(color=colors['inter_zone_movement'], label='Inter-Zone Movement'),
        patches.Patch(color=colors['intra_zone_movement'], label='Intra-Zone Movement'),
        patches.Patch(color=colors['idle'], label='Idle')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or return base64
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        print(f"Timeline plot saved to: {save_path}")
        plt.close(fig)
        return save_path
    else:
        # Return base64 for display
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        img_buffer.close()
        plt.close(fig)
        return img_base64