from generate_sphere import * 

from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, DensityMatrix, random_statevector, partial_trace
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile

from scipy.linalg import sqrtm

import numpy as np
import secrets, time, uuid, hashlib

class Scramble_WaveFunctions: 

	def __init__(self, image_path, n, depth, verbose): 

		self.image_path = image_path
		self.depth = depth
		self.n = n
		self.verbose = verbose

	def define_wave_functions(self): 

		wave_functions = []
		spherical_coordinates = Sphere(self.image_path).angles()

		# generate wave functions using theta, phi from spherical 
		# pixel-statevectors
		for theta, phi in spherical_coordinates: 

			alpha = np.cos(theta / 2)
			beta = np.exp(1j * phi) * np.sin(theta / 2)

			wave_functions.append(Statevector([alpha, beta]))

		return wave_functions 

	def group_wave_functions(self): 

		wave_functions = self.define_wave_functions()
		num_wave_functions = len(wave_functions)

		quotient = num_wave_functions // self.n 

		# groups ordered sets of self.n qubits together 
		groups = [] 

		i = 0 
		while i < num_wave_functions: 
			groups.append([wave_functions[i:i+self.n]])
			i += self.n

		return groups 

	def generate_circuits(self): 

		groups = self.group_wave_functions()
		self.wave_functions = groups

		# generic seed generation coupling many different encryption mechanisms
		def generate_seed():

            entropy = secrets.token_bytes(16) + time.time_ns().to_bytes(8, 'big') + uuid.uuid4().bytes
            seed = int(hashlib.sha256(entropy).hexdigest(), 16) % 2**32

            return seed

       	seeds = []
		circuits = []

		for i in range(len(groups)): 

			num_wave_functions = len(groups[i][0])

			# generate a randomized circuit of a certain depth using 
			# our generated seed
			seed = generate_seed() 
			circuit = random_circuit(num_wave_functions, depth = self.depth, seed = seed)

			circuits.append(circuit)
			seeds.append(seed)

		return circuits, seeds

	def run_circuits(self): 

		circuits, seeds = self.generate_circuits()
		all_wave_functions = self.wave_functions

		# store circuits and seeds universally
		self.circuits = circuits
		self.seeds = seeds

		simulator = Aer.get_backend('statevector_simulator')
		final_wave_functions = []

		# cyclic permutation of a list
		def cycle(arr): 
			return [arr[-1]] + arr[:-1]

		def run(current_wave_funcs, i):

			result_wf = []

			num_wave_functions = len(current_wave_funcs)

			initial_wave_function = current_wave_funcs[0]

			for wf in current_wave_funcs[1:]:
				initial_wave_function = initial_wave_function.tensor(wf)

			main_circuit = circuits[i]

			# create a circuit to first prepare the initial_wave_function 
			# from a circuit starting from |0>|0>|0>...
			initialization_circuit = QuantumCircuit(num_wave_functions)
			initialization_circuit.initialize(initial_wave_function.data, 
											  [j for j in range(num_wave_functions)])

			# append the randomized circuit to the initialization_circuit
			full_circuit = initialization_circuit.compose(main_circuit)

			# transpile the circuit onto the simulator backend
			transpiled_circuit = transpile(full_circuit, simulator)
			result = simulator.run(transpiled_circuit).result()

			# output the resultant statevector
			final_wave_function = result.get_statevector()
			result_wf.append(final_wave_functions)

			return result_wf

		for i in range(len(groups)): 
			wave_functions = groups[i][0]
			ancillary_wave_functions = []

			# perform the circuit running algorithm on every permutation 
			# of self.n qubits stored in each index of groups
			j = 0 
			while j < len(wave_functions): 
				trial_wave_function = cycle(wave_functions)
				output_wave_functions = run(trial_wave_function, i)

				ancillary_wave_functions.append(output_wave_functions)
				j += 1

			final_wave_functions.append(ancillary_wave_functions)

		return final_wave_functions

	def density_matrix(self): 

		final_wave_functions = self.run_circuits()
		density_matrices = []

		# converts the resultant pure states into their density matrices
		for i in final_wave_functions: 
			coupled_density_matrices = [] 

			for j in range(len(i)): 
				coupled_density_matrices.append(DensityMatrix(i[j][0]))

			density_matrices.append(coupled_density_matrices)

		return density_matrices

	def fidelities(self): 

		density_matrices = self.density_matrix()

		# generating a new seed based on the original seeds generated 
		# for the randomized circuits
		combined_string = ''.join(map(str, self.seeds))
		hash_object = hashlib.sha256(combined_string.encode())
		hex_dig = hash_object.hexdigest()[:5]
		seed = int(hex_dig, 16)

		# generate a temporary randomized statevector whhich we will 
		# use to calculate fidelities for each of the pure states
		randomized_pure_state = random_statevector(dims = 2**self.n, seed = seed)
		randomized_density_matrix = DensityMatrix(randomized_pure_state)

		fidelities = []

		def calculate_fidelitiy(rho, sigma): 

			sqrt_rho = sqrtm(rho)
			product = np.dot(sqrt_rho, np.dot(sigma, sqrt_rho))
		    sqrt_product = sqrtm(product)
		    trace = np.trace(sqrt_product)
		    
		    return np.real(trace)**2

		# calculate all the fidelities and store them in a list
		for coupled_states in density_matrices: 
		 	coupled_fidelities = []

		 	for matrix in coupled_states: 

		 		fidelity = calculate_fidelitiy(matrix, randomized_density_matrix)
		 		coupled_fidelities.append(fidelity)

		 	fidelities.append(coupled_fidelities)

		 return fidelities, density_matrices

	def pure_to_mixed(self): 

		fidelities, density_matrices = self.fidelities()

		weights = [] 

		# using the calculated fidelities, generate self.n density matrix
		# coefficients that together sum to one
		for values in fidelities: 
			coupled_weights = []
			total = 0 

			for fidelity in values: 
				total += fidelity 

			for fidelity in values: 
				coupled_weights.append(fidelity / total)

			weights.append(coupled_weights)

		mixed_density_matrices = []

		# generate mixed_states using the weights for each of the pure states
		# the mixed states are just generated via ∑ weight_n * |Ψ_n><Ψ_n| 
		for coupled_states_idx in range(len(density_matrices)):
			mixed_density_matrix = np.zeros((2**self.n, 2**self.n), dtype=np.complex128)

			for matrix_idx in range(len(density_matrices[coupled_states_idx])):
				current_matrix = density_matrices[coupled_states_idx][matrix_idx].data

				mixed_density_matrix += weights[coupled_states_idx][matrix_idx] * current_matrix


			mixed_density_matrices.append(DensityMatrix(mixed_density_matrix))

		return mixed_density_matrices

	def spectral_decomposition(self): 

		density_matrices = self.pure_to_mixed()

		full_eigenvalues, full_eigenvectors = [], []

		# perform spectral/eigendecomposition on the resultant mixed states
		# to retrieve the eigenvalues and eigenvectors 
		for matrix in density_matrices: 

			is_unit_trace = np.isclose(np.trace(matrix), 1.0)
			is_hermitian = np.allclose(matrix, np.matrix(matrix.data).getH())

			# implement checks to ensure the provided density matrix is valid
			if not is_unit_trace: 
				raise Exception("Tr(ρ) ≠ 1")
			if not is_hermitian: 
				raise Exception("ρ not Hermitian")

			eigenvalues, eigenvectors = np.linalg.eigh(matrix.data)
			full_eigenvalues.append([eigenvalues])
			full_eigenvectors.append([eigenvectors])

			reconstructed_matrix = sum(eigenvalues[i] * np.outer(eigenvectors[:, i], 
								       np.conj(eigenvectors[:, i])) for i in range(len(eigenvalues)))

			spectral_decomp_worked = np.allclose(matrix, reconstructed_matrix)

			if not spectral_decomp_worked: 
				raise Exception("spectral decomposition failed")

		return full_eigenvalues, full_eigenvectors 

	def entropy(self): 

		eigenvalues, eigenvectors = self.spectral_decomposition() 
		entropies = []

		# calculating the von-neumann entropy of the mixed_states 
		# using their eigenvalues
		def calculate_entropy(eigenvalues):  

			eigenvalues = np.array(eigenvalues)
			non_zero_eigenvalues = eigenvalues[eigenvalues > 0]

			entropy = -np.sum(non_zero_eigenvalues * np.log(non_zero_eigenvalues))

			return entropy

		for eigenlist in eigenvalues: 
			entropies.append(calculate_entropy(eigenlist))

		return entropies, eigenvalues, eigenvectors

	def organize_wave_functions(self): 

		entropies, eigenvalues, eigenvectors = self.entropy() 

		sorted_indices = np.argsort(entropies)[::-1]
		
		entropies = np.array(entropies)
		eigenvalues = np.array(eigenvalues)
		eigenvectors = np.array(eigenvectors)

		eigenvalues = eigenvalues[sorted_indices]
		eigenvectors = eigenvectors[sorted_indices]

		# extracting all the zero eigenvalues that represent impossible states
		# to retrieve upon measurement
		non_zero_eigenvalue_idxs = []
		for eigenlist in eigenvalues: 
			non_zero_eigenvalue_idxs.append(np.where(eigenlist[0] > 1e-10)[0])

		organized_eigenvectors = []
		statevectors = []

		# organize all the statevectors based on decreasing entropy
		# the eigenvectors from the mixed states with maximum entropy 
		# are placed in the lowest index
		for idx in range(len(eigenvectors)):
			allowed_idxs = non_zero_eigenvalue_idxs[idx]

			organized_eigenvectors.append(eigenvectors[idx][0][allowed_idxs])

		for coupled_statevectors in organized_eigenvectors: 
			for statevector in coupled_statevectors: 
				statevectors.append(Statevector(statevector))

		return statevectors

	def statevectors(self): 

		multi_qubit_statevectors = self.organize_wave_functions()

		# convert all the self.n-qubit statevectors into self.n single-qubit statevectors
		# via partial_tracing -- I just extract the eigenvector with the lowest 
		# eigenvalue/probability
		def n_to_single(statevector): 

			single_qubit_density_matrix = partial_trace(statevector, [i for i in range(self.n - 1)])
			eigenvalues, eigenvectors = np.linalg.eigh(single_qubit_density_matrix.data)

			sorted_eigenvectors = np.array(eigenvectors)[np.argsort(eigenvalues)]

			return Statevector(sorted_eigenvectors[0])

		statevectors = []

		for statevec in multi_qubit_statevectors: 
			statevectors.append(n_to_single(statevec))

		return statevectors













		






			


























