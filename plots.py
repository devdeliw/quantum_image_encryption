from generate_sphere import Sphere
from wave_functions import Scramble_Wavefunctions

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

class Generate_Plots: 

	def __init__(self, image_path, n, depth, verbose): 

		self.image_path = image_path
		self.n = n 
		self.depth = depth 
		self.verbose = verbose

	def bloch_sphere(self): 

		statevectors = Scramble_Wavefunctions(self.image_path,
											  self.n, 
											  self.depth, 
											  self.verbose
	    ).statevectors()

	    for wave_function in statevectors:
	    	plot_bloch_multivector(sv)

	   	plt.show()							