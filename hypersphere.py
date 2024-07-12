from PIL import Image
from qiskit.quantum_info import Statevector

import numpy as np 

class Hypersphere: 

	"""
	Generates a 2^n dimensional Hypersphere containing the 
	RGB information of pixels in an image.

	The Hypersphere is afterwards used to reversibly generate 
	n-dimensional quantum wavefunctions/statevectors

	Parameters: 
	-----------

	n 			: int 
	# of qubits per statevector you wish to generate 

	image_path 	: str
	location of image for hypersphere generation

	verbose		: bool
	to print out information regarding the image and statevectors
	generated 

	Methods: 
	--------

	def get_pixels(self): 
		returns the RGB pixel information and image size 

	def pixel_to_int(self): 
		reversibly converts all the (R,G,B) information per pixel 
		into an single integer

	def hypersphere(self): 
		groups n integers into a set and converts them to unit vectors
		on an n-dimensional sphere

	def statevectors(self): 
		using the hypersphere and the groups of 2^n dimensional unit vectors,
		reversibly generates n-dimensional statevectors 

	"""

	def __init__(self, n, image_path, verbose = False): 

		self.image_path = image_path
		self.n = n 
		self.verbose = verbose 

	def get_pixels(self): 

		im = Image.open(self.image_path, mode = 'r')
		image = im.convert('RGB')
		width, height = image.size

		pixel_values = np.array(list(image.getdata()))

		return pixel_values, width, height

	def pixel_to_int(self): 

		pixel_values, width, height = self.get_pixels()

		def rgb_to_decimal(r, g, b):

		    return (r << 16) + (g << 8) + b

		int_values = []

		for r, g, b in pixel_values: 
			int_values.append(rgb_to_decimal(r, g, b))
		int_values = np.array(int_values)

		return int_values

	def hypersphere(self): 

		int_values = self.pixel_to_int() 
		groups = []

		i = 0
		while i < len(int_values): 
			groups.append(int_values[i:i + 2**self.n])
			i += 2**self.n

		def to_unit_vector(arr):

		    magnitude = np.linalg.norm(arr)

		    if magnitude == 0:
		        return arr, magnitude

		    unit_vector = arr / magnitude
		    return unit_vector, magnitude

		unit_vector_groups = []
		magnitudes = [] 

		for group in groups:
			unit_vector, magnitude = to_unit_vector(group)
			magnitudes.append(magnitude)
			unit_vector_groups.append(unit_vector)

		return unit_vector_groups, magnitudes

	def statevectors(self): 

		unit_vector_groups, magnitudes = self.hypersphere()

		def unit_vector_to_statevector(unit_vector):

		    dim = len(unit_vector) // 2
		    real_parts = unit_vector[:dim]
		    imag_parts = unit_vector[dim:]
		    statevector = real_parts + 1j * imag_parts
		    statevector /= np.linalg.norm(statevector)

		    return statevector

		statevectors = []
		for i in unit_vector_groups: 
			statevectors.append(unit_vector_to_statevector(i))

		statevectors = [Statevector(i) for i in statevectors]

		return statevectors, magnitudes

class Inverse_Hypersphere: 



	def __init__(self, width, height, statevectors, magnitudes, name, save_image_path): 

		self.width = width 
		self.height = height
		self.statevectors = statevectors
		self.magnitudes = magnitudes
		self.name = name
		self.save_image_path = save_image_path

	def recover_hypersphere(self): 

		def statevector_to_unit_vector(statevector):
		    real_parts = np.real(statevector)
		    imag_parts = np.imag(statevector)
		    combined = np.concatenate([real_parts, imag_parts])
		    unit_vector = combined / np.linalg.norm(combined)
		    return unit_vector

		recovered_unit_vectors = []

		for i in self.statevectors: 
			recovered_unit_vectors.append(statevector_to_unit_vector(i.data))

		return recovered_unit_vectors

	def recover_pixels(self): 

		recovered_unit_vectors = self.recover_hypersphere()

		def decimal_to_rgb(decimal):
		    r = (decimal >> 16) & 0xFF
		    g = (decimal >> 8) & 0xFF
		    b = decimal & 0xFF

		    return r, g, b

		int_vectors = []
		for i in range(len(recovered_unit_vectors)): 
			int_vectors.append(recovered_unit_vectors[i] * self.magnitudes[i])

		recovered_int_values = []
		for i in int_vectors: 
			for j in i: 
				recovered_int_values.append(j)
		recovered_int_values = np.array(recovered_int_values)

		recovered_pixel_values = []
		for i in recovered_int_values: 
			recovered_pixel_values.append(decimal_to_rgb(int(i)))
		recovered_pixel_values = np.array(recovered_pixel_values)

		return recovered_pixel_values

	def recover_image(self): 

		recovered_pixel_values = self.recover_pixels()

		reshaped_pixel_values = recovered_pixel_values.reshape((self.height, self.width, 3))
		reconstructed_image = Image.fromarray(reshaped_pixel_values.astype(np.uint8), 'RGB')

		reconstructed_image.save(f'{self.save_image_path}{self.name}_reconstructed.png')

		return 

HypersphereClass = Hypersphere(n = 6, image_path = "images/el_primo_rectangle.jpeg")
pixel_values, width, height = HypersphereClass.get_pixels()
statevectors, magnitudes = HypersphereClass.statevectors() 

Inverse_HypersphereClass = Inverse_Hypersphere(width = width, height = height, 
											   statevectors = statevectors, 
											   magnitudes = magnitudes, 
											   name = "el_primo_rectangle", 
											   save_image_path = "images/")
Inverse_HypersphereClass.recover_image()





	





