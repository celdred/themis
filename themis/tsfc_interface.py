import tsfc
from formmanipulation import split_form


class ThemisKernel():
	def __init__(self,kernel):
		self.assemblycompiled = False
		self.tsfckernel = kernel
		self.ast = kernel.ast
		self.integral_type = kernel.integral_type
		self.oriented = kernel.oriented
		self.coefficient_numbers = kernel.coefficient_numbers
		self.evaluate = 0
		#FACET RELATED STUFF?
		self.zero = False


def compile_form(form):
	

	idx_kernels_list = []
    # A map from all form coefficients to their number.
	coefficient_numbers = dict((c, n) for (n, c) in enumerate(form.coefficients()))
	for idx, f in split_form(form):
		arg = f.arguments()
		coeff = f.coefficients()
		numbering = f.coefficient_numbering()
		
        # Map local coefficient numbers (as seen inside the
        # compiler) to the global coefficient numbers
		number_map = dict((n, coefficient_numbers[c]) for (n, c) in enumerate(f.coefficients()))
		
		tsfc_kernels = tsfc.compile_form(f)
		
		kernels = []
		for kernel in tsfc_kernels:
			
			tkernel = ThemisKernel(kernel)
			tkernel.formdim = len(f.arguments())
			
			#map kernel coefficient numbers to global coefficient numbers
			numbers = tuple(number_map[c] for c in kernel.coefficient_numbers)
			tkernel.coefficient_map = numbers
			tkernel.coefficients = form.coefficients()
			tkernel.name = kernel.ast.operands()[0][1]
			tkernel.mesh = form.ufl_domain()
			kernels.append(tkernel)

		idx_kernels_list.append((idx,kernels))
	
	return idx_kernels_list
