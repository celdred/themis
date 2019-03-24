import tsfc
from formmanipulation import split_form
import tsfc.kernel_interface.firedrake as kernel_interface


class ThemisKernel():
    def __init__(self, kernel):
        self.assemblycompiled = False
        self.tsfckernel = kernel
        self.ast = kernel.ast
        self.integral_type = kernel.integral_type
        self.coefficient_numbers = kernel.coefficient_numbers
        self.zero = False
        self.interpolate = False


def compile_form(form):

    idx_kernels_list = []
# A map from all form coefficients to their number.
    coefficient_numbers = dict((c, n) for (n, c) in enumerate(form.coefficients()))
    for idx, f in split_form(form):

        # Map local coefficient numbers (as seen inside the
        # compiler) to the global coefficient numbers
        number_map = dict((n, coefficient_numbers[c]) for (n, c) in enumerate(f.coefficients()))

        tsfc_kernels = tsfc.compile_form(f, interface=kernel_interface.KernelBuilder)

        kernels = []
        for kernel in tsfc_kernels:
            tkernel = ThemisKernel(kernel)
            tkernel.formdim = len(f.arguments())

            # map kernel coefficient numbers to global coefficient numbers
            numbers = tuple(number_map[c] for c in kernel.coefficient_numbers)
            tkernel.coefficient_map = numbers
            tkernel.coefficients = form.coefficients()
            tkernel.name = kernel.ast.name
            tkernel.mesh = form.ufl_domain()
            tkernel.finatquad = kernel.quadrature_rule

            tkernel.tabulations = []

            # for tabname,pts in kernel.tabulations:
            for tabname, shape in kernel.tabulations:
                splittab = tabname.split('_')

                tabobj = {}
                tabobj['name'] = tabname

                # this matches the string generated in runtime_tabulated.py in FInAT
                # ie variant_order_derivorder_shiftaxis_{d,c}_restrict
                tabobj['variant'] = splittab[1]
                tabobj['order'] = int(splittab[2])
                tabobj['derivorder'] = int(splittab[3])
                tabobj['shiftaxis'] = int(splittab[4])
                if splittab[5] == 'd':
                    tabobj['discont'] = True
                if splittab[5] == 'c':
                    tabobj['discont'] = False
                tabobj['restrict'] = splittab[6]
                tabobj['shape'] = shape

                tkernel.tabulations.append(tabobj)

            kernels.append(tkernel)

        idx_kernels_list.append((idx, kernels))

    return idx_kernels_list
