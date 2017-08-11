import tsfc
from formmanipulation import split_form


class ThemisKernel():
    def __init__(self, kernel):
        self.assemblycompiled = False
        self.tsfckernel = kernel
        self.ast = kernel.ast
        self.integral_type = kernel.integral_type
        # self.oriented = kernel.oriented #DONT NEED THIS I THINK
        self.coefficient_numbers = kernel.coefficient_numbers
        self.evaluate = 0
        self.zero = False


def compile_form(form):

    # print('form')
    # print(form)
    # print(form.coefficients())
    # print(form.arguments())
    # print(form.coefficient_numbering())

    # CHECK THIS- ULTIMATELY NEED A LIST OF COEFFICIENTS THAT ARE FED INTO EACH SUBFORM
    # DO BY CREATING A DICTIONARY LINKING COEFFICIENTS AND NUMBERS
    # I THINK THE CURRENT WILL FAIL FOR FORMS WITH FACET INTEGRALS AND COEFFICIENTS THAT APPEAR ONLY IN ONE OR THE OTHER

    idx_kernels_list = []
# A map from all form coefficients to their number.
    coefficient_numbers = dict((c, n) for (n, c) in enumerate(form.coefficients()))
    for idx, f in split_form(form):

        # arg = f.arguments()
        # coeff = f.coefficients()
        # numbering = f.coefficient_numbering()

        # print(idx)
        # print(arg)

        # Map local coefficient numbers (as seen inside the
        # compiler) to the global coefficient numbers
        number_map = dict((n, coefficient_numbers[c]) for (n, c) in enumerate(f.coefficients()))

        tsfc_kernels = tsfc.compile_form(f)

        kernels = []
        for kernel in tsfc_kernels:

            tkernel = ThemisKernel(kernel)
            tkernel.formdim = len(f.arguments())

            # map kernel coefficient numbers to global coefficient numbers
            numbers = tuple(number_map[c] for c in kernel.coefficient_numbers)
            tkernel.coefficient_map = numbers
            tkernel.coefficients = form.coefficients()
            # tkernel.name = kernel.ast.operands()[0][1] #CAN DROP THIS I THINK IF WE ARE MORE CLEVER...
            tkernel.name = kernel.ast.name
            tkernel.mesh = form.ufl_domain()
            tkernel.finatquad = kernel.quad

            # print(kernel)
            tkernel.tabulations = []
            for tabulation in kernel.tabulations:  # THIS SHOULD HAVE A BETTER NAME
                splittab = tabulation.split('_')

                tabobj = {}
                tabobj['name'] = tabulation

                # this matches the string generated in runtime_tabulated.py in FInAT
                tabobj['variant'] = splittab[1]
                tabobj['order'] = int(splittab[2])
                tabobj['derivorder'] = int(splittab[3])
                tabobj['shiftaxis'] = int(splittab[4])
                if splittab[5] == 'd':
                    tabobj['discont'] = True
                if splittab[5] == 'c':
                    tabobj['discont'] = False
                tabobj['restriction'] = splittab[6]
                tkernel.tabulations.append(tabobj)

            kernels.append(tkernel)

        idx_kernels_list.append((idx, kernels))

    return idx_kernels_list
