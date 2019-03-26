
backend = 'themis'

if backend == 'firedrake':
    from firedrake import *

    import matplotlib.pyplot as plt
    def plot_function(func, coords, name):
        plt.close('all')
        fig = plt.figure(figsize=(10, 10))
        axes = fig.add_subplot(111)
        if len(func.ufl_shape) == 1 and func.ufl_shape[0] == 1:
            plot(func[0], axes=axes)
        else:
            plot(func, axes=axes)
        fig.savefig(name + '.png')
        plt.close('all')

    def get_plotting_spaces(mesh, nquadplot, nquadplot_v=None):
        return None,None,None

    def evaluate_and_store_field(evalspace, opts, field, name, checkpoint):
        return field

    BoxMesh = None
    coordvariant = None

if backend == 'themis':

    from themis import *
    CubedSphereMesh = None
    IcosahedralSphereMesh = None
    coordvariant = 'feec'
