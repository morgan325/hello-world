# -*- coding: utf-8 -*-

"""

This is a converter to Tecplot data format.

"""

import numpy as np


# ==================================================
# Test data generators
# ==================================================

def generate_unstructured_data(n=1000):
    """
    Unstructured (scatter) data:
    data(n), x(n), y(n), z(n)
    """
    x = np.random.rand(n)
    y = np.random.rand(n)
    z = np.random.rand(n)
    data = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * z
    return data, x, y, z


def generate_structured_data(nx=50, ny=40, nz=30):
    """
    Structured grid data:
    data(nx, ny, nz), x(nx), y(ny), z(nz)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    data = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * Z
    return data, x, y, z



# ==================================================
# Data loading
# ==================================================


# ==================================================
# Tecplot data writing
# ==================================================


def write_structured_tecplot(filename, variables, X=[], Y=[], Z=[]):
    """
    X, Y, Z are the lists of xyz coordinates. If not provided, intergers
    from 0 will be used.

    `variables` is a dict of the variables to store with the variable names as
    the keys. Each variable should be 2 or 3 dimensional array using numpy's
    row-major order.

    Check the test function to see how to create input data structure.

    Notice that tecplot format use 'column-major order' as in Fortran, which is
    different from that of Numpy or C.
    """
    if filename[-4:] != '.txt':
        filename += '.txt'

    with open(filename, 'w') as f:
        ## 2D case
        if len(Z) == 0:
            f.write('Variables = "X", "Y"')
            for key in variables.keys():
                f.write(', "' + key + '"')
            f.write('\n\nZone I='+str(len(X))+', J='+str(len(Y))+', F=POINT\n')

            for j in range(len(Y)):
                for i in range(len(X)):
                    f.write(str(X[i]) + ' ' + str(Y[j]))
                    for var in variables.values():
                        f.write(' ' + str(var[i,j]))
                    f.write('\n')

        ## 3D case
        else:
            f.write('Variables = "X", "Y", "Z"')
            for key in variables.keys():
                f.write(', "' + key + '"')
            f.write('\n\nZone I =' + str("{:6}".format(len(X))) + ', J =' + str("{:6}".format(len(Y))) + ', K =' + str("{:6}".format(len(Z))) + ', F=POINT\n')

            for k in range(len(Z)):
                for j in range(len(Y)):
                    for i in range(len(X)):
                        f.write(str("{:.6E}".format(X[i])) + ' ' + str("{:.6E}".format(Y[j])) + ' ' + str("{:.6E}".format(Z[k])))
                        for var in variables.values():
                            f.write(' ' + str("{:.6E}".format(var[i,j,k])))
                        f.write('\n')
                        
                        
                        
def write_unstructured_tecplot(filename, variables, x, y, z, zone_name="zone1"):
    """
    Write non-FE (scatter) Tecplot ASCII data with DATAPACKING=POINT.

    Parameters
    ----------
    filename : str
        Output file name (.dat recommended).
    variables : dict
        {var_name: array_like}, each variable must be shape (n_points,)
    x, y : array_like
        Coordinates, shape (n_points,)
    z : array_like or None
        If provided -> 3D scatter, shape (n_points,)
        If None     -> 2D scatter
    zone_name : str
        Tecplot zone name

    Notes
    -----
    - This writer DOES NOT use I/J/K
    - Zone type is scatter (POINT)
    - Suitable for unstructured / non-FE data
    """

    # ---------- filename ----------
    if not (filename.endswith(".dat") or filename.endswith(".txt")):
        filename += ".dat"
    
    # ---------- coordinates ----------
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = x.size

    if y.size != n:
        raise ValueError("x and y must have the same length")

    is_3d = z is not None
    if is_3d:
        z = np.asarray(z).ravel()
        if z.size != n:
            raise ValueError("z must have the same length as x and y")
            
    with open(filename, 'w') as f:
        ## 2D case
        if len(z) == 0:
            f.write('Variables = "x" "y"')
            for key in variables.keys():
                f.write(', "' + key + '"\n')
            f.write(f'ZONE T="{zone_name}", I={n}, DATAPACKING=POINT\n')

            for i in range(len(x)):
                f.write(str(x[i]) + ' ' + str(y[i]))
                for var in variables.values():
                    f.write(' ' + str(var[i]))
                f.write('\n')

        ## 3D case
        else:
            f.write('Variables = "x" "y" "z"')
            for key in variables.keys():
                f.write(', "' + key + '"\n')
            f.write(f'ZONE T="{zone_name}", I={n}, DATAPACKING=POINT\n')

            for i in range(len(x)):
                f.write(str("{:.6E}".format(x[i])) + ' ' + str("{:.6E}".format(y[i])) + ' ' + str("{:.6E}".format(z[i])))
                for var in variables.values():
                    f.write(' ' + str("{:.6E}".format(var[i])))
                f.write('\n')
                        
# ==================================================
# Main workflow
# ==================================================

def main():
    # ----------------------------------------------
    # Control flag
    # ----------------------------------------------
    # test_data = 0
    test_data = 1
    
    idx = 0
    # idx = 1

    print(f">>> Grid type idx = {idx}")

    # ----------------------------------------------
    # Data generation
    # ----------------------------------------------
    if test_data == 1:
        
        if idx == 0:
            print(">>> Generating UNSTRUCTURED dataset")
            data, x, y, z = generate_unstructured_data(n=5000)

            print("    data shape:", data.shape)
            print("    x,y,z shape:", x.shape, y.shape, z.shape)

            # 后续操作（示例）
            write_unstructured_tecplot('unstructure_3d', {'rad': data}, x, y, z)
            # POD / DMD / visualization

        elif idx == 1:
            print(">>> Generating STRUCTURED dataset")
            data, x, y, z = generate_structured_data(nx=60, ny=40, nz=20)

            print("    data shape:", data.shape)
            print("    x,y,z shape:", x.shape, y.shape, z.shape)

            # 后续操作（示例）
            write_structured_tecplot('test_structure_3d', {'rad': data}, x, y, z)
            # write_structured_tecplot('test_structure_2d', {'rad': data}, x, y)
            # LES / contour / iso-surface

        else:
            raise ValueError("idx must be 0 (unstructured) or 1 (structured)")

        print(">>> Workflow finished")
    
    else:
        
        if idx == 0:
            print(">>> Loading UNSTRUCTURED dataset")
            #
            #
            print(">>> Writing Tecplot UNSTRUCTURED dataset")
            write_unstructured_tecplot('unstructure_3d', {'rad': data}, x, y, z)
        
        elif idx == 1:
            print(">>> Loading STRUCTURED dataset")
            
            
            print(">>> Writing Tecplot STRUCTURED dataset")
            write_structured_tecplot('structure_3d', {'rad': data}, x, y, z)
            # write_structured_tecplot('test_structure_2d', {'rad': data}, x, y)
            
        else:
            raise ValueError("idx must be 0 (unstructured) or 1 (structured)")

        print(">>> Workflow finished")


if __name__ == "__main__":
    main()






def npz2tecplot(input_file, filename=None):
    if filename == None:
        filename = input_file + '_.dat'
    npz = np.load(input_file)
    var_names = npz.keys()
    data = {var: npz[var] for var in var_names}

    ## sometimes the velocity is 3d but the data is actually 2d, so it's quite
    ## tricky to figure out the actual dimension.
    dims = [len(data[var].shape) for var in var_names]
    print('dims', dims)
    if min(dims) == 2:
        dim = 2
    elif len(dims) == 1 and ('u' in var_names[0]):  # only u stored
        dim = 2
    else:
        dim = 3

    xlen, ylen = data[var_names[0]].shape[:2]
    if dim == 3:
        zlen = data[var_names[0]].shape[2]
        tecplot_writer(filename, data, range(xlen), range(ylen), range(zlen))
        return
    else:
        ## handling the 3d arrays (velocity, etc) requires careful work
        if max(dims) == 2:
            tecplot_writer(filename, data, range(xlen), range(ylen))
            return
        else:
            data = {}
            for var in var_names:
                if len(npz[var].shape) == 3:
                    for i in range(npz[var].shape[2]):
                        ## this is very specific to my own custom
                        data[var+str(i)] = npz[var][:,:,i]
                else:
                    data[var] = npz[var]
            tecplot_writer(filename, data, range(xlen), range(ylen))
            return


