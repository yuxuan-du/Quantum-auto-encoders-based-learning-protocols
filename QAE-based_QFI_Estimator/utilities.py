from pennylane import numpy as np
from args import args


def exact_gs(reduced_mat):
    """ Calculates the eigen-decomposition of the reduced density matrix.
        
    Args:
        reduced_mat: reduced density matrix.

    Output:
        float: outputs the eigen decomposition results.
    """
    state_real = reduced_mat
    eigen_values, evecs = np.linalg.eig(state_real)
    
    return eigen_values, evecs  


def my_log(s):
    if args.file_trained_name:
        with open('out/' + args.file_trained_name +'seed'+str(args.seed) + '.log', 'a', newline='\n') as f:
            f.write(s + u'\n')

