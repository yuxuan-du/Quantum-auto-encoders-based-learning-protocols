#================================================================================================================================
# This file contains the code of implemening QAE-based fidelity estimator for low-rank states.
# To estimate full rank states, substitute the data loading directory in Lines 76-77 to  'dataset/Task2_state_rho.npy'. 
#===============================================================================================================================


import pennylane as qml
from pennylane import numpy as np
from argparse import Namespace
import os
from utilities import *
import math
from scipy.linalg import sqrtm
from pennylane.optimize import GradientDescentOptimizer, AdamOptimizer 

np.random.seed(args.seed)
dev = qml.device('default.mixed', wires=args.n_qubit)
def hardware_efficient_ansatz(paras_, wires):
    # Implement hardware efficent ansatz according to Fig 3
    r'''
    :param paras_: trainable parameters; shape is [Layer, num_qubits, 3]
    :param wires: number of qubits
    '''
    Layers = paras_.shape[0]
    n_qubits = len(wires)
    for i in range(Layers):
        # single-qubit gates
        for j in range(n_qubits):
            qml.RZ(paras_[i, j, 0], wires=wires[j])
            qml.RY(paras_[i, j, 1], wires=wires[j])
            qml.RZ(paras_[i, j, 2], wires=wires[j])
        # two-qubit gates
        for j in range(n_qubits - 1):
            qml.CNOT(wires=[j, j + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])


@qml.qnode(dev)
def encoder_compressed_state(rho, preserved_qubits_):
    r'''
    :param rho: input mixed state
    :param phi: trainable parameters
    :preserved_qubits_: preserved_qubits denotes K in paper.
    :return: The reduced density mat.
    # The wires in density_matrix argument gives the possibility to trace out a part of the system.
    '''
    qml.QubitDensityMatrix(rho, wires=range(n_wires))

    return qml.density_matrix(wires=range(preserved_qubits_))


@qml.qnode(dev)
def encoder_rubbish_state(rho, phi, preserved_qubits_):
    r'''
    :param rho: input mixed state
    :param phi: trainable parameters
    :preserved_qubits_: preserved_qubits denotes K in paper.
    :return: The reduced density mat (rubbish state for QAE).
    # The wires in density_matrix argument gives the possibility to trace out a part of the system.
    '''
    qml.QubitDensityMatrix(rho, wires=range(n_wires))
    hardware_efficient_ansatz(phi, wires=range(n_wires))
    return qml.probs(wires=range(preserved_qubits_ , n_wires, 1))  

def cost_QAE( para_, rho, preserved_qubits_):
    reduced_density = encoder_rubbish_state(rho, para_, preserved_qubits_)
    loss = 1 - reduced_density[0]
    return loss




if __name__ == '__main__':

    # Data loading
    state_rho = np.load('dataset/state_rho.npy')  
    state_kappa = np.load('dataset/state_kappa.npy') 
    # build folder to store data

    # load hyperparamters
    n_wires = args.n_qubit  # number of qubits
    rho = state_rho  
    rho.requires_grad = False
    kappa = state_kappa 
    kappa.requires_grad = False
    layers_HEA = args.num_blocks  # number of layers of the employed ansatz
    latent_qubits = args.n_latent_qubit
    para = np.random.normal(loc=0, scale = 1.4, size=(layers_HEA, n_wires, 3), requires_grad=True)  # number of trainable parameters

    my_log('The problem size is: {} qubit states'.format(n_wires))
    my_log('The n_epochs is: {}'.format(args.epoch_num))
    my_log('The lr of PQC is: {}'.format(args.lr))
    my_log('The state fidety exact is:{}'.format(qml.math.fidelity(rho, kappa)))
    print('fidelity', np.sqrt(qml.math.fidelity(rho, kappa)) )

    r'''First stage, train QAE to obtain compressed states'''
    loss_list = []
    reduced_den_mat_list = []
    para_list = []
    opt = AdamOptimizer(args.lr)
    for epoch in range(args.epoch_num):
        loss = cost_QAE( para, rho, latent_qubits)
        loss_list.append(loss)
        para_list.append(para)
        para, _, _ = opt.step(cost_QAE, para, rho, latent_qubits)
        print('Epoch {}|| Loss {}'.format(epoch, loss))
    my_log('The training loss of QAE is:{}'.format(loss_list[-1]))
    my_log('The optimized para of QAE is:{}'.format(para))


    np.save('out/'+'loss'+'latent'+str(args.n_latent_qubit)+'seed'+str(args.seed), loss_list)
    np.save('out/'+'reduced density matrix'+'latent'+str(args.n_latent_qubit)+'seed'+str(args.seed), reduced_den_mat_list)
    np.save('out/' + 'parameters_all' + 'latent' + str(args.n_latent_qubit)+'seed'+str(args.seed), para_list)


    r'''Second stage, eigen decompositon of compressed states'''
    # Following Fig. 2, we apply quantum state tomography to the reduced density matrix and get its eigen-decomposition
    # Compute the reconstructed state based on Eq. (12)
    unitary_trained = qml.matrix(hardware_efficient_ansatz)
    trained_para_QAE = para_list[-1]
    trained_para_QAE.requires_grad = False
    unitary_encoder = unitary_trained(trained_para_QAE, range(n_wires))
    state_before_compress = unitary_encoder @ rho @ unitary_encoder.conj().T

    mea_ops = np.zeros(shape=(2 ** (n_wires - latent_qubits), 2 ** (n_wires - latent_qubits)))
    mea_ops[0, 0] = 1
    post_mea_ops = np.kron(np.eye(2 **  latent_qubits), mea_ops)
    state_after_mea_before_compress = post_mea_ops @ state_before_compress @ post_mea_ops.conj().T
    my_log('The trace of state after measurement is:{}'.format(np.trace(state_after_mea_before_compress)))

    state_after_mea_before_compress = state_after_mea_before_compress / np.trace(state_after_mea_before_compress)
    compressed_state =  encoder_compressed_state(state_after_mea_before_compress, latent_qubits)

    outputstate = np.kron(compressed_state, mea_ops)
    recovered_state_rho = unitary_encoder.conj().T @ outputstate @ unitary_encoder
    my_log('The fidelity of reconstructed and true without eigen is:{}'.format(np.sqrt(qml.math.fidelity(rho, recovered_state_rho))))
 
    eigen_values, eigen_vects = exact_gs(compressed_state)
    print('test reconstruction')
    state_recons = np.zeros(shape = (2 ** latent_qubits, 2 ** latent_qubits), dtype = 'complex128')
    for i in range(len(eigen_values)):
        tempt = np.outer(eigen_vects[:,i], eigen_vects[:,i].conj())
        state_recons += eigen_values[i] * tempt

    my_log('error reconstruction :{}'.format(np.linalg.norm(compressed_state - state_recons) ))
    #Compute matrix W in Eq. (13)

    r'''Third stage, compute mat W'''
    W_mat = np.zeros(shape=(2 ** latent_qubits, 2 ** latent_qubits), dtype='complex128')
    rebuilded_state = unitary_encoder @ kappa @ unitary_encoder.conj().T
    for i in range(2 ** latent_qubits):
        for j in range(2 ** latent_qubits):
            mea_ops_temt = np.outer(eigen_vects[:, i], eigen_vects[:, j].conj().T)
            mea_ops_temt = np.kron(mea_ops_temt, mea_ops)
            tempt = np.trace(rebuilded_state @ mea_ops_temt)
            tempt2 = np.sqrt(eigen_values[i] * eigen_values[j]) * tempt
            W_mat[i, j] = tempt2

    my_log('The Wmat QAE is:{}'.format(W_mat))

    r'''Fourth stage, compute fidelity'''
    Fidelity_est = np.trace(sqrtm(W_mat))  
    my_log('The training fide QAE is:{}'.format(Fidelity_est))
    print('fide_est', Fidelity_est)
