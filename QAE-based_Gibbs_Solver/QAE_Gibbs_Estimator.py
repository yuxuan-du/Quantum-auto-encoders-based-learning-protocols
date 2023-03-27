r'''
This file collects the code of QAE-based Gibbs solver. The optimization of parameterized Gibbs state is completed
by the parameter-shift-rule. The optimization of QAE to estimate Von Neumann entropy is completed by auto-grad.
'''

import pennylane as qml
from pennylane import numpy as np
from argparse import Namespace
import os
from ops import *
from utilities import *
import math
from scipy.linalg import sqrtm


np.random.seed(args.seed)

dev_Gibbs = qml.device('default.qubit', wires=args.n_qubit_Gibbs)
def hardware_efficient_ansat_Gibbs(paras_, wires):
    # Implement hardware efficent ansatz according to Fig 3 and Fig 8 used to prepare parameterized Gibbs state
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


@qml.qnode(dev_Gibbs)
def Gibbs_state_compress(phi,  n_wires_, n_preserved):
    r'''
    :param phi: trainable parameters
    :return: The reduced parameterized Gibbs state.
    # The wires in density_matrix argument gives the possibility to trace out a part of the system [qubits 0-2].
    '''
    hardware_efficient_ansat_Gibbs(phi, wires=range(n_wires_))

    return qml.density_matrix(wires=range(n_preserved))

 

def cost_energy(rho_, Ham_):
    return np.trace(rho_ @ Ham_)

def cost_entropy(rho_, para_QAE_):
    entropy_estimated, trained_para = entropy_QAE(rho_, para_QAE_, latent_qubits=args.n_latent_qubit)
    return entropy_estimated, trained_para

def cost_entropy_grad(rho_, para_QAE_):
    entropy_estimated, trained_para = entropy_QAE_grad(rho_, para_QAE_, latent_qubits=args.n_latent_qubit)
    return entropy_estimated, trained_para



if __name__ == '__main__':

    # Data loading
    beta = args.beta
    state_Gibbs = np.load('dataset/Gibbs_state_target_beta_'+str(int(10*beta))+'.npy') #  load Gibbs state target
    Hamiltonian = hamiltonian_Gibbs(n_qubits = args.n_qubit, delta = -1)
    Hamiltonian = qml.matrix(Hamiltonian)

    # build folder to store data

    # load hyperparamters
    n_wires_Gibbs = args.n_qubit_Gibbs  # number of qubits
    n_wires_QAE = args.n_qubit  # number of qubits
    rho = state_Gibbs
    rho.requires_grad = False
    kappa = state_Gibbs
    kappa.requires_grad = False
    layers_HEA_QAE = args.num_blocks  # number of layers of the employed ansatz for QAE
    layers_HEA_Gibbs = args.num_blocks_Gibbs # number of layers of the employed ansatz for parameterized Gibbs state
    latent_qubits = args.n_latent_qubit  # number of latent qubits for QAE
    para = np.random.random(size=(layers_HEA_Gibbs, n_wires_Gibbs, 3), requires_grad=False)
    #Initialize trainable parameters of parameterized Gibbs state

    my_log('The problem size for Gibbs state is: {} qubit states'.format(n_wires_Gibbs))
    my_log('The problem size for QAE is: {} qubit states'.format(n_wires_QAE))
    my_log('The n_epochs for Gibbs is: {}'.format(args.epoch_num_Gibbs))
    my_log('The n_epochs for QAE is: {}'.format(args.epoch_num))
    my_log('The lr of PQC is: {}'.format(args.lr))
    my_log('The state fidety exact is:{}'.format(qml.math.fidelity(rho, kappa)))
    print('fidelity', qml.math.fidelity(rho, kappa) )

    r'''Outer loop, update trainable parameters of Gibbs circuit to estimate target Gibbs states'''
    loss_Gibbs_list = []
    para_Gibbs_list = []
    para_QAE_initial_each_list = []
    para_QAE_trained_each_list = []
    Fidelity_list = []
    for epoch in range(args.epoch_num_Gibbs):
        my_log('Epoch of Gibbs state optimization is {}'.format(epoch))
        para_Gibbs_list.append(para)
        compressed_Gibbs = Gibbs_state_compress(para, n_wires_Gibbs, n_wires_QAE)
        fidelity = np.sqrt(qml.math.fidelity(compressed_Gibbs, rho))
        my_log('Fidelity between the prepared and true Gibbs state is {}'.format(fidelity))
        print('Fidelity is :{}'.format(fidelity))
        Fidelity_list.append(fidelity)
        para_QAE = np.random.random(size=(layers_HEA_QAE, n_wires_QAE, 3), requires_grad=True)
        para_QAE_initial_each_list.append(para_QAE)
        loss_energy = cost_energy(compressed_Gibbs, Hamiltonian)
        # Inner loop of QAE
        loss_entropy, trained_para_QAE = cost_entropy(rho, para_QAE)
        para_QAE_trained_each_list.append(trained_para_QAE)
        loss = loss_energy -  loss_entropy / beta
        loss_Gibbs_list.append(loss)
        my_log('Loss is {}|| Energy loss is {}|| Entropy loss is {}'.format(loss, loss_energy, loss_entropy))

        # Start to update para of Gibbs based on para-shift rule
        grad_pos = np.zeros(shape=(layers_HEA_Gibbs, n_wires_Gibbs, 3))
        grad_neg = np.zeros(shape=(layers_HEA_Gibbs, n_wires_Gibbs, 3))
        for i in range(layers_HEA_Gibbs):
            for j in range(n_wires_Gibbs):
                for k in range(3):
                    para_pos = para.copy()
                    para_neg = para.copy()
                    para_QAE_pos = para_QAE.copy()
                    para_QAE_neg = para_QAE.copy()
                    para_pos[i,j,k] += np.pi /2
                    para_neg[i, j, k] -= np.pi / 2
                    compressed_Gibbs_pos = Gibbs_state_compress(para_pos, n_wires_Gibbs, n_wires_QAE)
                    loss_energy_pos = cost_energy(compressed_Gibbs_pos, Hamiltonian)
                    loss_entropy_pos, trained_para_QAE_grad_pos = cost_entropy_grad(compressed_Gibbs_pos, para_QAE_pos)
                    compressed_Gibbs_neg = Gibbs_state_compress(para_neg, n_wires_Gibbs, n_wires_QAE)
                    loss_energy_neg = cost_energy(compressed_Gibbs_neg, Hamiltonian)
                    loss_entropy_neg, trained_para_QAE_grad_neg = cost_entropy_grad(compressed_Gibbs_neg, para_QAE_neg)
                    grad_pos[i,j,k] = loss_energy_pos - loss_entropy_pos / beta
                    grad_neg[i,j,k] = loss_energy_neg - loss_entropy_neg / beta
        para = para - args.lr * (grad_pos - grad_neg) / 2

    my_log('The training loss of Gibbs is:{}'.format(loss_Gibbs_list[-1]))
    my_log('The optimized para of Gibbs is:{}'.format(para))


    np.save('out/'+'loss'+'latent'+str(args.n_latent_qubit)+'seed'+str(args.seed), loss_Gibbs_list)
    np.save('out/'+'fidelity_estmated'+'latent'+str(args.n_latent_qubit)+'seed'+str(args.seed), Fidelity_list)
    np.save('out/' + 'parameters_all_Gibbs' + 'latent' + str(args.n_latent_qubit)+'seed'+str(args.seed), para_Gibbs_list)


