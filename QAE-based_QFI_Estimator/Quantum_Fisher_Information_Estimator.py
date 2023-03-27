r'''
This file collects the code of QAE-based Quantum Fisher Information Estimator. The optimization of parameterized probe state
is completed by the parameter-shift-rule. The optimization of QAE to estimate Fidelity is completed by auto-grad.
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

dev_probe = qml.device('default.mixed', wires=args.n_qubit)
def hardware_efficient_ansatz_probe(paras_, wires):
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
            #qml.RZ(paras_[i, j, 0], wires=wires[j])
            qml.RY(paras_[i, j, 0], wires=wires[j])
            #qml.RZ(paras_[i, j, 2], wires=wires[j])
        # two-qubit gates
        for j in range(n_qubits - 1):
            qml.CNOT(wires=[j, j + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])

@qml.qnode(dev_probe)
def evolved_chanel(rho_, mu_, n_wires):
    # Interact probe state with environment
    r'''
    :param mu_: para of environment to be estimated
    :param wires: number of qubits
    '''
    qml.QubitDensityMatrix(rho_, wires=range(n_wires))
    for i in range(n_wires):
        qml.RZ(mu_, wires=[i])
    return qml.density_matrix(wires=range(n_wires))


@qml.qnode(dev_probe)
def Probe_state(phi,  n_wires_):
    r'''
    :param phi: trainable parameters
    :return: The probe state.
    # The wires in density_matrix argument gives the possibility to trace out a part of the system [qubits 0-2].
    '''
    hardware_efficient_ansatz_probe(phi, wires=range(n_wires_))

    return qml.density_matrix(wires=range(n_wires))

def cost_QFI(rho_, mu_, tau_, grad_=False):
    r'''
    :param rho_: probe state
    :para mu_: the para of environment to be estimated
    :param tau_: the hyper-para applied to mu_
    :return: QFI
    '''
    para_QAE = np.random.random(size=(layers_HEA_QAE, args.n_qubit, 3), requires_grad=True)
    rho_evolved = evolved_chanel(rho_, mu_, args.n_qubit)
    rho_evolved_tau = evolved_chanel(rho_, mu_ + tau_, args.n_qubit)
    Fidleity_estimated, trained_para = Fidelity_QAE(rho_evolved, rho_evolved_tau, para_QAE, args.n_latent_qubit, grad_)
    QFI = 8 * (1 - Fidleity_estimated**4) / tau_**2
    return QFI, trained_para

def cost_entropy_grad(rho_, mu_, para_QAE_, tau_, grad_=True):
    rho_evolved = evolved_chanel(rho_, mu_, args.n_qubit)
    rho_evolved_tau = evolved_chanel(rho_, mu_ + tau_, args.n_qubit)
    Fidleity_estimated, trained_para = Fidelity_QAE(rho_evolved, rho_evolved_tau, para_QAE_,  args.n_latent_qubit, grad_)
    QFI = 8 * (1 - Fidleity_estimated ** 4) / tau_**2
    return QFI, trained_para



if __name__ == '__main__':
    # Load data (i.e., define the parameter of environement
    mu = 0.1
    tau = 0.1
    # build folder to store data

    # load hyperparamters
    n_wires = args.n_qubit  # number of qubits
    layers_HEA_QAE = args.num_blocks  # number of layers of the employed ansatz for QAE
    layers_HEA_probe = args.num_blocks_probe # number of layers of the employed ansatz for parameterized GHZ state
    latent_qubits = args.n_latent_qubit  # number of latent qubits for QAE
    para = np.random.random(size=(layers_HEA_probe, n_wires, 1), requires_grad=False) #Initialize parameters of probe state
    my_log('The problem size for probe state is: {} qubit states'.format(n_wires))
    my_log('The problem size for QAE is: {} qubit states'.format(n_wires))
    my_log('The n_epochs for probe state updating is: {}'.format(args.epoch_num_probe))
    my_log('The n_epochs for QAE is: {}'.format(args.epoch_num))
    my_log('The lr of PQC is: {}'.format(args.lr))
    my_log('The exact result is:{}'.format(4 * n_wires ** 2))

    r'''Outer loop, update trainable parameters of Gibbs circuit to estimate target Gibbs states'''
    loss_probe_list = []
    para_probe_list = []
    para_QAE_initial_each_list = []
    para_QAE_trained_each_list = []
    QFI_estimated_list = []
    for epoch in range(args.epoch_num_probe):
        my_log('Epoch of Gibbs state optimization is {}'.format(epoch))
        probe_state = Probe_state(para, n_wires)
        probe_state.requires_grad = False
        loss, _ =  cost_QFI(probe_state, mu, tau, grad_=True)
        my_log('QFI is {}'.format(loss))
        print('QFI is :{}'.format(loss))
        QFI_estimated_list.append(loss)
        para_probe_list.append(para)

        # Inner loop of QAE
        # Start to update para of probe state based on para-shift rule
        grad_pos = np.zeros(shape=(layers_HEA_probe, n_wires, 1), requires_grad=False)
        grad_neg = np.zeros(shape=(layers_HEA_probe, n_wires, 1), requires_grad=False)
        for i in range(layers_HEA_probe):
            for j in range(n_wires):
                for k in range(1):
                    para_pos = para.copy()
                    para_neg = para.copy()
                    para_pos[i,j,k] += np.pi /2
                    para_neg[i, j, k] -= np.pi / 2
                    probe_state_pos = Probe_state(para_pos, n_wires)
                    grad_pos[i,j,k], _ = cost_QFI(probe_state_pos, mu, tau)
                    probe_state_neg = Probe_state(para_neg, n_wires)
                    grad_neg[i,j,k], _  = cost_QFI(probe_state_neg, mu, tau)
        para = para + args.lr * (grad_pos - grad_neg) / 2

    my_log('The optimized para of GHZ is:{}'.format(para))
    np.save('out/'+'QFI'+'latent'+str(args.n_latent_qubit)+'seed'+str(args.seed), QFI_estimated_list)
    np.save('out/' + 'parameters_all_probe' + 'latent' + str(args.n_latent_qubit)+'seed'+str(args.seed), para_probe_list)


