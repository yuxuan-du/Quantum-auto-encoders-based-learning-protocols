# ==========================================================================  
# The codes presented below are used to compute Von-Neumann entropy
# ========================================================================= 

from pennylane import numpy as np
import pennylane as qml
from args import args
from utilities import *
from pennylane.optimize import GradientDescentOptimizer, AdamOptimizer
import os
import scipy

np.random.seed(args.seed)

dev_QAE = qml.device('default.mixed', wires=args.n_qubit)
def hardware_efficient_ansatz_QAE(paras_, wires):
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


@qml.qnode(dev_QAE)
def encoder_compressed_state(rho, preserved_qubits_):
    r'''
    :param rho: input mixed state
    :param phi: trainable parameters
    :preserved_qubits_: preserved_qubits denotes K in paper.
    :return: The reduced density mat.
    # The wires in density_matrix argument gives the possibility to trace out a part of the system.
    '''
    qml.QubitDensityMatrix(rho, wires=range(args.n_qubit))

    return qml.density_matrix(wires=range(preserved_qubits_))

@qml.qnode(dev_QAE)
def encoder_rubbish_state(rho, phi, preserved_qubits_):
    r'''
    :param rho: input mixed state
    :param phi: trainable parameters
    :preserved_qubits_: preserved_qubits denotes K in paper.
    :return: The reduced density mat (rubbish state for QAE).
    # The wires in density_matrix argument gives the possibility to trace out a part of the system.
    '''
    qml.QubitDensityMatrix(rho, wires=range(args.n_qubit))
    hardware_efficient_ansatz_QAE(phi, wires=range(args.n_qubit))
    return qml.probs(wires=range(preserved_qubits_ , args.n_qubit, 1))  



def cost_QAE( para_,  rho, preserved_qubits_):
    reduced_density = encoder_rubbish_state(rho, para_, preserved_qubits_)
    loss = 1 - reduced_density[0]
    return loss


dev_QAE_grad = qml.device('default.mixed', wires=args.n_qubit)

@qml.qnode(dev_QAE_grad)
def encoder_rubbish_state_grad(rho, phi, preserved_qubits_):
    r'''
    :param rho: input mixed state
    :param phi: trainable parameters
    :preserved_qubits_: preserved_qubits denotes K in paper.
    :return: The reduced density mat (rubbish state for QAE).
    # The wires in density_matrix argument gives the possibility to trace out a part of the system.
    '''
    qml.QubitDensityMatrix(rho, wires=range(args.n_qubit))
    hardware_efficient_ansatz_QAE(phi, wires=range(args.n_qubit))
    return qml.probs(wires=range(preserved_qubits_ , args.n_qubit, 1))

def cost_QAE_grad_pos(para_, rho,  preserved_qubits_):
    reduced_density = encoder_rubbish_state_grad(rho, para_, preserved_qubits_)
    loss = 1 - reduced_density[0]

    return loss  



def entropy_QAE(rho, para, latent_qubits):
    loss_list = []
    reduced_den_mat_list = []
    para_list = []
    opt_loss = AdamOptimizer(0.02)
    for epoch in range(args.epoch_num):
        para, _, _ = opt_loss.step(cost_QAE, para, rho, latent_qubits)
        loss = cost_QAE(para, rho, latent_qubits)
        loss_list.append(loss)
        para_list.append(para)
 

    unitary_trained = qml.matrix(hardware_efficient_ansatz_QAE)
    trained_para_QAE = para_list[-1]
    trained_para_QAE.requires_grad = False
    unitary_encoder = unitary_trained(trained_para_QAE, range(args.n_qubit))
    state_before_compress = unitary_encoder @ rho @ unitary_encoder.conj().T

    mea_ops = np.zeros(shape=(2 ** (args.n_qubit - latent_qubits), 2 ** (args.n_qubit - latent_qubits)))
    mea_ops[0, 0] = 1
    post_mea_ops = np.kron(np.eye(2 ** latent_qubits), mea_ops)
    state_after_mea_before_compress = post_mea_ops @ state_before_compress @ post_mea_ops.conj().T
     

    state_after_mea_before_compress = state_after_mea_before_compress / np.trace(state_after_mea_before_compress)
    compressed_state = encoder_compressed_state(state_after_mea_before_compress, latent_qubits)

    eigen_values, eigen_vects = exact_gs(compressed_state)
    print('est loss entroy', loss_list[-1])
    # compute entropy
    von_Neumann = 0
    for i in range(len(eigen_values)):
        von_Neumann += eigen_values[i] * np.log(eigen_values[i])
    return von_Neumann, para

def entropy_QAE_grad(rho, para, latent_qubits):
    opt_pos = AdamOptimizer(0.02)
    for epoch in range(args.epoch_num):
        para, _, _ = opt_pos.step(cost_QAE_grad_pos, para, rho, latent_qubits )

    unitary_trained = qml.matrix(hardware_efficient_ansatz_QAE)
    trained_para_QAE = para
    trained_para_QAE.requires_grad = False
    unitary_encoder = unitary_trained(trained_para_QAE, range(args.n_qubit))
    state_before_compress = unitary_encoder @ rho @ unitary_encoder.conj().T

    mea_ops = np.zeros(shape=(2 ** (args.n_qubit - latent_qubits), 2 ** (args.n_qubit - latent_qubits)))
    mea_ops[0, 0] = 1
    post_mea_ops = np.kron(np.eye(2 ** latent_qubits), mea_ops)
    state_after_mea_before_compress = post_mea_ops @ state_before_compress @ post_mea_ops.conj().T

    state_after_mea_before_compress = state_after_mea_before_compress / np.trace(state_after_mea_before_compress)
    compressed_state = encoder_compressed_state(state_after_mea_before_compress, latent_qubits)

    eigen_values, eigen_vects = exact_gs(compressed_state)

    # compute entropy
    von_Neumann = 0
    for i in range(len(eigen_values)):
        if eigen_values[i] == 0:
            continue # Avoid log 0
        von_Neumann -= eigen_values[i] * np.log(eigen_values[i]) 


    return von_Neumann, para



def hamiltonian_Gibbs(n_qubits: int, delta: float, Periodic = False) -> qml.Hamiltonian:
    """ Creates the hamiltonian, which is given by:
    Args:
        n_qubits(int): number of spins in the chain.
        delta(float): delta parameter. J_Z
        eta(float): eta parameter. J_X
    """
    hamiltonian = []
    coeffs = []


    for qubits in range(n_qubits - 1):
        hamiltonian.append(qml.PauliZ(qubits) @ qml.PauliZ(qubits + 1))
        coeffs.append(delta)
    hamiltonian.append(qml.PauliZ(n_qubits - 1) @ qml.PauliZ(0))
    coeffs.append(delta)

    H = qml.Hamiltonian(coeffs, hamiltonian, simplify=True)

    return H
