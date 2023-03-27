import argparse

parser = argparse.ArgumentParser()

group = parser.add_argument_group('Hyper-parameters of QAE-based Gibbs state solver')


 

group.add_argument(
    '--n_qubit_Gibbs',
    type=int,
    default=4,
    help='number of qubits for variaitonal Gibbs circuits')

group.add_argument(
    '--n_qubit',
    type=int,
    default=3,
    help='number of qubits for QAE')

group.add_argument(
    '--num_blocks_Gibbs',
    type=int,
    default=5,
    help='number of layers of ansatz for preparing parameterized Gibbs state')

group.add_argument(
    '--num_blocks',
    type=int,
    default=4,
    help='number of layers of ansatz for QAE')


group.add_argument(
    '--n_latent_qubit',
    type=int,
    default=2,
    help='number of lantent qubits')

group.add_argument(
    '--seed', type=int, default=1, help='random seed, 0 for randomized')



group.add_argument('--lr', type=float, default=0.2, help='learning rate')
group.add_argument('--beta', type=float, default=4, help='Inverse temperature for Gibbs state')

group.add_argument(
    '--epoch_num', type=int, default=101, help='maximum number of iterations to train QAE')

group.add_argument(
    '--epoch_num_Gibbs', type=int, default=201, help='maximum number of iterations to train para Gibbs')


group.add_argument(
    '--file_trained_name',
    type=str,
    default='out-res-Gibbs',
    help='directory prefix for output, empty for disabled')



args = parser.parse_args()
