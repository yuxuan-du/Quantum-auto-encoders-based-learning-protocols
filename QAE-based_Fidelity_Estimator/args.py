import argparse

parser = argparse.ArgumentParser()

group = parser.add_argument_group('Hyper-parameters of QAEs')


group.add_argument(
    '--n_qubit',
    type=int,
    default=8,
    help='number of qubits')

group.add_argument(
    '--num_blocks',
    type=int,
    default=7,
    help='number of layers of ansatz')


group.add_argument(
    '--n_latent_qubit',
    type=int,
    default=4,
    help='number of lantent qubits')

group.add_argument(
    '--seed', type=int, default=100, help='random seed, 0 for randomized')



group.add_argument('--lr', type=float, default=0.08, help='learning rate')

group.add_argument(
    '--epoch_num', type=int, default=201, help='maximum number of iterations of QAE')


group.add_argument(
    '--file_trained_name',
    type=str,
    default='out-fidelity-rho-kappa',
    help='directory prefix for output, empty for disabled')



args = parser.parse_args()
