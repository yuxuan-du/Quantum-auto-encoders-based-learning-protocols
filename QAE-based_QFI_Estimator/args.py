import argparse

parser = argparse.ArgumentParser()

group = parser.add_argument_group('Hyper-parameters of QAE-based QFI estimator')

 


group.add_argument(
    '--n_qubit',
    type=int,
    default=4,
    help='number of qubits for probe state')

group.add_argument(
    '--num_blocks_probe',
    type=int,
    default=3,
    help='number of layers of ansatz for parameterized probe state')

group.add_argument(
    '--num_blocks',
    type=int,
    default=5,
    help='number of layers of ansatz for quantum auto-encoder')


group.add_argument(
    '--n_latent_qubit',
    type=int,
    default=2,
    help='number of lantent qubits for quantum auto-encoder')

group.add_argument(
    '--seed', type=int, default=1, help='random seed, 0 for randomized')



group.add_argument('--lr', type=float, default=0.08, help='learning rate')

group.add_argument(
    '--epoch_num', type=int, default=301, help='maximum number of iterations to train QAE')

group.add_argument(
    '--epoch_num_probe', type=int, default=75, help='maximum number of iterations to train para GHZ')


group.add_argument(
    '--file_trained_name',
    type=str,
    default='out-res-QFI',
    help='directory prefix for output, empty for disabled')



args = parser.parse_args()
