from neural_network import NeuralNetwork
import numpy as np
from utils import onehot
import dill as pickle
from train_test_params import SortParams1, SortParams2, AddParams, TextGenParams
from layers_numba import FeedForward, Attention, EmbedPosition, Softmax, LinearLayer


def init_neural_network(t_params: SortParams1|SortParams2|AddParams|TextGenParams):
    """t_params: instance of a parameter data class. see the dataclasses in `train_test_params.py`."""
    transformer = [
        (
            FeedForward(d=t_params.d, p=t_params.p),
            Attention(d=t_params.d, k=t_params.k),
        )
        for _ in range(t_params.L)
    ]
    embed_pos = EmbedPosition(
        n_max=t_params.n_max, m=t_params.m, d=t_params.d
    )
    un_embed = LinearLayer(input_size=t_params.d, output_size=t_params.m)
    softmax = Softmax()

    network = NeuralNetwork(
        [
            embed_pos,

            # don't even try to understand this...
            *[
                t_layer
                for transformer_layer in transformer
                for t_layer in transformer_layer
            ],
            un_embed,
            softmax,
        ]
    )
    return network


def train_network(
    network: NeuralNetwork,
    x_train: np.ndarray,
    y_train: np.ndarray,
    loss_func,
    alpha: float,
    n_iter: int,
    num_ints: int,
    dump_to_pickle_file: bool = False,
    is_numba_dump: bool = True,
    file_name_dump: str = "nn_dump.pkl",
) -> None:
    """optimizes the paramaters in the network using the Adam algorithm

    Args:
        network: NeuralNetwork to train
        x_train: (num_batches, batch_size, n), input data
        y_train: (num_batches, batch_size, n_y), output data
        loss_func: Object function
        alpha: alpha paramater in the Adam algorithm (see p. 19, alg. 3)
        n_iter: number of iterations before the training ends
        num_ints: size of vocabulary (m in the project description)
        dump_to_pickle_file: bool
        is_numba_dump: bool. True if numba is used
        file_name_dump: default is `nn_dump.pkl`
    """
    num_batches, batch_size, n = x_train.shape
    _, _, n_y = y_train.shape

    pad_matrix = np.zeros((batch_size, num_ints, n - n_y))

    print("\nstart training...\n")
    print(f"{'iter. step':>15} | {'object function L':>15}")
    print("-" * 40)

    for i in range(n_iter):
        L = np.zeros(num_batches)

        for j in range(num_batches):
            Z = network.forward(onehot(x_train[j], num_ints))
            L[j] = loss_func.forward(Z[:, :, -n_y:], y_train[j])
            grad_Z = loss_func.backward()
            grad_Z = np.concatenate((pad_matrix, grad_Z), axis=2)   # pad grad_Z with zeros
            network.backward(grad_Z)
            network.step_adam(alpha)

        # TODO: use NeuralNetwork's interface for dumping both python and numba layers.
        if dump_to_pickle_file:
            if is_numba_dump:
                network.numba_dump(file_name_dump)
            else:
                with open(file_name_dump, "wb") as f:
                    pickle.dump(network, f)

        print(f"{i+1:>15} | {np.mean(L):>15.10f}")
    print("\nend of training...\n")