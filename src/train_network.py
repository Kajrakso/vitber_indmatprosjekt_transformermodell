from neural_network import NeuralNetwork
import numpy as np
from utils import onehot
from train_test_params import SortParams1, SortParams2, AddParams, TextGenParams
from layers_numba import FeedForward, Attention, EmbedPosition, Softmax, LinearLayer
from typing import Callable


def init_neural_network(
    t_params: SortParams1 | SortParams2 | AddParams | TextGenParams,
) -> NeuralNetwork:
    """Initializes a NeuralNetwork with the following layers:

    EmbedPosition -> Transformer -> LinearLayer (unembedding) -> Softmax

    Parameters are found in t_params.

    Args:
        t_params: See the dataclasses in `train_test_params.py`.

    Returns:
        an instance of NeuralNetwork"""

    embed_pos = EmbedPosition(n_max=t_params.n_max, m=t_params.m, d=t_params.d)

    transformer = [
        (
            Attention(d=t_params.d, k=t_params.k),
            FeedForward(d=t_params.d, p=t_params.p),
        )
        for _ in range(t_params.L)
    ]

    un_embed = LinearLayer(input_size=t_params.d, output_size=t_params.m)

    softmax = Softmax()

    network = NeuralNetwork(
        [
            embed_pos,
            *[
                t_layer
                for transformer_layer in transformer
                for t_layer in transformer_layer
            ],  # it unpacks a list of tuples: [(a,b), (c,d)] -> a, b, c, d
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
    n_iter: int,
    num_ints: int,
    alpha: float,
    dump_to_pickle_file: bool = True,
    file_name_dump: str = "nn_dump.pkl",
    verbose_logging: bool = False,
    dynamic_step_size: Callable[[int, float], float] | None = None,
) -> np.ndarray:
    """optimizes the parameters in the network using
    the Adam algorithm (Algorithm 3, p. 19 in the project description).
    If `dump_to_pickle_file` is set to `True` the network will be dumped every 10th iteration.

    Args:
        network: NeuralNetwork to train
        x_train: (num_batches, batch_size, n), input data
        y_train: (num_batches, batch_size, n_y), output data
        loss_func: Object function
        alpha: alpha paramater in the Adam algorithm (see p. 19, alg. 3)
        n_iter: number of iterations before the training ends
        num_ints: size of vocabulary (m in the project description)
        dump_to_pickle_file: bool. Defaults to True
        file_name_dump: default is `nn_dump.pkl`
        verbose_logging: logs more
        dymanic_step_size: function for computing a new learning rate each iteration

    Returns:
        array with values of the object function at each iteration
    """
    num_batches, batch_size, n = x_train.shape
    _, _, n_y = y_train.shape

    alpha_0 = alpha

    # prepare matrices
    pad_matrix = np.zeros((batch_size, num_ints, n - n_y))
    L_batches = np.zeros(num_batches)
    L = np.zeros(n_iter)

    print("\nstart training...\n")
    print(f"{'iter. step':>15} | {'object function L':>15}")
    print("-" * 40)

    for i in range(n_iter):
        for j in range(num_batches):
            Z = network.forward(onehot(x_train[j], num_ints))
            L_batches[j] = loss_func.forward(Z[:, :, -n_y:], y_train[j])
            grad_Z = loss_func.backward()
            grad_Z = np.concatenate(
                (pad_matrix, grad_Z), axis=2
            )  # pad grad_Z with zeros
            network.backward(grad_Z)
            network.step_adam(alpha)
            if verbose_logging and j % 10 == 0:
                print(f"done with batch {j}/{num_batches}")

        L[i] = np.mean(L_batches)
        print(f"{i+1:>15} | {L[i]:>15.10f}")

        if dump_to_pickle_file and (i + 1) % 10 == 0:
            print(f"network dumped at iteration {i+1}")
            network.dump(file_name_dump)
            L.dump(f"L_{file_name_dump}")

        if dump_to_pickle_file and (i + 1) % 100 == 0:
            print(f"Major-dump: network dumped at iteration {i+1}")
            network.dump(f"major_{i+1}_{file_name_dump}")

        if L[i] < 0.01:
            print("L < 0.01 so we break")
            break

        if dynamic_step_size is not None:
            alpha = dynamic_step_size(i, alpha_0)

    print("\nend of training...\n")
    return L
