from neural_network import NeuralNetwork
import numpy as np
from utils import onehot
import dill as pickle


def train_network(
    network: NeuralNetwork,
    x_train: np.ndarray,
    y_train: np.ndarray,
    loss_func,
    alpha: float,
    n_iter: int,
    num_ints: int,
    dump_to_pickle_file: bool = False,
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
        file_name_dump: default is `nn_dump.pkl`
    """
    num_batches = np.shape(x_train)[0]
    n_y = np.shape(y_train)[-1]

    print("\nstart training...\n")
    print(f"{'iter. step':>15} | {'object function L':>15}")
    print("-" * 40)

    for i in range(n_iter):
        L = np.zeros(num_batches)
        for j in range(num_batches):
            Z = network.forward(onehot(x_train[j], num_ints))
            L[j] = loss_func.forward(Z[:, :, -n_y:], y_train[j])
            grad_Z = loss_func.backward()
            _ = network.backward(grad_Z)
            _ = network.step_adam(alpha)

        if dump_to_pickle_file:
            with open(file_name_dump, "wb") as f:
                pickle.dump(network, f)

        print(f"{i+1:>15} | {np.mean(L):>15.10f}")
    print("\nend of training...\n")