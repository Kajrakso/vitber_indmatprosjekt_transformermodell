from neural_network import NeuralNetwork
import numpy as np
from utils import onehot


def test_trained_network(
    network: NeuralNetwork, x_test: np.ndarray, y_test: np.ndarray, num_ints: int
) -> None:
    """After having trained the network it should be tested on
    data it was not trained on.

    Args:
        network: pre-trained network
        x_test: (num_batches, batch_size, n)
        y_test: (num_batches, batch_size, n_y)
        num_ints: size of vocabulary (m in the project description)
    """
    num_batches_x, batch_size_x, n = x_test.shape
    _, _, r = y_test.shape

    print("\nstart testing...\n")
    print(f"{'batch nr.':>15} | {'percentage correct':>15}")
    print("-" * 40)

    # prepare a matrix for holding the sequences (3.2.2 in proj. desc.)
    x = np.zeros((batch_size_x, n + r))

    for i in range(num_batches_x):
        x[:, :n] = x_test[i]

        for j in range(r):
            Z = network.forward(onehot(x[:, : n + j], m=num_ints))
            x[:, n + j] = np.argmax(Z, axis=1)[:, -1]
        y = x[:, -r:]

        num_correct = np.sum(np.all(y_test[i] == y, axis=1))
        print(f"{i+1:>15} | {num_correct/batch_size_x:.3%}")

    print("\nend of testing...\n")