from neural_network import NeuralNetwork
from layers import (
    LinearLayer,
    CrossEntropy,
    EmbedPosition,
    Softmax,
    FeedForward,
    Attention,
)
import layers_numba as nl
from data_generators import get_train_test_sorting, get_train_test_addition
import numpy as np
from utils import onehot
import dill as pickle


def alg4(
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
            Z = network.forward(onehot(x_train[j], m=num_ints))
            L[j] = loss_func.forward(y_hat=Z[:, :, -n_y:], y=y_train[j])
            grad_Z = loss_func.backward()
            _ = network.backward(grad_Z)
            _ = network.step_adam(alpha)

        if dump_to_pickle_file:
            with open(file_name_dump, "wb") as f:
                pickle.dump(network, f)

        print(f"{i+1:>15} | {np.mean(L):>15.10f}")
    print("\nend of training...\n")


def test_trained_network_sorting(
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
    num_batches_y, batch_size_y, r = y_test.shape

    print("\nstart testing...\n")
    print(f"{'batch nr.':>15} | {'percentage correct':>15}")
    print("-" * 40)

    # prepare a x matrix for holding the sequences (3.2.2 in proj. desc.)
    x = np.zeros((batch_size_x, n + r))
    for i in range(num_batches_x):
        x[:, :n] = x_test[i]
        for j in range(r):
            Z = network.forward(onehot(x[:, : (n + j)], m=num_ints))
            x[:, n + j] = np.argmax(Z, axis=1)[:, -1]
        y = x[:, -r:]

        num_correct = np.sum(np.all(y_test[i] == y, axis=1))
        print(f"{i+1:>15} | {num_correct/batch_size_x:.3%}")
    print("\nend of testing...\n")


def main():
    #
    # this should probably be done in the notebook for handin
    #

    # alpha parameter for Adam
    alpha = 0.001

    # generate a training set as per
    # exercise 3.3

    D = 250  # number of datapoint (x, y)
    b = 10  # number of batches

    r = 7  # length of the sequences
    n_max = 2 * r - 1
    m = 5  # number of symbols

    d = 20  # output dimension for the linear layer.
    k = 10
    p = 25
    L = 2

    n_iter = 300  # force the training to stop after n_iter steps.

    # initialize the network and the loss function
    softmax = Softmax()
    feed_forward = FeedForward(d=d, p=p)
    feed_forward2 = FeedForward(d=d, p=p)
    attention = Attention(d=d, k=k)
    attention2 = Attention(d=d, k=k)
    embed_pos = EmbedPosition(n_max=n_max, m=m, d=d)
    un_embed = LinearLayer(input_size=d, output_size=m)
    softmax = Softmax()

    network = NeuralNetwork(
        [
            embed_pos,
            attention,
            feed_forward,
            attention2,
            feed_forward2,
            un_embed,
            softmax,
        ]
    )

    # with open("nn_dump.pkl", "rb") as f:
    #     network = pickle.load(f)

    loss = CrossEntropy()

    # prepare training and test data for sorting
    training_data = get_train_test_sorting(
        length=r,
        num_ints=m,
        samples_per_batch=D,
        n_batches_train=b,
        n_batches_test=b,
    )
    x_train = training_data["x_train"]
    y_train = training_data["y_train"]
    x_test = training_data["x_test"]
    y_test = training_data["y_test"]

    # train the network
    alg4(
        network=network,
        x_train=x_train,
        y_train=y_train,
        loss_func=loss,
        alpha=alpha,
        n_iter=n_iter,
        num_ints=m,
        dump_to_pickle_file=False,
    )

    # load nn from file
    # with open("nn_dump.pkl", "rb") as f:
    #     network = pickle.load(f)

    # test against test data:
    test_trained_network_sorting(
        network=network, x_test=x_test, y_test=y_test, num_ints=m
    )


if __name__ == "__main__":
    main()
