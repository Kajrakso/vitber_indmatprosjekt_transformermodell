from neural_network import NeuralNetwork
from layers import (
    LinearLayer,
    CrossEntropy,
    EmbedPosition,
    Softmax,
    FeedForward,
    Attention,
)
from data_generators import get_train_test_sorting
import numpy as np
from utils import onehot


def alg4(
    network: NeuralNetwork,
    x: np.ndarray,
    y: np.ndarray,
    loss_func,
    alpha: float,
    n_iter: int,
    num_ints: int,
) -> None:
    """optimizes the paramaters in the network using the Adam algorithm

    Args:
        network: The BOSS
        x: (num_batches, batch_size, n), input data
        y: (num_batches, batch_size, n), output data
        loss_func
        alpha: alpha paramater in the Adam algorithm (see p. 19, alg. 3)
        n_iter: number of iterations before the training ends
        num_ints: size of vocabulary (m in the project description)
    """
    num_batches = np.shape(x_train)[0]

    print("\nstart training...\n")
    print(f"{'iter. step':>15} | {'object function L':>15}")
    print("-" * 40)
    for i in range(n_iter):
        L = np.zeros(num_batches)
        for j in range(num_batches):
            Z = network.forward(onehot(x[j], m=num_ints))
            L[j] = loss_func.forward(y_hat=Z, y=y[j])
            grad_Z = loss.backward()
            _ = network.backward(grad_Z)
            # _ = network.step_gd(alpha)
            _ = network.step_adam(alpha)

        # hopefully L will decrease as we optimize the params
        print(f"{i+1:>15} | {np.mean(L):>15.10f}")
    print("\nend of training...\n")


def test_trained_network(
        network: NeuralNetwork, x_test: np.ndarray, y_test: np.ndarray, num_ints: int
) -> None:
    """After having trained the network it should be tested on
    data it was not trained on.

    Args:
        network: pre-trained network
        x_test: (num_batches, batch_size, n)
        y_test: (num_batches, batch_size, n)
        num_ints: size of vocabulary (m in the project description)
    """
    batch_size = np.shape(x_test)[1]
    n = np.shape(x_test)[-1]            # length of input x


    print("\nstart testing...\n")
    print(f"{'batch nr.':>15} | {'percentage correct':>15}")
    print("-" * 40)
    for i, x_batch in enumerate(x_test):
        for _ in range(n):
            Z = network.forward(onehot(x_batch, m=num_ints))
            x_batch = np.insert(x_batch, -1, np.argmax(Z, axis=1)[:, -1], axis=1)
        y = x_batch[:, r:]

        num_correct = np.sum(np.all(y_test[i] == y, axis=1))
        print(f"{i+1:>15} | {num_correct/batch_size:.3%}")
    print("\nend of testing...\n")


if __name__ == "__main__":
    #
    # this should probably be done in the notebook for handin
    #

    # alpha parameter for Adam
    alpha = 0.01

    # generate a training set as per
    # exercise 3.3
    # D = 1000  # number of datapoint (x, y)
    # b = 200  # number of batches

    # D, b over tar aaaaalt for lang tid...
    D = 50  # number of datapoint (x, y)
    b = 5  # number of batches

    r = 5  # length of the sequences
    n_max = 2 * r - 1
    m = 2  # number of symbols

    d = 10  # output dimension for the linear layer.
    k = 5
    p = 15
    L = 2

    n_iter = 20  # force the training to stop after n_iter steps.

    # initialize the network and the loss function
    softmax = Softmax()
    # feed_forward = [FeedForward(d=d,p=p) for _ in range(L)]
    feed_forward = FeedForward(d=d, p=p)
    attention = Attention(
        d=d, k=k
    )  # OBS: her har vi byttet rekkefølge på parameterliste i forhold til utdelt kode.
    embed_pos = EmbedPosition(n_max=n_max, m=m, d=d)
    un_embed = LinearLayer(input_size=d, output_size=m)
    softmax = Softmax()

    network = NeuralNetwork([embed_pos, feed_forward, attention, un_embed, softmax])

    loss = CrossEntropy()

    # prepare training and test data for sorting
    training_data = get_train_test_sorting(
        length=r,
        num_ints=m,
        samples_per_batch=D,
        n_batches_train=b,
        n_batches_test=int(b / 2),
    )
    x_train = training_data["x_train"]
    y_train = training_data["y_train"]
    x_test = training_data["x_test"]
    y_test = training_data["y_test"]

    # train the network
    alg4(network, x_train, y_train, loss, alpha, n_iter, m)

    # test against test data:
    test_trained_network(network, x_test, y_test, m)
