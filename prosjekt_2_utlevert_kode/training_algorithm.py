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
    training_data,
    network,
    loss_func,
    alpha,
    n_iter,
    num_ints,
    num_batches,
    batch_size,
) -> None:
    """optimizes the paramaters in the network using the Adam algorithm"""
    x = training_data["x_train"]
    y = training_data["y_train"]

    print("\nstart training...\n")
    print(f"{'iter. step':>15} | {'object function L':>15}")
    print("-" * 40)
    for i in range(n_iter):
        L = np.zeros(batch_size)
        for j in range(b):
            Z = network.forward(onehot(x[j], m=num_ints))
            L[j] = loss_func.forward(y_hat=Z, y=y[j])
            grad_Z = loss.backward()
            _ = network.backward(grad_Z)
            # _ = network.step_gd(alpha)
            _ = network.step_adam(alpha)

        # hopefully L will decrease as we optimize
        print(f"{i+1:>15} | {np.mean(L):>15.10f}")
    print("\nend of training...\n")


if __name__ == "__main__":
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

    training_data = get_train_test_sorting(
        length=r,
        num_ints=m,
        samples_per_batch=D,
        n_batches_train=b,
        n_batches_test=int(b / 2),
    )

    n_iter = 20  # force the training to stop after n_iter steps.

    alg4(
        training_data,
        network,
        loss,
        alpha,
        n_iter,
        num_ints=m,
        num_batches=b,
        batch_size=D,
    )
