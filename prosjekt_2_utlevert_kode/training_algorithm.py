from neural_network import NeuralNetwork
from layers import LinearLayer, CrossEntropy, EmbedPosition, Softmax
from data_generators import get_train_test_sorting
import numpy as np

def alg4(training_data, network, loss_func, alpha, beta1, beta2, n_iter) -> None:
    """optimizes the paramaters in the network using the Adam algorithm"""
    x = training_data['x_train']
    y = training_data['y_train']
    
    for i in range(n_iter):
        # print(np.shape(x))
        Z = network.forward(x)
        L = loss_func.forward(Z, y)
        
        # hopefully L will decrease as we optimize
        print(L)

        grad_Z = loss.backward()
        _ = network.backward(grad_Z)
        _ = network.step_gd(alpha)



if __name__ == "__main__":
    # parameters for Adam
    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999

    # generate a training set as per
    # exercise 3.3
    D = 1000  # number of datapoint (x, y)
    b = 200  # number of batches

    r = 5  # length of the sequences
    m = 2  # number of symbols

    d = 10 # output dimension for the linear layer.
    k = 5
    p = 15
    L = 2

    # initialize the network and the loss function
    embed_pos = EmbedPosition(r,m,d)
    un_embed = LinearLayer(d,m)
    softmax = Softmax()

    network = NeuralNetwork([embed_pos, un_embed, softmax])

    loss = CrossEntropy()

    training_data = get_train_test_sorting(
        length=r,
        num_ints=m,
        samples_per_batch=D,
        n_batches_train=b,
        n_batches_test=int(b / 2),
    )

    n_iter = 100        # force the training to stop after n_iter steps.

    alg4(training_data, network, loss, alpha, beta1, beta2, n_iter)