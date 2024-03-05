from neural_network import NeuralNetwork
from layers import LinearLayer, CrossEntropy
from data_generators import get_train_test_sorting


def alg4(training_data, network, loss_func, alpha, beta1, beta2, n_iter) -> None:
    """optimizes the paramaters in the network using the Adam algorithm"""
    for i in range(n_iter):
        x = training_data['x_train']
        y = training_data['y_train']
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

    # initialize the network and the loss function
    network = NeuralNetwork([LinearLayer])
    loss = CrossEntropy()

    # generate a training set as per
    # exercise 3.3
    D = 1000  # number of datapoint (x, y)
    b = 200  # number of batches

    r = 5  # length of the sequences
    m = 2  # number of symbols

    d = 10
    k = 5
    p = 15
    L = 2


    training_data = get_train_test_sorting(
        length=r,
        num_ints=m,
        samples_per_batch=D,
        n_batches_train=b,
        n_batches_test=b / 2,
    )

    n_iter = 100        # force the training to stop after n_iter steps.

    alg4(training_data, network, loss, alpha, beta1, beta2, n_iter)