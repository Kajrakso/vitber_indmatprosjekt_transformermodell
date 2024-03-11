from train_network import init_neural_network
from data_generators import get_train_test_addition
from train_network import train_network
from layers_numba import CrossEntropy
from train_test_params import *
import dill as pickle


def train():
    add_params = AddParams()
    add_params.b_train = 30
    add_params.n_iter = 50_000
    add_params.L = 5

    network = init_neural_network(add_params)

    # prepare training and test data for addition problem
    training_data = get_train_test_addition(
        n_digit = add_params.r,
        samples_per_batch = add_params.D,
        n_batches_train = add_params.b_train,
        n_batches_test=add_params.b_test
    )

    x_train = training_data["x_train"]
    y_train = training_data["y_train"][:, :, add_params.r*2 - 1:]
    x_test = training_data["x_test"]
    y_test = training_data["y_test"][:, :, ::-1]    # remember that (c0, c1, c2) is reversed in the training data.

    loss = CrossEntropy()

    L = train_network(
        network=network,
        x_train=x_train,
        y_train=y_train,
        loss_func=loss,
        alpha=add_params.alpha,
        n_iter=50000,
        num_ints=add_params.m,
        dump_to_pickle_file=True,
        file_name_dump="nn_dump_addition.pkl"
    )

    with open("nn_dump_addition_L.pkl", "wb") as f:
        pickle.dump(L, f)


if __name__ == "__main__":
    train()