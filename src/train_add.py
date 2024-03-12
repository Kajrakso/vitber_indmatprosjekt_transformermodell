from test_network import test_trained_network
from train_test_params import *
from train_network import init_neural_network
from data_generators import get_train_test_addition
from train_network import train_network
from layers_numba import CrossEntropy


DUMP_FILENAME = "nn_dump_addition.pkl"


def load_from_pkl_and_gen_text(filename: str, params) -> None:
    net = init_neural_network(params)
    net.load(filename)

    test_trained_network(
        network=net, x_test=x_test, y_test=y_test, num_ints=params.m
    )


if __name__ == "__main__":
    add_params = AddParams()

    add_params.n_iter = 10_000

    network = init_neural_network(add_params)

    # prepare training and test data for addition problem
    training_data = get_train_test_addition(
        n_digit=add_params.r,
        samples_per_batch=add_params.D,
        n_batches_train=add_params.b_train,
        n_batches_test=add_params.b_test,
    )

    x_train = training_data["x_train"]
    y_train = training_data["y_train"][:, :, add_params.r * 2 - 1 :]
    x_test = training_data["x_test"]
    y_test = training_data["y_test"][:, :, ::-1]

    loss = CrossEntropy()

    L = train_network(
        network=network,
        x_train=x_train,
        y_train=y_train,
        loss_func=loss,
        alpha=add_params.alpha,
        n_iter=add_params.n_iter,
        num_ints=add_params.m,
        dump_to_pickle_file=True,
        file_name_dump=DUMP_FILENAME,
    )

    # test_trained_network( network=network, x_test=x_test, y_test=y_test, num_ints=add_params.m)
    # load_from_pkl_and_gen_text(DUMP_FILENAME, add_params)