from train_test_params import *
from data_generators import get_train_test_sorting
from train_network import init_neural_network
from train_network import train_network
from layers_numba import CrossEntropy
from test_network import test_trained_network


sort_params = SortParams2()

training_data = get_train_test_sorting(
    length=sort_params.r,
    num_ints=sort_params.m,
    samples_per_batch=sort_params.D,
    n_batches_train=sort_params.b_train,
    n_batches_test=sort_params.b_test,
)

x_train = training_data["x_train"]
y_train = training_data["y_train"][:, :, sort_params.r - 1 :]
x_test = training_data["x_test"]
y_test = training_data["y_test"]


network = init_neural_network(sort_params)


loss = CrossEntropy()

train_network(
    network=network,
    x_train=x_train,
    y_train=y_train,
    loss_func=loss,
    alpha=sort_params.alpha,
    n_iter=sort_params.n_iter,
    num_ints=sort_params.m,
    dump_to_pickle_file=False,
)


test_trained_network(
    network=network, x_test=x_test, y_test=y_test, num_ints=sort_params.m
)
