from neural_network import NeuralNetwork
from layers_numba import Softmax, FeedForward, Attention, EmbedPosition, LinearLayer
from train_test_params import get_training_params_sort
from data_generators import get_train_test_sorting
from train_network import train_network
from layers_numba import CrossEntropy
from test_network import test_trained_network


def init_neural_network(t_params):
    """t_params: training paramaters. see the functions in `train_test_params.py`."""
    transformer = [
        (
            FeedForward(d=t_params["d"], p=t_params["p"]),
            Attention(d=t_params["d"], k=t_params["k"]),
        )
        for _ in range(t_params["L"])
    ]
    embed_pos = EmbedPosition(n_max=t_params["n_max"], m=t_params["m"], d=t_params["d"])
    un_embed = LinearLayer(input_size=t_params["d"], output_size=t_params["m"])
    softmax = Softmax()

    network = NeuralNetwork(
        [
            embed_pos,
            # don't even try to understand this...
            *[
                t_layer
                for transformer_layer in transformer
                for t_layer in transformer_layer
            ],
            un_embed,
            softmax,
        ]
    )
    return network


sort_params = get_training_params_sort()


training_data = get_train_test_sorting(
    length=sort_params["r"],
    num_ints=sort_params["m"],
    samples_per_batch=sort_params["D"],
    n_batches_train=sort_params["b_train"],
    n_batches_test=sort_params["b_test"],
)

x_train = training_data["x_train"]
y_train = training_data["y_train"]
x_test = training_data["x_test"]
y_test = training_data["y_test"]


network = init_neural_network(sort_params)

network.numba_dump("numba_dump.pkl")

loss = CrossEntropy()

# train_network(
#     network=network,
#     x_train=x_train,
#     y_train=y_train,
#     loss_func=loss,
#     alpha=sort_params["alpha"],
#     n_iter=sort_params["n_iter"],
#     num_ints=sort_params["m"],
#     dump_to_pickle_file=False,
# )
# train_network(
#     network=network,
#     x_train=x_train,
#     y_train=y_train,
#     loss_func=loss,
#     alpha=sort_params["alpha"] / 10,
#     n_iter=sort_params["n_iter"],
#     num_ints=sort_params["m"],
#     dump_to_pickle_file=False,
# )


# test_trained_network(
#     network=network, x_test=x_test, y_test=y_test, num_ints=sort_params["m"]
# )
