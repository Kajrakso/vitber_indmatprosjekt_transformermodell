import dill as pickle
from layers import (
    LinearLayer,
    EmbedPosition,
    FeedForward,
    Attention,
    Softmax,
    Relu,
    CrossEntropy,
    Layer,
)
import layers_numba as nl
import numpy as np
from numba import types
from numba.typed.typeddict import Dict


class NeuralNetwork:
    """
    Neural network class that takes a list of layers
    and performs forward and backward pass, as well
    as gradient descent step.
    """

    def __init__(self, layers: list):
        # layers is a list where each element is of the Layer class
        self.layers = layers

    def dump(self, filename: str) -> None:
        if isinstance(self.layers[-1], nl.Softmax):
            self._numba_dump(filename)
        else:
            with open(filename, "wb") as f:
                pickle.dump(("normal", self.layers), f)

    def _numba_dump(self, filename: str) -> None:
        dump_data = []
        for layer in self.layers:
            dump = dump_layer(layer)
            dump_data.append(dump)
            # dump = layer.dump()  # [Error] Can not dump from within the class

        with open(filename, "wb") as f:
            pickle.dump(("numba", dump_data), f)

    def load(self, filename: str):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        if data[0] == "numba":
            self.layers = load_layers(data)
        else:
            self.layers = data[1]

    def forward(self, x):
        # Recursively perform forward pass from initial input x
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        """
        Recursively perform backward pass
        from grad : derivative of the loss wrt
        the final output from the forward pass.
        """

        # reversed yields the layers in reversed order
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def step_gd(self, alpha):
        """
        Perform a gradient descent step for each layer,
        but only if it is of the class LinearLayer.
        """
        for layer in self.layers:
            # Check if layer is of class a class that has parameters
            if not isinstance(
                layer,
                (Softmax, Relu, CrossEntropy, nl.Softmax, nl.Relu, nl.CrossEntropy),
            ):
                layer.step_gd(alpha)
        return

    def step_adam(self, alpha):
        """
        Perform a adam step for each layer,
        but only if it is of the class LinearLayer.
        """
        for layer in self.layers:
            # Check if layer is of class a class that has parameters
            if not isinstance(
                layer,
                (Softmax, Relu, CrossEntropy, nl.Softmax, nl.Relu, nl.CrossEntropy),
            ):
                layer.step_adam(alpha)
        return


def dump_layer(layer: nl.Layer) -> tuple[str, list]:
    if isinstance(
        layer,
        (
            FeedForward,
            EmbedPosition,
            Softmax,
            Relu,
            CrossEntropy,
        ),
    ):
        raise ValueError(
            f"Can not dump normal layers with 'numba_dump'. Argument was of type {type(layer)}."
        )
    if isinstance(layer, (nl.FeedForward)):
        return feedforward_dump(layer)
    if isinstance(layer, (nl.EmbedPosition)):
        return embedpostition_dump(layer)
    if isinstance(layer, (nl.Softmax, nl.Relu, nl.CrossEntropy)):
        return (layer.name, [])
    return generic_dump(layer)


def generic_dump(layer: nl.Layer) -> tuple[str, list]:
    params = convert_params_to_python(layer.params)
    adam_params = convert_adam_params_to_python(layer.adam_params)
    return (layer.name, [params, adam_params])


def feedforward_dump(layer: nl.FeedForward) -> tuple[str, list]:
    l1_dump = generic_dump(layer.l1)
    l2_dump = generic_dump(layer.l2)
    return (layer.name, [l1_dump, l2_dump])


def embedpostition_dump(layer: nl.EmbedPosition) -> tuple[str, list]:
    embed_dump = generic_dump(layer.embed)
    params = convert_params_to_python(layer.params)
    adam_params = convert_adam_params_to_python(layer.adam_params)
    return (layer.name, [params, adam_params, embed_dump])


def convert_params_to_python(
    params: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict[str, np.ndarray]]:
    new_params = dict()
    for key in params:
        W = params[key]["w"]
        d = params[key]["d"]
        new_params[key] = {"w": np.asarray(W), "d": np.asarray(d)}
    return new_params


def convert_params_to_numba(params: dict[str, dict[str, np.ndarray]]) -> types.DictType:
    new_params = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.DictType(types.unicode_type, types.float64[:, :]),
    )
    sub_params = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:, :])
    for key in params:
        W = params[key]["w"]
        d = params[key]["d"]
        sub_params["w"] = W
        sub_params["d"] = d
        new_params[key] = sub_params
    return new_params


def convert_adam_params_to_python(adam_params: dict[str, float]) -> dict[str, float]:
    return {"M": float(adam_params["M"]), "V": float(adam_params["V"])}


def convert_adam_params_to_numba(adam_params: dict[str, float]) -> types.DictType:
    params = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    params["M"] = adam_params["M"]
    params["V"] = adam_params["V"]
    return params


def load_layers(input_data: list[tuple[str, list]]) -> list[nl.Layer]:
    layers = []
    for layer_data in input_data:
        layer_name = layer_data[0]
        if layer_name == "cross-entropy":
            layers.append(nl.CrossEntropy())
        elif layer_name == "relu":
            layers.append(nl.Relu())
        elif layer_name == "softmax":
            layers.append(nl.Softmax())
        elif layer_name == "feed-forward":
            layers.append(feedforward_load(layer_data))
        elif layer_name == "embed-position":
            layers.append(embed_position_load(layer_data))
        elif layer_name == "attention":
            layers.append(generic_load(layer_data, nl.Attention))
        elif layer_name == "linear-layer":
            layers.append(generic_load(layer_data, nl.LinearLayer))
    return layers


def generic_load(
    input_data: tuple[str, list], layer_type: type[nl.Attention | nl.LinearLayer]
) -> nl.Attention | nl.LinearLayer:
    params, adam_params = input_data[1]
    params = convert_params_to_numba(params)
    adam_params = convert_adam_params_to_numba(adam_params)
    layer = layer_type(0, 0)
    layer.load(
        params, adam_params
    )  # Have to convert params and adam_params to numba types
    return layer


def feedforward_load(input_data: tuple[str, list]) -> nl.FeedForward:
    l1_data, l2_data = input_data[1]
    l1 = generic_load(l1_data, nl.LinearLayer)
    l2 = generic_load(l2_data, nl.LinearLayer)
    layer = nl.FeedForward(0, 0)
    layer.load(l1, l2)
    return layer


def embed_position_load(input_data: tuple[str, list]) -> nl.EmbedPosition:
    params, adam_params, embed_data = input_data[1]
    params = convert_params_to_numba(params)
    adam_params = convert_adam_params_to_numba(adam_params)
    layer = nl.EmbedPosition(0, 0, 0)
    embed = generic_load(embed_data, nl.LinearLayer)
    layer.load(params, adam_params, embed)
    return layer
