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
        """Dump the network to a file so it can be reconstructed later

        NOTE:
        The network has to be a pure numba-network or a pure normal-network in order
        to be dumped.

        Args:
            filename (str): file to dump the network to
        """
        if isinstance(self.layers[-1], nl.Softmax):
            # If the network consists of numba-layers, the information have to be converted to
            # native python types in order to be serialized and dumped.
            # This function is relatively slow, because it essentially creates a copy of all the data.
            self._numba_dump(filename)
        else:
            # If the network just consists of normal layers, then the network can be dumped as is
            # by dumping the self.layers variable.
            with open(filename, "wb") as f:
                # ("normal", *) to indicate for loaders that this is a normal network
                pickle.dump(("normal", self.layers), f)

    def _numba_dump(self, filename: str) -> None:
        """Dump a network based on only numba-layers

        Args:
            filename (str): file to dump the network to
        """
        dump_data = []
        for layer in self.layers:
            # Create dump of each layer separately, then append to a list
            # which will be dumped at the end.
            dump = dump_layer(layer)
            dump_data.append(dump)

        with open(filename, "wb") as f:
            # ("numba", *) to indicate for loaders that this is a numba-network
            pickle.dump(("numba", dump_data), f)

    def load(self, filename: str):
        """Load a network from pickle-file

        Args:
            filename (str): file to load network from
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        # Check if the loaded network uses numba-layers or not
        if data[0] == "numba":
            self.layers = load_numba_layers(data[1])
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


# -------------------------------
# Dumping logic for numba-layers
# -------------------------------


def dump_layer(layer: nl.Layer) -> tuple[str, list]:
    """Create a dump of layer

    Args:
        layer (nl.Layer): The layer to be dumped

    Raises:
        ValueError: If the network contains normal layers, the program will fail.

    Returns:
        tuple[str, list]: A tuple containing the name of the network and a list of
        data to be able to reconstruct the layer when loaded later on.
    """
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
    """Fallback dumping logic for numba-layers.

    Args:
        layer (nl.Layer): Layer to be dumped

    Returns:
        tuple[str, list]: A tuple containing the name of the network and a list
        containing the params-dictionary and the adam_params-dictionary.
    """
    # Note: Each conversion has to make a copy of the data, and is
    # therefore quite expensive.
    params = convert_params_to_python(layer.params)
    return (layer.name, [params])


def feedforward_dump(layer: nl.FeedForward) -> tuple[str, list]:
    """Dump a FeedForward-layer

    Args:
        layer (nl.FeedForward): The layer to be dumped

    Returns:
        tuple[str, list]: A tuple containing the name of the layer, feedforward,
        and a list with the dumps of the two sublayers.
    """
    l1_dump = generic_dump(layer.l1)
    l2_dump = generic_dump(layer.l2)
    return (layer.name, [l1_dump, l2_dump])


def embedpostition_dump(layer: nl.EmbedPosition) -> tuple[str, list]:
    """Dump an EmbedPosition-layer

    Args:
        layer (nl.EmbedPosition): The layer to be dumped

    Returns:
        tuple[str, list]: A tuple containing the name of the layer, embed-position,
        and a list with the params- and adam_params-dictionaries and the dump
        of the embed-sublayer.
    """
    embed_dump = generic_dump(layer.embed)
    params = convert_params_to_python(layer.params)
    return (layer.name, [params, embed_dump])


def convert_params_to_python(
    params: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict[str, np.ndarray]]:
    """Convert the params-dictionary from numba-types to
    native python, so it can be serialized.
    Creates a copy of all the data.

    Args:
        params (dict[str, dict[str, np.ndarray]]): The parameter-dict to convert.

    Returns:
        dict[str, dict[str, np.ndarray]]: A copied and converted dictionary
    """
    new_params = dict()
    for key in params:
        W = params[key]["w"]
        d = params[key]["d"]
        M = params[key]["M"]
        V = params[key]["V"]
        new_params[key] = {
            "w": np.asarray(W),
            "d": np.asarray(d),
            "M": np.asarray(M),
            "V": np.asarray(V),
        }
    return new_params


def convert_params_to_numba(params: dict[str, dict[str, np.ndarray]]) -> types.DictType:
    """Convert parameter-dict from a pure python type to a numba supported type.

    Args:
        params (dict[str, dict[str, np.ndarray]]): Params to convert

    Returns:
        types.DictType: The numba-compatible version of the dict
    """
    new_params = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.DictType(types.unicode_type, types.float64[:, :]),
    )
    for key in params:
        sub_params = Dict.empty(
            key_type=types.unicode_type, value_type=types.float64[:, :]
        )
        W = params[key]["w"]
        d = params[key]["d"]
        M = params[key]["M"]
        V = params[key]["V"]
        sub_params["w"] = W
        sub_params["d"] = d
        sub_params["M"] = M
        sub_params["V"] = V

        new_params[key] = sub_params
    return new_params


def load_numba_layers(input_data: list[tuple[str, list]]) -> list[nl.Layer]:
    """Load numba-layers from a dump.

    Args:
        input_data (list[tuple[str, list]]): The dump to load from

    Returns:
        list[nl.Layer]: A list of the loaded layers in the correct order.
    """
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
    """Fallback load function for loading a layer from a dump

    Args:
        input_data (tuple[str, list]): The data to load from
        layer_type (type[nl.Attention  |  nl.LinearLayer]): The type of layer to create from the dump

    Returns:
        nl.Attention | nl.LinearLayer: The reconstructed layer
    """
    params, *_ = input_data[1]
    params = convert_params_to_numba(params)
    layer = layer_type(0, 0)
    layer.load(params)
    return layer


def feedforward_load(input_data: tuple[str, list]) -> nl.FeedForward:
    """Load function for FeedForward-layer

    Args:
        input_data (tuple[str, list]): The data to load from

    Returns:
        nl.FeedForward: The reconstructed network
    """
    l1_data, l2_data = input_data[1]
    l1 = generic_load(l1_data, nl.LinearLayer)
    l2 = generic_load(l2_data, nl.LinearLayer)
    layer = nl.FeedForward(0, 0)
    layer.load(l1, l2)
    return layer


def embed_position_load(input_data: tuple[str, list]) -> nl.EmbedPosition:
    """Load function for EmbedPosition-layer

    Args:
        input_data (tuple[str, list]): The data to load from

    Returns:
        nl.FeedForward: The reconstructed network
    """
    params, embed_data = input_data[1]
    params = convert_params_to_numba(params)
    layer = nl.EmbedPosition(0, 0, 0)
    embed = generic_load(embed_data, nl.LinearLayer)
    layer.load(params, embed)
    return layer
