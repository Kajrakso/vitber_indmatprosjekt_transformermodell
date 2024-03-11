import numpy as np
from utils import onehot
from utils_numba import onehot_numba
from numba.experimental import jitclass
from numba import types
import numba as nm


# Test if precomputing the optimal einsum-path has an effect on performance.


class Layer:
    """
    Base class for layers in the neural network with forward and backward pass.
    """

    def __init__(self):
        self.params: dict[str, dict[str, np.ndarray]] = dict()
        self.adam_params: dict[str, float] = {
            "M": 0,
            "V": 0,
        }
        self.epsilon = 1e-8
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.name = "Layer"

    def load(self) -> None:
        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs a forward pass of the layer.
        It also stores variables which will be used later in the backward pass.

        Args:
            x (np.ndarray[b, d, n]): input matrix

        Returns:
            np.ndarray[b, d, n]: the result of the forward pass
        """
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Performs a backward pass of the layer
        Computes the gradient of the loss function wrt the parameter matrices by averaging over a batch.
        When computing wrt the input, it is done for each batch separately.

        Args:
            grad (np.ndarray[b, d, n]): Gradient of loss wrt to the previous layer. [g_{i+1}]

        Returns:
            np.ndarray[b, d, n]: Gradient of loss wrt to the input matrix. [g_i]
        """
        raise NotImplementedError

    def step_gd(self, alpha: float) -> None:
        """
        Performs a gradient descent step given learning rate. The parameter matrices
        are updated in-place.
        Assumes that the layer has a parameter dictionary "params" on the form

        params = {
            'w1': {
                'w': w,         The parameter matrix
                'd': d,         The gradient of loss wrt the parameter matrix
                },
            'w2': {....},
        }
        where each parameter has a key 'w' for weights and 'd' for gradients.
        """
        for param in self.params:
            self.params[param]["w"] -= alpha * self.params[param]["d"]

    def step_adam(self, alpha: float):
        """
        Performs a gradient descent step given learning rate. The parameter matrices
        are updated in-place.
        Assumes that the layer has a parameter dictionary "params" on the form

        params = {
            'w1': {
                'w': w,         The parameter matrix
                'd': d,         The gradient of loss wrt the parameter matrix
                },
            'w2': {....},
        }
        where each parameter has a key 'w' for weights and 'd' for gradients.
        """
        for param in self.params.values():
            G = param["d"]
            M = self.beta_1 * self.adam_params["M"] + (1 - self.beta_1) * G
            V = self.beta_2 * self.adam_params["V"] + (1 - self.beta_2) * G**2
            M_hat = M / (1 - self.beta_1)
            V_hat = V / (1 - self.beta_2)
            param["w"] -= alpha * (M_hat / (np.sqrt(V_hat) + self.epsilon))


# Can not subclass when using numba

softmax_specs = [
    ("epsilon", types.float64),
    ("beta_1", types.float64),
    ("beta_2", types.float64),
    ("P", types.float64[:, :, :]),
    ("Q", types.float64[:, :, :]),
    ("z", types.float64[:, :, :]),
    ("name", types.unicode_type),
]


@jitclass(softmax_specs)
class Softmax(Layer):
    def __init__(self):
        self.epsilon = 1e-8
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.name = "softmax"

    def forward(self, x: np.ndarray) -> np.ndarray:
        # #!TODO: Skrive om softmax (axis) til å være kompatibel med numba
        # To prevent overflow, we use the trick given in the task description.
        b, d, n = x.shape
        max_x = np.zeros_like(x)
        for i in range(b):
            for col in range(n):
                max_x[i, :, col] = np.max(x[i, :, col])

        self.P = np.exp(x - max_x)

        self.Q = np.zeros_like(x)
        for i in range(b):
            for col in range(n):
                self.Q[i, :, col] = np.sum(self.P[i, :, col])
        self.z = self.P / (self.Q + self.epsilon)
        return self.z

    def backward(self, grad: np.ndarray) -> np.ndarray:
        S = self.P / (self.Q**2 + self.epsilon)
        gS = grad * S
        sum_gS = np.zeros_like(grad)
        b, d, n = grad.shape
        for i in range(b):
            for col in range(n):
                sum_gS[i, :, col] = np.sum(gS[i, :, col])
        return grad * self.z - sum_gS * self.P


attention_specs = [
    ("W_K", types.float64[:, :]),
    ("W_Q", types.float64[:, :]),
    ("W_O", types.float64[:, :]),
    ("W_V", types.float64[:, :]),
    (
        "params",
        types.DictType(
            types.unicode_type, types.DictType(types.unicode_type, types.float64[:, :])
        ),
    ),
    ("softmax", nm.typeof(Softmax())),
    ("adam_params", types.DictType(types.unicode_type, types.float64)),
    ("epsilon", types.float64),
    ("beta_1", types.float64),
    ("beta_2", types.float64),
    ("B", types.float64[:, :]),
    ("W_QK", types.float64[:, :]),
    ("x", types.float64[:, :, :]),
    ("A", types.float64[:, :, :]),
    ("name", types.unicode_type),
]


@jitclass(attention_specs)
class Attention(Layer):
    def __init__(self, k: int, d: int, initial_scale=0.1):
        """
        Args:
            (k, d): shape of parameter matrices.
        """
        self.name = "attention"
        self.adam_params = {
            "M": 0.0,
            "V": 0.0,
        }
        self.W_K = np.random.randn(k, d) * initial_scale
        self.W_Q = np.random.randn(k, d) * initial_scale
        self.W_V = np.random.randn(k, d) * initial_scale
        self.W_O = np.random.randn(k, d) * initial_scale

        self.params = {
            "W_K": {"w": self.W_K, "d": np.zeros_like(self.W_K)},
            "W_Q": {"w": self.W_Q, "d": np.zeros_like(self.W_Q)},
            "W_V": {"w": self.W_V, "d": np.zeros_like(self.W_V)},
            "W_O": {"w": self.W_O, "d": np.zeros_like(self.W_O)},
        }

        self.softmax = Softmax()

        self.epsilon = 1e-8
        self.beta_1 = 0.9
        self.beta_2 = 0.999

    def load(
        self, params: dict[str, dict[str, np.ndarray]], adam_params: dict[str, float]
    ) -> None:
        self.params = params
        self.adam_params
        self.W_K = self.params["W_K"]["w"]
        self.W_Q = self.params["W_Q"]["w"]
        self.W_V = self.params["W_V"]["w"]
        self.W_O = self.params["W_O"]["w"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x

        batch_size, _, n = x.shape
        self.B = np.zeros((n, n))
        # i1, i2 = np.tril_indices(n, -1)
        # self.B[i1, i2] -= np.inf
        for i in range(n):
            for j in range(i):
                self.B[i, j] = -np.inf

        # Is used again in the backward pass.
        self.W_QK = self.W_Q.T @ self.W_K  # shape=(d, d)

        # z.T @ W_Q.T @ W_K @ z
        M = np.zeros((batch_size, n, n))  # shape[b, n, n]
        for i in range(batch_size):
            M[i] = self.x[i].T @ self.W_QK @ self.x[i]

        self.A = self.softmax.forward(M + self.B)  # shape=(b, n, n)

        # z = x + W_O.T @ W_V @ x @ A
        z_temp = np.zeros_like(x)
        for i in range(batch_size):
            z_temp[i] = self.W_O.T @ self.W_V @ x[i] @ self.A[i]

        return x + z_temp

    def backward(self, grad: np.ndarray) -> np.ndarray:
        b, d, n = grad.shape

        g_OV = np.zeros_like(grad)  # shape[b, d, n]
        for i in range(b):
            g_OV[i] = self.W_V.T @ self.W_O @ grad[i]

        # shape[b, n, n]
        g_S_temp = np.zeros((b, n, n))
        for i in range(b):
            g_S_temp[i] = self.x[i].T @ g_OV[i]
        g_S = self.softmax.backward(g_S_temp)

        self.params["W_K"]["d"] = np.zeros_like(self.W_K)
        self.params["W_Q"]["d"] = np.zeros_like(self.W_Q)
        self.params["W_O"]["d"] = np.zeros_like(self.W_O)
        self.params["W_V"]["d"] = np.zeros_like(self.W_V)

        result1, result2, result3 = (
            np.zeros_like(grad),
            np.zeros_like(grad),
            np.zeros_like(grad),
        )

        for i in range(b):
            # dL/dW
            self.params["W_K"]["d"] += (self.W_Q @ self.x[i] @ g_S[i] @ self.x[i].T) / b

            self.params["W_Q"]["d"] += (
                self.W_K @ self.x[i] @ g_S[i].T @ self.x[i].T
            ) / b

            self.params["W_O"]["d"] += (
                self.W_V @ self.x[i] @ self.A[i] @ grad[i].T
            ) / b

            self.params["W_V"]["d"] += (
                self.W_O @ grad[i] @ self.A[i].T @ self.x[i].T
            ) / b

            # g_{l+1}
            # Note W_K.T @ W_Q = (W_Q.T @ W_K).T = W_QK.T
            result1[i] = g_OV[i] @ self.A[i].T
            result2[i] = self.W_QK.T @ self.x[i] @ g_S[i]
            result3[i] = self.W_QK @ self.x[i] @ g_S[i].T

        return grad + result1 + result2 + result3


cross_entropy_specs = [
    ("n", types.int64),
    ("epsilon", types.float64),
    ("y_hot", types.float64[:, :, :]),
    ("y_hat", types.float64[:, :, :]),
    ("name", types.unicode_type),
]


@jitclass(cross_entropy_specs)
class CrossEntropy:
    def __init__(self):
        self.name = "cross-entropy"
        self.epsilon = 1e-8

    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """forward step for one batch

        Args:
            y_hat (np.ndarray): the prediction matrix from the transformer model, dim (b, m, n)
            y (np.ndarray): array of the correct solutions, dim (b, n)

        Returns:
            float: the average loss of the entire batch
        """
        b, m, n = np.shape(y_hat)
        self.n = n
        self.y_hot = onehot_numba(y, m)
        self.y_hat = y_hat

        p = np.zeros((b, self.n))
        for i in range(b):
            p[i] = np.sum(self.y_hot[i] * self.y_hat[i], axis=0)
        q = -np.log(p)
        return np.average(q)

    def backward(self) -> np.ndarray:
        """backward step for cross entropy

        Returns:
            np.ndarray: gradient wrt the prediciton from the transformer model
        """
        return -(self.y_hot / (self.y_hat + self.epsilon)) / self.n


linear_specs = [
    ("w", types.float64[:, :]),
    (
        "params",
        types.DictType(
            types.unicode_type, types.DictType(types.unicode_type, types.float64[:, :])
        ),
    ),
    ("adam_params", types.DictType(types.unicode_type, types.float64)),
    ("epsilon", types.float64),
    ("beta_1", types.float64),
    ("beta_2", types.float64),
    ("x", types.float64[:, :, :]),
    ("name", types.unicode_type),
]


@jitclass(linear_specs)
class LinearLayer(Layer):
    """
    Linear Layer
    """

    def __init__(self, input_size, output_size, init_scale=0.1):
        """
        Constructor takes input size and output size of layer
        and scale for the weights
        """
        self.name = "linear-layer"
        self.adam_params = {
            "M": 0.0,
            "V": 0.0,
        }
        # Initializes the four matrices to something random.
        self.w = np.random.randn(output_size, input_size) * init_scale
        self.params = {"w": {"w": self.w, "d": np.zeros_like(self.w)}}

        self.epsilon = 1e-8
        self.beta_1 = 0.9
        self.beta_2 = 0.999

        # Initialize weights using a sample from the normal distribution
        # scaled with the init_scale

    def load(self, params, adam_params) -> None:
        self.params = params
        self.adam_params = adam_params
        self.w = self.params["w"]["w"]

    def forward(self, x) -> np.ndarray:
        """
        Computes the affine transformation of the forward pass
        Stores input for backwards pass and returns output y = Wx.

        Args:
            x: array of shape (batch_size, input_size, n) = (b,d,n)

        Returns:
            y: array of shape (batch_size, output_size, n) = (b,o,n)
        """
        self.x = x

        # Return output of layer
        # y = w@x
        b, d, n = x.shape
        k, d = self.w.shape
        y = np.zeros((b, k, n))
        for i in range(b):
            y[i] = self.params["w"]["w"] @ x[i]
        return y

    def backward(self, grad) -> np.ndarray:
        # """
        # Performs backward pass.

        # Args:
        #     grad: gradient of loss wrt output of layer, shape (batch_size, output_size, n) = (b,o,n)
        # """

        b, d, n = grad.shape
        k, d = self.w.shape

        # Compute gradient (average over B batches) of loss wrt weight w:
        # dL/dw = (1/B)*sum_b^B (grad_b@x_b^T)
        self.params["w"]["d"] = np.zeros_like(self.w)
        for i in range(b):
            self.params["w"]["d"] += (grad[i] @ self.x[i].T) / b

        # Return gradient of loss wrt input of layer
        # dL/dx = w.T@grad
        res = np.zeros((b, d, n))
        for i in range(b):
            res[i] = self.params["w"]["w"].T @ grad[i]
        return res


relu_specs = [
    ("x", types.float64[:, :, :]),
    ("name", types.unicode_type),
]


@jitclass(relu_specs)
class Relu:
    """
    Relu activation function
    """

    def __init__(self):
        self.name = "relu"
        return

    def relu(self, x):
        # relu(x) = max(0,x)
        return np.maximum(np.zeros(x.shape), x)

    def forward(self, x):
        # Store input for backwards pass
        self.x = x
        return self.relu(x)

    def backward(self, grad):
        # dL/dx = grad * relu'(x)
        return grad * np.where(self.x > 0, np.ones_like(self.x), np.zeros_like(self.x))


embed_specs = [
    ("w", types.float64[:, :]),
    (
        "params",
        types.DictType(
            types.unicode_type, types.DictType(types.unicode_type, types.float64[:, :])
        ),
    ),
    ("embed", nm.typeof(LinearLayer(4, 4))),
    ("adam_params", types.DictType(types.unicode_type, types.float64)),
    ("epsilon", types.float64),
    ("beta_1", types.float64),
    ("beta_2", types.float64),
    ("name", types.unicode_type),
]


@jitclass(embed_specs)
class EmbedPosition(Layer):
    def __init__(self, n_max, m, d, init_scale=1e-1):
        """
        n_max: maximum length of input sequence
        m: number of items in the vocabulary / number of integers
        d: embedding dimension
        """
        self.name = "embed-position"
        self.adam_params = {
            "M": 0.0,
            "V": 0.0,
        }
        # Initializes the four matrices to something random.
        # Initialize a linear layer for the embedding
        self.w = np.random.randn(d, n_max) * init_scale
        # Initialize the position embedding matrix
        self.embed = LinearLayer(m, d, init_scale)
        self.params = {"Wp": {"w": self.w, "d": np.zeros_like(self.w)}}
        self.epsilon = 1e-8
        self.beta_1 = 0.9
        self.beta_2 = 0.999

        # Initialize the parameter dictionary for weight with key "Wp"

    def load(
        self,
        params: dict[str, dict[str, np.ndarray]],
        adam_params: dict[str, float],
        embed: LinearLayer,
    ) -> None:
        self.params = params
        self.w = self.params["Wp"]["w"]
        self.adam_params = adam_params
        self.embed = embed

    def forward(self, X):
        """
        Input:
            X: one-hot encoded array of shape (b,m,n).

        Output:
            z_0: array of shape (b,d,n)

        embed.forward(X) maps (b,m,n) to (b,d,n).
        Assigns a column of size d to each integer in the sequence
        and add positional embedding matrix (params['Wp']['w'][:,:n]) (b,d,n).

        Equivalent to

        z_0 = W_E@X + W_P[:,:n]

        """

        # We assume that n < n_max
        n = X.shape[-1]
        z_0 = self.embed.forward(X) + self.params["Wp"]["w"][:, :n]
        return z_0

    def backward(self, grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - None
        """

        b = grad.shape[0]

        # Compute gradient (average over B batches) of loss wrt positional embedding w:
        self.params["Wp"]["d"] = np.zeros_like(self.w)
        self.params["Wp"]["d"] += np.sum(grad, axis=0) / b

        # Use backwards pass of the linear layer
        self.embed.backward(grad)

        # This is always the final layer, so we return None
        return None


feedforward_specs = [
    ("x", types.float64[:, :, :]),
    ("l1", nm.typeof(LinearLayer(4, 4))),
    ("l2", nm.typeof(LinearLayer(4, 4))),
    ("activation", nm.typeof(Relu())),
    ("name", types.unicode_type),
]


@jitclass(feedforward_specs)
class FeedForward(Layer):
    def __init__(self, d, p, init_scale=0.1):
        """
        Input:
            d: input dimension of first layer and output of second
            p: output dimension of first and input of second.

        """
        self.name = "feed-forward"

        # Initializes the four matrices to something random.
        # first linear layer with input size d and output size p
        self.l1 = LinearLayer(d, p, init_scale)
        # second linear layer with input size p and output size d
        self.l2 = LinearLayer(p, d, init_scale)
        # We use the Relu activation function
        self.activation = Relu()

    def load(self, l1: LinearLayer, l2: LinearLayer) -> None:
        self.l1 = l1
        self.l2 = l2

    def forward(self, x):
        """
        Input:
            - x of shape (b,d,n)
        Output:
            - shape (b,d,n)

        This is equivalent to
        y = x + W2.T@Relu(W1@x)

         (W1,W2 are p x d)
        """

        self.x = x

        return x + self.l2.forward(self.activation.forward(self.l1.forward(x)))

    def backward(self, grad) -> np.ndarray:
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - derivative of loss wrt input x. Shape (b,d,n)

        """

        # We use backward pass of the linear layers and activation.
        # Recall that the backward pass reverse the order of the layers.
        grad_feed_forward = self.l1.backward(
            self.activation.backward(self.l2.backward(grad))
        )

        # Since forward pass is x + W2.T@Relu(W1@x)
        return grad + grad_feed_forward

    def step_gd(self, step_size):
        # Call the step_gd method of the linear layers
        self.l1.step_gd(step_size)
        self.l2.step_gd(step_size)

    def step_adam(self, alpha: float):
        self.l1.step_adam(alpha)
        self.l2.step_adam(alpha)
