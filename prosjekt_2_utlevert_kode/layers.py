import numpy as np
from utils import onehot


# Test if precomputing the optimal einsum-path has an effect on performance.


class Layer:
    """
    Base class for layers in the neural network with forward and backward pass.
    """

    def __init__(self):
        self.params: dict[str, dict[str, np.ndarray]] = dict()

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


class Attention(Layer):
    def __init__(self, k: int, d: int, initial_scale=0.1):
        """
        Args:
            (k, d): shape of parameter matrices.
        """
        # Initializes the four matrices to something random.
        self.W_K = np.random.randn(k, d) / initial_scale
        self.W_Q = np.random.randn(k, d) / initial_scale

        self.W_V = np.random.randn(k, d) / initial_scale
        self.W_O = np.random.randn(k, d) / initial_scale

        self.params = {
            "W_K": {"w": self.W_K, "d": np.zeros_like(self.W_K)},
            "W_Q": {"w": self.W_Q, "d": np.zeros_like(self.W_Q)},
            "W_V": {"w": self.W_V, "d": np.zeros_like(self.W_V)},
            "W_O": {"w": self.W_O, "d": np.zeros_like(self.W_O)},
        }

        self.softmax = Softmax()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x

        # !TODO: Possible to extract it so we dont compute the same matrix over and over again?
        n = x.shape[2]
        self.B = np.zeros((n, n))
        i1, i2 = np.tril_indices(n, -1)
        self.B[i1, i2] -= np.inf

        # Is used again in the backward pass.
        # !TODO: Check if it is faster than inserting it into an einsum three times
        self.W_QK = self.W_Q.T @ self.W_K  # shape=(d, d)

        # z.T @ W_Q.T @ W_K @ z
        M = np.einsum("bni,nm,bmj->bij", x, self.W_QK, x, out=None)
        # !TODO: Possible to take the B out and just force the lower triangle of A to zero?
        self.A = self.softmax.forward(M + self.B)  # shape=(b, n, n)

        # z = x + W_O.T @ W_V @ x @ A
        return x + np.einsum("ki,km,bmn,bnj->bij", self.W_O, self.W_V, x, self.A)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        b = grad.shape[0]
        g_OV = np.einsum("ki,km,bmj->bij", self.W_V, self.W_O, grad)
        g_S = self.softmax.backward(np.einsum("bki,bkj->bij", self.x, g_OV))

        self.params["W_K"]["d"] = (
            np.einsum("ik,bkn,bnm,bjm->ij", self.W_Q, self.x, g_S, self.x) / b
        )
        self.params["W_Q"]["d"] = (
            np.einsum("ik,bkn,bmn,bjm->ij", self.W_K, self.x, g_S, self.x) / b
        )

        self.params["W_V"]["d"] = (
            np.einsum("ik,bkn,bmn,bjm->ij", self.W_O, grad, self.A, self.x) / b
        )
        self.params["W_O"]["d"] = (
            np.einsum("ik,bkn,bnm,bjm->ij", self.W_V, self.x, self.A, grad) / b
        )

        # Note W_K.T @ W_Q = (W_Q.T @ W_K).T = W_QK.T
        return (
            grad
            + np.einsum("bik,bjk->bij", g_OV, self.A)
            + np.einsum("ki,bkn,bnj->bij", self.W_QK, self.x, g_S)
            + np.einsum("ik,bkn,bjn->bij", self.W_QK, self.x, g_S)
        )

    def step_gd(self, alpha: float) -> None:
        self.softmax.step_gd(alpha)
        super().step_gd(alpha)


class Softmax(Layer):
    epsilon = 1e-8

    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        # To prevent overflow, we use the trick given in the task description.
        self.P = np.exp(x - x.max(axis=1, keepdims=True))
        self.Q = np.sum(self.P, axis=1, keepdims=True)
        self.z = self.P / (self.Q + self.epsilon)
        return self.z

    def backward(self, grad: np.ndarray) -> np.ndarray:
        S = self.P / (self.Q * self.Q + self.epsilon)
        return grad * self.z - np.sum(grad * S, axis=1, keepdims=True) * self.P


class CrossEntropy(Layer):
    epsilon = 1e-8
    def __init__(self, your_arguments_here):
        return

    def forward(self, y_hat: np.ndarray, y:np.ndarray) -> np.ndarray:
        b, m, n = np.shape(y_hat)
        self.b = b
        self.m = m
        self.n = n
        self.y_hot = onehot(y, m)
        self.y_hat = y_hat
        p = np.sum(np.einsum('bij,bij->bij', self.y_hot, y_hat))
        q = -np.log(p)         
        object_func = np.average(q)
        return object_func


    def backward(self) -> np.ndarray:
        return -(self.y_hot / (self.y_hat + self.epsilon)) / self.n
      


class LinearLayer(Layer):
    """
    Linear Layer
    """

    def __init__(self, input_size, output_size, init_scale=0.1):
        """
        Constructor takes input size and output size of layer
        and scale for the weights
        """

        # Initialize weights using a sample from the normal distribution
        # scaled with the init_scale
        self.w = np.random.randn(output_size, input_size) * init_scale
        self.params = {"w": {"w": self.w, "d": np.zeros_like(self.w)}}

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
        y = np.einsum("od,bdn->bon", self.params["w"]["w"], x)
        return y

    def backward(self, grad) -> np.ndarray:
        # """
        # Performs backward pass.

        # Args:
        #     grad: gradient of loss wrt output of layer, shape (batch_size, output_size, n) = (b,o,n)
        # """

        b = grad.shape[0]

        # Compute gradient (average over B batches) of loss wrt weight w:
        # dL/dw = (1/B)*sum_b^B (grad_b@x_b^T)
        self.params["w"]["d"] = np.einsum("bon,bdn->od", grad, self.x) / b

        # Return gradient of loss wrt input of layer
        # dL/dx = w.T@grad
        return np.einsum("od,bon->bdn", self.params["w"]["w"], grad)


class Relu(Layer):
    """
    Relu activation function
    """

    def __init__(self):
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


class EmbedPosition(Layer):
    def __init__(self, n_max, m, d, init_scale=1e-1):
        """
        n_max: maximum length of input sequence
        m: number of items in the vocabulary / number of integers
        d: embedding dimension
        """

        # Initialize a linear layer for the embedding
        self.embed = LinearLayer(m, d, init_scale)
        # Initialize the position embedding matrix
        self.w = np.random.randn(d, n_max) * init_scale

        # Initialize the parameter dictionary for weight with key "Wp"
        self.params = {"Wp": {"w": self.w, "d": np.zeros_like(self.w)}}

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

    def step_gd(self, step_size):

        # We need to call the step_gd method of the linear layer
        self.embed.step_gd(step_size)

        # And since we override step_gd(), we use super
        # which calls the step_gd() of the base class
        # and does gd for the paramters in the params dict
        super().step_gd(step_size)


class FeedForward(Layer):

    def __init__(self, d, p, init_scale=0.1):
        """
        Input:
            d: input dimension of first layer and output of second
            p: output dimension of first and input of second.

        """

        # first linear layer with input size d and output size p
        self.l1 = LinearLayer(d, p, init_scale)

        # We use the Relu activation function
        self.activation = Relu()

        # second linear layer with input size p and output size d
        self.l2 = LinearLayer(p, d, init_scale)

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
