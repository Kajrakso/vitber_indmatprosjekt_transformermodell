from layers import *
from neural_network import NeuralNetwork
from utils import onehot
import numpy as np
from data_generators import get_train_test_sorting, get_train_test_addition


# --------------------
# We choose some arbitrary values for the dimensions
b = 6
n_max = 7
m = 8
n = 5

d = 10
k = 5
p = 20

# Create an arbitrary dataset
x = np.random.randint(0, m, (b, n))
y = np.random.randint(0, m, (b, n_max))

# initialize the layers
feed_forward = FeedForward(d=d, p=p)
attention = Attention(
    d=d, k=k
)  # OBS: her har vi byttet rekkefølge på parameterliste i forhold til utdelt kode.
embed_pos = EmbedPosition(n_max=n_max, m=m, d=d)
un_embed = LinearLayer(input_size=d, output_size=m)
softmax = Softmax()

# a manual forward pass
X = onehot(x, m)
assert X.shape == (b, m, n), f"X.shape={X.shape}, expected {(b,m,n)}"
z0 = embed_pos.forward(X)
z1 = feed_forward.forward(z0)
z2 = attention.forward(z1)
z3 = un_embed.forward(z2)
Z = softmax.forward(z3)

# check the shapes
assert z0.shape == (b, d, n), f"z0.shape={z0.shape}, expected {(b,d,n)}"
assert z1.shape == (b, d, n), f"z1.shape={z1.shape}, expected {(b,d,n)}"
assert z2.shape == (b, d, n), f"z2.shape={z2.shape}, expected {(b,d,n)}"
assert z3.shape == (b, m, n), f"z3.shape={z3.shape}, expected {(b,m,n)}"
assert Z.shape == (b, m, n), f"Z.shape={Z.shape}, expected {(b,m,n)}"

# is X one-hot?
assert X.sum() == b * n, f"X.sum()={X.sum()}, expected {b*n}"


assert np.allclose(
    Z.sum(axis=1), 1
), f"Z.sum(axis=1)={Z.sum(axis=1)}, expected {np.ones(b)}"
assert np.abs(Z.sum() - b * n) < 1e-5, f"Z.sum()={Z.sum()}, expected {b*n}"
assert np.all(Z >= 0), f"Z={Z}, expected all entries to be non-negative"

# -----------------------------


# test the forward pass
x = np.random.randint(0, m, (b, n_max))
X = onehot(x, m)

# we test with a y that is shorter than the maximum length
n_y = n_max - 1
y = np.random.randint(0, m, (b, n_y))

# initialize a neural network based on the layers above
network = NeuralNetwork([embed_pos, feed_forward, attention, un_embed, softmax])
# and a loss function
loss = CrossEntropy()

# do a forward pass
Z = network.forward(X)

# compute the loss
L = loss.forward(Z, y)

# get the derivative of the loss wrt Z
grad_Z = loss.backward()

# and perform a backward pass
_ = network.backward(grad_Z)

# and and do a gradient descent step
_ = network.step_gd(0.01)


"""
Here you may add additional tests to for example:

- Check if the ['d'] keys in the parameter dictionaries are not None, or receive something when running backward pass
- Check if the parameters change when you perform a gradient descent step
- Check if the loss decreases when you perform a gradient descent step

This is voluntary, but could be useful.
"""


# check if loss is non-negative
assert L >= 0, f"L={L}, expected L>=0"
assert grad_Z.shape == Z.shape, f"grad_Z.shape={grad_Z.shape}, expected {Z.shape}"

# check if onehot(y) gives zero loss
Y = onehot(y, m)
L = loss.forward(Y, y)
assert L < 1e-5, f"L={L}, expected L<1e-5"
