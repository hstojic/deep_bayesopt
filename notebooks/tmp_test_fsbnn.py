# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from uanets.models.layers import UnitNormalLayer, DenseStochasticLayer, GaussianLikelihoodLayer
from uanets.models.function_space_bnn import FunctionSpaceBNN

# %matplotlib notebook


# %%
def _f(x):
    return np.sqrt(x) * np.sin(2 * x) - 0.5


# %%
x = np.vstack(
    (
        0.2 * np.random.default_rng(4).uniform(size=(100, 1)),
        0.2 * np.random.default_rng(4).uniform(size=(100, 1)) + 0.8,
    )
)
y = _f(x) + 0.05 * np.random.default_rng(4).normal(size=x.shape)
x -= 0.5

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(x, y, s=2)

# %%
X = tf.convert_to_tensor(x, dtype=tf.float32)
Y = tf.convert_to_tensor(y, dtype=tf.float32)

# %%
prior = tf.keras.Sequential(layers=[UnitNormalLayer(units=1)])

# %%
posterior = tf.keras.Sequential(
    layers=[
        DenseStochasticLayer(2**10, activation="swish"),
        tf.keras.layers.Dense(1),
    ]
)

# %%
mean_function = tf.keras.Sequential(layers=[tf.keras.layers.Dense(1, kernel_initializer="ones")])

# %%
likelihood = tf.keras.Sequential(layers=[GaussianLikelihoodLayer(0.1)])

# %%
model = FunctionSpaceBNN(
    prior=prior,
    posterior=posterior,
    likelihood=likelihood,
    mean_function=mean_function,
    num_data=0,  # X.shape[0],  test prior reversion
)

# %%
model.compile("adam")

# %%
model.fit(X, Y, batch_size=20, epochs=500, shuffle=True)

# %%
x_test = tf.linspace(-2, 2, 401)[:, None]
means = tf.stack([model(x_test).mean() for _ in range(100)], axis=0)
# means = tf.stack([posterior(x_test) for _ in range(100)], axis=0)

# %%
mean = tf.reduce_mean(means, axis=0)
std = tf.math.reduce_std(means, axis=0)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(x, y, s=2)
(l1,) = ax.plot(x_test, mean)
ax.fill_between(
    x_test[:, 0],
    mean[:, 0] - 1.96 * std[:, 0],
    mean[:, 0] + 1.96 * std[:, 0],
    color=l1.get_color(),
    alpha=0.2,
)

# %%
