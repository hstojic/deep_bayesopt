# Copyright 2022 The deep_bayesopt Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import tensorflow as tf
import matplotlib.pyplot as plt

from tests.util.misc import random_seed
from uanets.models.mc_dropout import DropoutNetwork


def _power_function(x: tf.Tensor, error=True) -> tf.Tensor:
    """
    A simple one-dimensional toy problem introduced by <cite data-cite="hernandez2015probabilistic"/>.
    The toy problem is a simple cubic function with some Normally distributed noise around it.
    """
    y = tf.pow(x, 3)
    if error:
        y += tf.random.normal(x.shape, 0, 3, dtype=x.dtype)
    return y


def _plot_predictions(
    x: tf.Tensor, y: tf.Tensor, mu: tf.Tensor, sigma2: tf.Tensor, name: str
) -> None:
    """ plots 2 dimensional objective space with the front"""
    plt.figure(figsize=(7, 5))
    # plot data
    plt.scatter(x, y, marker="x", color="k", alpha=0.5)
    # plot predictions
    # lower = mu - tf.math.sqrt(sigma2)
    # upper = mu + tf.math.sqrt(sigma2)
    # plt.fill_between(tf.squeeze(x), tf.squeeze(lower), tf.squeeze(upper), color="C1", alpha=0.3)
    plt.plot(x, mu, "C1")
    # styles and save
    # plt.set_xlabel("Input")
    # plt.set_ylabel("Output")
    plt.title(f"Data and predictions: {name}")
    plt.savefig(
        f"/tmp/test_predictions__model_predictions_{name}.png", bbox_inches="tight",
    )
    plt.close()



def _plot_training_loss(model: tf.keras.Model, name: str) -> None:
    """ plots 2 dimensional objective space with the front"""
    plt.figure(figsize=(7, 5))
    plt.plot(model.history.history["loss"])
    # plt.set_xlabel("Epoch")
    # plt.set_ylabel("Loss")
    plt.title(f"Training loss: {name}")
    plt.savefig(
        f"/tmp/test_predictions__training_loss_{name}.png",
        bbox_inches="tight",
    )
    plt.close()


@random_seed
def test_mc_dropout_predictions_close_to_actuals(
    max_error: float = 10.,
    rate: float = 0.03,
) -> None:
    num_points = 20
    num_passes = 100

    inputs = tf.random.uniform(shape=[num_points, 1], minval=-4, maxval=4)
    outputs = _power_function(inputs)

    input_tensor_spec = tf.TensorSpec.from_tensor(inputs, name="input")
    output_tensor_spec = tf.TensorSpec([1], outputs.dtype, name="output")

    model = DropoutNetwork(
        input_tensor_spec=input_tensor_spec,
        output_tensor_spec=output_tensor_spec,
        rate=rate,
    )
    model.build(inputs.shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    fit_args = {
        "verbose": 0,
        "epochs": 1000,
        "batch_size": 32,
        "callbacks": [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=100, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.3, patience=20
            )
        ],
    }
    loss = "mse"

    model.compile(
        optimizer,
        loss=[loss],
        metrics=["mse"],
    )
    model.fit(x=inputs, y=outputs, **fit_args)

    stochastic_passes = tf.stack(
        [model(inputs, training=True) for _ in range(num_passes)], axis=0
    )
    predicted_means = tf.math.reduce_mean(stochastic_passes, axis=0)

    predicted_vars = tf.subtract(
        tf.divide(
            tf.reduce_sum(tf.math.multiply(stochastic_passes, stochastic_passes), axis=0),
            num_passes,
        ),
        tf.math.square(predicted_means),
    )
    mean_abs_deviation = tf.reduce_mean(tf.abs(predicted_means - outputs))

    # plotting data, accuracy and training loss
    _plot_predictions(inputs, outputs, predicted_means, predicted_vars, "MC Dropout")
    _plot_training_loss(model, "MC Dropout")

    assert mean_abs_deviation < max_error
