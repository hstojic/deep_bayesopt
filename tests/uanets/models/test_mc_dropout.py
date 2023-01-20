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

from typing import Any, List, Union

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from tests.util.misc import inputs_outputs_spec
from uanets.models import MonteCarloDropout


@pytest.fixture(name="num_hidden_layers", params=[0, 1, 3])
def _num_hidden_layers_fixture(request: Any) -> int:
    return request.param


@pytest.fixture(name="input_shape", params=[[1], [5]])
def _input_shape_fixture(request: Any) -> List[int]:
    return request.param


@pytest.fixture(name="output_shape", params=[[1], [2]])
def _output_shape_fixture(request: Any) -> List[int]:
    return request.param


@pytest.mark.parametrize("num_hidden_layers, rate", [(1, 0.3), (3, 0.7), (5, 0.9)])
@pytest.mark.parametrize("units", [10, 50])
@pytest.mark.parametrize("activation", ["relu", tf.keras.activations.tanh])
def test_dropout_network_build_seems_correct(
    input_shape: List[int],
    output_shape: List[int],
    num_hidden_layers: int,
    units: int,
    activation: Union[str, tf.keras.layers.Activation],
    rate: float,
) -> None:
    """Tests the correct consturction of dropout network architectures"""

    inputs, outputs = inputs_outputs_spec(input_shape, output_shape)
    hidden_layer_args = [
        {"units": units, "activation": activation} for _ in range(num_hidden_layers)
    ]

    model = MonteCarloDropout(inputs, outputs, hidden_layer_args, rate)

    if not isinstance(rate, list):
        rate_all_layers = [rate for _ in range(num_hidden_layers + 1)]

    # basics
    assert isinstance(model, tf.keras.Model)

    # check the model has not been compiled
    assert model.compiled_loss is None
    assert model.compiled_metrics is None
    assert model.optimizer is None

    # check the number of layers is correct and they are properly constructed
    assert len(model.layers) == 2
    assert len(model.layers[0].layers) == num_hidden_layers * 2
    assert len(model.layers[1].layers) == 2

    for i, layer in enumerate(model.layers[0].layers):
        if i % 2 == 0:
            isinstance(layer, tf.keras.layers.Dropout)
            layer.rate == rate_all_layers[int(i / 2)]
        elif i % 2 == 1:
            isinstance(layer, tf.keras.layers.Dense)
            assert layer.units == units
            assert layer.activation == activation or layer.activation.__name__ == activation

    assert isinstance(model.layers[1].layers[0], tf.keras.layers.Dropout)
    assert model.layers[1].layers[0].rate == rate_all_layers[-1]

    assert isinstance(model.layers[1].layers[-1], tf.keras.layers.Dense)
    assert model.layers[1].layers[-1].units == int(np.prod(outputs.shape[1:]))
    assert model.layers[1].layers[-1].activation == tf.keras.activations.linear


def test_dropout_network_can_be_compiled(input_shape: List[int], output_shape: List[int]) -> None:
    """Checks that dropout networks are compilable."""
    inputs, outputs = inputs_outputs_spec(input_shape, output_shape)
    model = MonteCarloDropout(inputs, outputs)
    model.compile(tf.optimizers.Adam(), tf.losses.MeanSquaredError())

    assert model.compiled_loss is not None
    assert model.compiled_metrics is not None
    assert model.optimizer is not None


def test_dropout_network_can_dropout() -> None:
    """Tests the ability of architecture to dropout."""

    inputs, outputs = inputs_outputs_spec([1], [1])
    model = MonteCarloDropout(inputs, outputs, rate=0.999999999)
    model.compile(tf.optimizers.Adam(), tf.losses.MeanSquaredError())

    outputs = [model(tf.constant([[1.0]]), training=True) for _ in range(100)]
    npt.assert_almost_equal(
        0.0, np.mean(outputs), err_msg="MonteCarloDropout not dropping up to randomness"
    )


@pytest.mark.parametrize("rate", [1.5, -1.0])
def test_dropout_rate_raises_invalidargument_error(rate: float) -> None:
    """Tests that value error is raised when given wrong probability rates"""
    inputs, outputs = inputs_outputs_spec([1], [1])
    with pytest.raises(InvalidArgumentError):
        MonteCarloDropout(inputs, outputs, rate=rate)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_dropout_network_dtype(dtype: tf.DType) -> None:
    """Tests that network can infer data type from the data"""
    x = tf.constant([[1]], dtype=tf.float16)
    inputs, outputs = tf.TensorSpec([1], dtype), tf.TensorSpec([1], dtype)
    model = MonteCarloDropout(inputs, outputs)

    assert model(x).dtype == dtype


def test_dropout_network_accepts_scalars() -> None:
    """Tests that network can handle scalar inputs with ndim = 1 instead of 2"""
    inputs, outputs = inputs_outputs_spec([1, 1], [1, 1])
    model = MonteCarloDropout(inputs, outputs)

    test_points = tf.linspace(-1, 1, 100)

    assert model(test_points).shape == (100, 1)
