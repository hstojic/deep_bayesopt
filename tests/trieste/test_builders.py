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

from typing import Union

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import empty_dataset
from trieste.models.keras import (
    DropConnect,
    DropConnectNetwork,
    DropoutNetwork,
    build_vanilla_keras_mcdropout,
    build_vanilla_keras_ensemble,
)


@pytest.mark.parametrize("units, activation", [(10, "relu"), (50, tf.keras.activations.tanh)])
@pytest.mark.parametrize("ensemble_size", [2, 5])
@pytest.mark.parametrize("independent_normal", [False, True])
@pytest.mark.parametrize("num_hidden_layers", [0, 1, 3])
def test_build_vanilla_keras_ensemble(
    ensemble_size: int,
    num_hidden_layers: int,
    units: int,
    activation: Union[str, tf.keras.layers.Activation],
    independent_normal: bool,
) -> None:
    example_data = empty_dataset([1], [1])
    keras_ensemble = build_vanilla_keras_ensemble(
        example_data,
        ensemble_size,
        num_hidden_layers,
        units,
        activation,
        independent_normal,
    )

    assert keras_ensemble.ensemble_size == ensemble_size
    assert len(keras_ensemble.model.layers) == num_hidden_layers * ensemble_size + 3 * ensemble_size
    if independent_normal:
        assert isinstance(keras_ensemble.model.layers[-1], tfp.layers.IndependentNormal)
    else:
        assert isinstance(keras_ensemble.model.layers[-1], tfp.layers.MultivariateNormalTriL)
    if num_hidden_layers > 0:
        for layer in keras_ensemble.model.layers[ensemble_size : -ensemble_size * 2]:
            assert layer.units == units
            assert layer.activation == activation or layer.activation.__name__ == activation


@pytest.mark.mcdropout
@pytest.mark.parametrize("units, activation", [(10, "relu"), (50, tf.keras.activations.tanh)])
@pytest.mark.parametrize("num_hidden_layers", [0, 1, 3])
@pytest.mark.parametrize("rate", [0.1, 0.9])
@pytest.mark.parametrize("dropout", [DropoutNetwork, DropConnectNetwork])
def test_build_vanilla_keras_mcdropout(
    num_hidden_layers: int,
    units: int,
    activation: Union[str, tf.keras.layers.Activation],
    rate: float,
    dropout: str,
) -> None:
    example_data = empty_dataset([1], [1])
    mcdropout = build_vanilla_keras_mcdropout(
        example_data, num_hidden_layers, units, activation, rate, dropout
    )

    assert mcdropout.built
    assert isinstance(mcdropout, dropout)
    assert len(mcdropout.layers) == 2

    # Check Hidden Layers
    if num_hidden_layers > 0:
        for i, layer in enumerate(mcdropout.layers[0].layers):
            if dropout == DropConnectNetwork:
                assert len(mcdropout.layers[0].layers) == num_hidden_layers
                assert isinstance(layer, DropConnect)
                assert layer.units == units
                assert layer.activation == activation or layer.activation.__name__ == activation
            elif dropout == DropoutNetwork:
                assert len(mcdropout.layers[0].layers) == num_hidden_layers * 2
                if i % 2 == 0:
                    assert isinstance(layer, tf.keras.layers.Dropout)
                    assert layer.rate == rate
                elif i % 2 == 1:
                    assert isinstance(layer, tf.keras.layers.Dense)
                    assert layer.units == units
                    assert layer.activation == activation or layer.activation.__name__ == activation

    # Check Output Layers
    if dropout == DropConnectNetwork:
        assert isinstance(mcdropout.layers[1], DropConnect)
        assert mcdropout.layers[1].rate == rate
    elif dropout == DropoutNetwork:
        assert isinstance(mcdropout.layers[1].layers[0], tf.keras.layers.Dropout)
        assert mcdropout.layers[1].layers[0].rate == rate
        assert isinstance(mcdropout.layers[1].layers[1], tf.keras.layers.Dense)
