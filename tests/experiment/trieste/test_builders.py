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

from experiment.trieste import build_montecarlo_dropout
from tests.util.trieste import empty_dataset
from unflow.models import MonteCarloDropout


@pytest.mark.parametrize("units, activation", [(10, "relu"), (50, tf.keras.activations.tanh)])
@pytest.mark.parametrize("num_hidden_layers", [0, 1, 3])
@pytest.mark.parametrize("rate", [0.1, 0.9])
def test_build_montecarlo_dropout(
    num_hidden_layers: int,
    units: int,
    activation: Union[str, tf.keras.layers.Activation],
    rate: float,
) -> None:
    example_data = empty_dataset([1], [1])
    model = build_montecarlo_dropout(example_data, num_hidden_layers, units, activation, rate)

    assert model.built
    assert isinstance(model, MonteCarloDropout)
    assert len(model.layers) == 2

    # Check Hidden Layers
    if num_hidden_layers > 0:
        for i, layer in enumerate(model.layers[0].layers):
            assert len(model.layers[0].layers) == num_hidden_layers * 2
            if i % 2 == 0:
                assert isinstance(layer, tf.keras.layers.Dropout)
                assert layer.rate == rate
            elif i % 2 == 1:
                assert isinstance(layer, tf.keras.layers.Dense)
                assert layer.units == units
                assert layer.activation == activation or layer.activation.__name__ == activation

    # Check Output Layers
    assert isinstance(model.layers[1].layers[0], tf.keras.layers.Dropout)
    assert model.layers[1].layers[0].rate == rate
    assert isinstance(model.layers[1].layers[1], tf.keras.layers.Dense)
