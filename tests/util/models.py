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

import tensorflow as tf
from trieste.data import Dataset
from trieste.models.optimizer import KerasOptimizer

from experiment.trieste import TriesteMonteCarloDropout
from uanets.models.mc_dropout import MonteCarloDropout


def montecarlo_dropout_test(
    inputs: tf.Tensor,
    outputs: tf.Tensor,
    rate: float = 0.1,
) -> MonteCarloDropout:

    input_tensor_spec = tf.TensorSpec.from_tensor(inputs, name="input")
    output_tensor_spec = tf.TensorSpec.from_tensor(outputs, name="output")

    model = MonteCarloDropout(
        input_tensor_spec,
        output_tensor_spec,
        hidden_layer_args=[
            {"units": 300, "activation": "relu"},
            {"units": 300, "activation": "relu"},
            {"units": 300, "activation": "relu"},
        ],
        rate=rate,
    )
    model.build(inputs.shape)

    return model


def trieste_montecarlo_dropout_test(data: Dataset, rate: float = 0.1) -> TriesteMonteCarloDropout:

    model = montecarlo_dropout_test(data.query_points, data.observations, rate=rate)

    optimizer = tf.keras.optimizers.Adam(0.01)
    fit_args = {
        "batch_size": 10,
        "epochs": 100,
        "verbose": 0,
    }
    trieste_optimizer = KerasOptimizer(optimizer, fit_args)

    trieste_model = TriesteMonteCarloDropout(model, trieste_optimizer)

    return trieste_model, model, trieste_optimizer
