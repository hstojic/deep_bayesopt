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

import tensorflow as tf
from trieste.data import Dataset

from uanets.models.mc_dropout import MonteCarloDropout


def build_montecarlo_dropout(
    data: Dataset,
    num_hidden_layers: int = 5,
    units: int = 300,
    activation: Union[str, tf.keras.layers.Activation] = "relu",
    rate: float = 0.1,
) -> MonteCarloDropout:

    """
    Builds a simple dropout network in Keras where Dropout layers are followed by Dense layers,
    and we can conveniently specify the number of hidden layers, units and activation function
    in the hidden layers.

    Default number of hidden layers, units, rate, and activation function seem to work well in
    practice, in regression type of problems at least. Number of hidden layers and units per layer
    should be modified according to the dataset size and complexity of the function.

    The training is highly sensitive to the rate of dropout; a lower rate typically makes the
    function easier to learn at the expense of an easier time estimating the uncertainty and vice
    versa.

    :param data: Data for training, used for extracting input and output tensor specifications.
    :param num_hidden_layers: The number of hidden layers in each network.
    :param units: The number of nodes in each hidden Dense layer.
    :param activation: The activation function in each hidden Dense layer.
    :param rate: The rate of dropout of each Dropout layer.
    :return: MonteCarloDropout model that is already built.
    """
    hidden_layer_args = []
    for _ in range(num_hidden_layers):
        hidden_layer_args.append({"units": units, "activation": activation})

    model = MonteCarloDropout(
        input_tensor_spec=tf.TensorSpec.from_tensor(data.query_points, name="input"),
        output_tensor_spec=tf.TensorSpec.from_tensor(data.observations, name="output"),
        hidden_layer_args=hidden_layer_args,
        rate=rate,
    )
    model.build(data.query_points.shape)

    return model
