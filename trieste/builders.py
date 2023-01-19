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

from __future__ import annotations
from typing import Union

import tensorflow as tf
from trieste.data import Dataset
from trieste.models.keras.utils import get_tensor_spec_from_data

from uanets.models.mc_dropout import MonteCarloDropout


def build_vanilla_keras_mcdropout(
    data: Dataset,
    num_hidden_layers: int = 5,
    units: int = 300,
    activation: str | tf.keras.layers.Activation = "relu",
    rate: float = 0.1,
    dropout_network: MonteCarloDropout = MonteCarloDropout,
) -> MonteCarloDropout:

    """
    Builds a simple dropout network, in Keras where each network has the same
    architecture: number of hidden layers, nodes in hidden layers and activation function.

    Default number of hidden layers, units, rate, and activation function seem to work well in practice,
    in regression type of problems at least. Number of hidden layers and units per layer should be
    modified according to the dataset size and complexity of the function - the default values seem
    to work well for small datasets common in Bayesian optimization. The training is highly sensitive
    to the rate of dropout; a lower rate typically makes the function easier to learn at the expense of
    an easier time estimating the uncertainty and vice versa. DropConnectNetwork typically works better
    with a higher rate all else equal - a default of around 0.35 seems equivalent.

    :param dataset: Data for training, used for extracting input and output tensor specifications.
    :param num_hidden_layers: The number of hidden dropout layers in each network.
    :param units: The number of nodes in each hidden layer.
    :param activation: The activation function in each hidden layer.
    :param rate: The rate of dropout of each layer.
    :return: Keras MonteCarloDropout model.
    """
    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(data)

    hidden_layer_args = []
    for _ in range(num_hidden_layers):
        hidden_layer_args.append({"units": units, "activation": activation})

    keras_mcdropout = MonteCarloDropout(input_tensor_spec, output_tensor_spec, hidden_layer_args, rate)

    keras_mcdropout.build(data.query_points.shape)

    return keras_mcdropout
