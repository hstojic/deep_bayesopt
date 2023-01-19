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

from typing import Any, Dict, Sequence

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class MonteCarloDropout(tf.keras.Model):
    """
    This class builds a standard dropout neural network using Keras. The network
    architecture is a multilayer fully-connected feed-forward network, with Dropout layers
    preceding each fully connected dense layer. The network is meant to be passed to
    :class:`~trieste.models.keras.models.MonteCarloDropout` which will define the predict method
    to make this a probabilistic model. Otherwise this class will only use dropout in training.
    """

    def __init__(
        self,
        input_tensor_spec: tf.TensorSpec,
        output_tensor_spec: tf.TensorSpec,
        hidden_layer_args: Sequence[Dict[str, Any]] = (
            {"units": 300, "activation": "relu"},
            {"units": 300, "activation": "relu"},
            {"units": 300, "activation": "relu"},
            {"units": 300, "activation": "relu"},
            {"units": 300, "activation": "relu"},
        ),
        rate: float = 0.1,
    ):
        """
        :param input_tensor_spec: Tensor specification for the input of the network.
        :param output_tensor_spec: Tensor specification for the output of the network.
        :param hidden_layer_args: Specification for building dense hidden layers. Each element in
            the sequence should be a dictionary containing arguments (keys) and their values for a
            :class:`~tf.keras.layers.Dense` hidden layer. Please check Keras Dense layer API for
            available arguments. Objects in the sequence will sequentially be used to add
            :class:`~tf.keras.layers.Dense` layers with :class:`~tf.keras.layers.Dropout` layers
            added before each :class:`~tf.keras.layers.Dense` layer. Length of this sequence
            determines the number of hidden layers in the network. Default value is five hidden
            layers, 300 nodes each, with ReLu activation functions. Empty sequence needs to be passed
            to have no hidden layers.
        :param rate: Probability of dropout assigned to each `~tf.keras.layers.Dropout` layer. By default
            a rate of 0.1 is used.
        :raise ValueError: If objects in ``hidden_layer_args`` are not dictionaries.
        """
        super().__init__()
        self.input_tensor_spec = input_tensor_spec
        self.output_tensor_spec = output_tensor_spec
        self.flattened_output_shape = int(np.prod(self.output_tensor_spec.shape[1:]))
        self._hidden_layer_args = hidden_layer_args

        tf.debugging.assert_greater_equal(
            rate, 0.0, f"Rate needs to be a valid probability, instead got {rate}"
        )
        tf.debugging.assert_less_equal(
            rate, 1.0, f"Rate needs to be a valid probability, instead got {rate}"
        )
        self._rate = rate

        self.hidden_layers = self._gen_hidden_layers()
        self.output_layer = self._gen_output_layer()

    @property
    def rate(self) -> float:
        return self._rate

    def _gen_hidden_layers(self) -> tf.keras.Model:

        hidden_layers = tf.keras.Sequential(name="hidden_layers")
        for hidden_layer_args in self._hidden_layer_args:
            hidden_layers.add(
                tf.keras.layers.Dropout(rate=self.rate, dtype=self.input_tensor_spec.dtype)
            )
            hidden_layers.add(
                tf.keras.layers.Dense(dtype=self.input_tensor_spec.dtype, **hidden_layer_args)
            )
        return hidden_layers

    def _gen_output_layer(self) -> tf.keras.Model:

        output_layer = tf.keras.Sequential(name="output_layer")
        output_layer.add(
            tf.keras.layers.Dropout(rate=self.rate, dtype=self.input_tensor_spec.dtype)
        )
        output_layer.add(
            tf.keras.layers.Dense(
                units=self.flattened_output_shape, dtype=self.input_tensor_spec.dtype
            )
        )
        return output_layer

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        if inputs.shape.rank == 1:
            inputs = tf.expand_dims(inputs, axis=-1)

        hidden_output = self.hidden_layers(inputs)
        output = self.output_layer(hidden_output)

        return output
