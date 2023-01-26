# Copyright 2023 The deep_bayesopt Authors. All Rights Reserved.
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
"""
Contains some custom uanet layers.
"""
from typing import Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops import math_ops, nn_ops
from tensorflow_probability.python.layers.internal import distribution_tensor_coercible as dtc


def _as_tensor_coercible(distribution: tfp.distributions.Distribution) -> tf.Tensor:
    coerced_distribution = dtc._TensorCoercible(distribution=distribution)
    coerced_distribution._shape = coerced_distribution._value().shape
    return coerced_distribution


class UnitNormalLayer(tf.keras.layers.Layer):
    """Layer that returns a unit normal distribution i.e. N(0, 1)."""

    def __init__(self, units: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if units < 0:
            raise ValueError(f"units must be a positive integer, got {units:d}.")
        self.units = units

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        loc = tf.zeros(shape=(*inputs.shape[:-1], self.units), dtype=self._compute_dtype_object)
        scale_diag = tf.ones_like(loc)
        return _as_tensor_coercible(
            tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class DenseStochasticLayer(tf.keras.layers.Layer):
    """Just your regular densely-connected stochastic NN layer."""

    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, tf.keras.layers.Activation]] = None,
        use_bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(activity_regularizer=None, **kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape: tf.Tensor) -> None:
        dtype = tf.dtypes.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                f"Unable to build `DenseStochasticLayer` with non-floating point dtype {dtype}"
            )
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of inputs to `DenseStochasticLayer` must be defined."
                "Found `None`."
            )
        self.w_mean = self.add_weight(
            "w_mean",
            shape=[last_dim, self.units],
            initializer="zeros",
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=True,
        )
        self.w_std = self.add_weight(
            "w_std",
            shape=[last_dim, self.units],
            initializer="ones",
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=True,
        )
        glorot_init = tf.sqrt(max(1.0, 2.0 / (last_dim + self.units)))
        self.w_std.assign(self.w_std * tfp.math.softplus_inverse(glorot_init))
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer="zeros",
                regularizer=None,
                constraint=None,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

        scale = tf.nn.softplus(self.w_std)
        # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        scale /= 0.8796256610342398
        z = tf.random.truncated_normal(shape=self.w_mean.shape, dtype=self.dtype)
        weights = self.w_mean + scale * z

        outputs = tf.matmul(a=inputs, b=weights)

        if self.use_bias:
            outputs = nn_ops.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {"units": self.units, "activation": tf.keras.activations.serialize(self.activation)}
        )
        return config


class GaussianLikelihoodLayer(tf.keras.layers.Layer):
    def __init__(self, noise_std_init: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        self.untransformed_noise_std = tf.Variable(
            initial_value=tfp.math.softplus_inverse(noise_std_init),
            dtype=self.dtype,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)
        tf.debugging.assert_equal(tf.shape(inputs)[-1], 1)

        distribution = tfp.distributions.Normal(
            loc=inputs, scale=tf.nn.softplus(self.untransformed_noise_std)
        )
        return _as_tensor_coercible(distribution)

    def get_config(self):
        return super().get_config()
