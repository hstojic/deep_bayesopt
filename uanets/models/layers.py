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
Contains some custom uanet layers
"""
import tensorflow as tf
from tensorflow_probability import distributions as tfpd
from tensorflow_probability.python.layers.internal import distribution_tensor_coercible as dtc


def _as_tensor_coercible(distribution: tfpd.Distribution) -> tf.Tensor:
    coerced_distribution = dtc._TensorCoercible(distribution=distribution)
    coerced_distribution._shape = coerced_distribution._value().shape
    return coerced_distribution


class UnitNormalLayer(tf.keras.layers.Layer):
    """ Layer that returns a unit normal distribution i.e. N(0, 1). """

    def __init__(self, units: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if units <= 0:
            raise ValueError(f"units must be a positive integer, got {units:d}.")
        self.units = units

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        loc = tf.zeros(shape=(*inputs.shape[:-1], self.units), dtype=self._compute_dtype_object)
        scale_diag = tf.ones_like(loc)
        return _as_tensor_coercible(tfpd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag))

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
