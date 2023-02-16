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


def fnc_3x_plus_10(x: tf.Tensor) -> tf.Tensor:
    return 3.0 * x + 10


def fnc_2sin_x_over_3(x: tf.Tensor) -> tf.Tensor:
    return 2.0 * tf.math.sin(x / 3.0)


def quadratic(x: tf.Tensor) -> tf.Tensor:
    r"""
    The multi-dimensional quadratic function.

    :param x: A tensor whose last dimension is of length greater than zero.
    :return: The sum :math:`\Sigma x^2` of the squares of ``x``.
    :raise ValueError: If ``x`` is a scalar or has empty trailing dimension.
    """
    if x.shape == [] or x.shape[-1] == 0:
        raise ValueError(f"x must have non-empty trailing dimension, got shape {x.shape}")

    return tf.reduce_sum(x**2, axis=-1, keepdims=True)
