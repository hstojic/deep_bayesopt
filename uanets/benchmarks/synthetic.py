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

"""
This module contains synthetic benchmark functions, useful for experimentation and verifying
that model works as intended.
"""

import tensorflow as tf


def power(x: tf.Tensor, error: bool = True) -> tf.Tensor:
    """
    A simple one-dimensional toy problem introduced by
    <cite data-cite="hernandez2015probabilistic"/>, that can be used too provide some illustrative
    evidence that models do a good job of estimating uncertainty.

    The toy problem is a simple cubic function with some Normally distributed noise around it.
    Typical usage is to randomly sample 20 input points from [-4,4] interval that is used
    as a training data and test data sampled from wider [-6, 6] interval.
    """
    y = tf.pow(x, 3)
    if error:
        y += tf.random.normal(x.shape, 0, 3, dtype=x.dtype)
    return y
