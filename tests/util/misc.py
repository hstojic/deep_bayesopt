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

import functools
import os
import random
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, Union, cast, overload

import numpy as np
import tensorflow as tf
from trieste.data import Dataset

ShapeLike = Union[tf.TensorShape, Sequence[int]]
""" Type alias for types that can represent tensor shapes. """


C = TypeVar("C", bound=Callable[..., object])
""" Type variable bound to `typing.Callable`. """


@overload
def random_seed(f_py: C, seed: int = 0) -> C:
    ...


@overload
def random_seed(f_py: None = None, seed: int = 0) -> Callable[[C], C]:
    ...


def random_seed(f_py: Optional[C] = None, seed: int = 0) -> Union[Callable[[C], C], C]:
    """
    Decorates function ``f`` with TensorFlow, numpy and Python randomness seeds fixed to ``seed``.
    This decorator can be used without and with the ``seed`` parameter. When used with the default
    seed::

        @random_seed
        def foo():
            pass

    or::

        @random_seed()
        def foo():
            pass

    However, if ``seed`` needs to be set to a custom value parameter needs to be named::

        @random_seed(seed=1)
        def foo():
            pass

    :param f_py: A function to be decorated, used when ``seed`` parameter is not set.
    :param seed: A seed to be fixed, defaults to 0.
    """
    assert callable(f_py) or f_py is None

    def _decorator(f: C) -> C:
        """
        :param f: A function.
        :return: The function ``f``, but with TensorFlow, numpy and Python randomness seeds fixed.
        """

        @functools.wraps(f)
        def decorated(*args: Any, **kwargs: Any) -> Any:
            os.environ["PYTHONHASHSEED"] = str(seed)
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            return f(*args, **kwargs)

        return cast(C, decorated)

    if f_py is None:
        return _decorator
    else:
        return _decorator(f_py)


def inputs_outputs_spec(
    inputs_shape: ShapeLike, outputs_shape: ShapeLike, dtype: tf.DType = tf.float64
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    :param inputs_shape: The shape of an input without the first dimension.
    :param outputs_shape: The shape of an output without the first dimension.
    :param dtype: The dtype of the tensors.
    :return: Tensor specification for inputs and outputs.
    """
    inputs = tf.TensorSpec(tf.TensorShape([0]) + inputs_shape, dtype)
    outputs = tf.TensorSpec(tf.TensorShape([0]) + outputs_shape, dtype)
    return inputs, outputs


def empty_dataset(
    query_point_shape: ShapeLike, observation_shape: ShapeLike, dtype: tf.DType = tf.float64
) -> Dataset:
    """
    :param query_point_shape: The shape of query points without the first dimension.
    :param observation_shape: The shape of observations without the first dimension.
    :return: An empty dataset with points of the specified shapes, and dtype `tf.float64`.
    """
    qp = tf.zeros(tf.TensorShape([0]) + query_point_shape, dtype)
    obs = tf.zeros(tf.TensorShape([0]) + observation_shape, dtype)
    return Dataset(qp, obs)


def random_dataset(
    query_point_shape: ShapeLike, observation_shape: ShapeLike, dtype: tf.DType = tf.float64
) -> Dataset:
    """
    :param query_point_shape: The shape of query points.
    :param observation_shape: The shape of observations.
    :return: A dataset with points of the specified shapes, and dtype `tf.float64`.
    """
    qp = tf.random.uniform(query_point_shape, dtype=dtype)
    obs = tf.random.uniform(observation_shape, dtype=dtype)
    return Dataset(qp, obs)
