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
    Input output tensor specification, assuming empty tensors that have 0 for the leading
    dimension.

    :param inputs_shape: The shape of an input without the first dimension.
    :param outputs_shape: The shape of an output without the first dimension.
    :param dtype: The dtype of the tensors.
    :return: An empty dataset with points of the specified shapes, and dtype `tf.float64`.
    """
    inputs = tf.TensorSpec(tf.TensorShape([0]) + inputs_shape, dtype)
    outputs = tf.TensorSpec(tf.TensorShape([0]) + outputs_shape, dtype)
    return inputs, outputs


def random_inputs_outputs(
    inputs_shape: ShapeLike, outputs_shape: ShapeLike, dtype: tf.DType = tf.float64
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Creates input and output tensors filled with random numbers drawn from uniform distribution.
    Leading dimension of ``inputs_shape`` and ``outputs_shape`` has to be the same.

    :param inputs_shape: The shape of an input.
    :param outputs_shape: The shape of an output.
    :param dtype: The dtype of the tensors.
    :return: Random tensors with points of the specified shapes, and dtype `tf.float64`.
    :raise ValueError (or InvalidArgumentError): If ``inputs_shape`` and ``outputs_shape`` have
        unequal shape in any but their last dimension.
    """
    if inputs_shape[:-1] != outputs_shape[:-1]:
        raise ValueError(
            f"Leading shapes of inputs_shape and outputs_shape must match. Got shapes"
            f" {inputs_shape}, {outputs_shape}."
        )
    inputs = tf.random.uniform(inputs_shape, dtype=dtype)
    outputs = tf.random.uniform(outputs_shape, dtype=dtype)
    return inputs, outputs
