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

from typing import Tuple

import tensorflow as tf
from trieste.data import Dataset
from trieste.models.optimizer import KerasOptimizer

from experiment.trieste import TriesteMonteCarloDropout
from tests.util.misc import ShapeLike
from tests.util.models import montecarlo_dropout_test
from unflow.models.dropout import MonteCarloDropout


def trieste_montecarlo_dropout_test(data: Dataset, rate: float = 0.1) -> Tuple[TriesteMonteCarloDropout, MonteCarloDropout, KerasOptimizer]:

    model = montecarlo_dropout_test(data.query_points, data.observations, rate=rate)

    optimizer = tf.keras.optimizers.Adam(0.01)
    fit_args = {
        "batch_size": 10,
        "epochs": 100,
        "verbose": 0,
    }
    trieste_optimizer = KerasOptimizer(optimizer, fit_args)

    trieste_model = TriesteMonteCarloDropout(model=model, optimizer=trieste_optimizer)

    return trieste_model, model, trieste_optimizer


def empty_dataset(
    query_point_shape: ShapeLike, observation_shape: ShapeLike, dtype: tf.DType = tf.float64
) -> Dataset:
    """
    Returns an empty Trieste dataset, with query point and observations shapes

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
    Creates a Trieste dataset filled with random numbers drawn from uniform distribution.
    Leading dimension of ``query_point_shape`` and ``observation_shape`` has to be the same.
    This will be enforced when creating the ``Dataset`` object.

    :param query_point_shape: The shape of query points.
    :param observation_shape: The shape of observations.
    :return: A dataset with points of the specified shapes, and dtype `tf.float64`.
    """
    qp = tf.random.uniform(query_point_shape, dtype=dtype)
    obs = tf.random.uniform(observation_shape, dtype=dtype)
    return Dataset(qp, obs)
