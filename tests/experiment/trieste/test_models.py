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

from typing import Any, List

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
from trieste.data import Dataset
from trieste.models.optimizer import KerasOptimizer

from experiment.trieste import TriesteMonteCarloDropout
from tests.util.functions import fnc_2sin_x_over_3
from tests.util.misc import ShapeLike, empty_dataset, random_dataset, random_seed
from tests.util.models import montecarlo_dropout_test, trieste_montecarlo_dropout_test


@pytest.fixture(name="rate", params=[0.1])
def _rate_fixture(request: Any) -> float:
    return request.param


def _fnc_2sin_x_over_3_data(query_point_shape: ShapeLike) -> Dataset:
    qp = tf.random.uniform(tf.TensorShape(query_point_shape), dtype=tf.float64)
    obs = fnc_2sin_x_over_3(qp)
    return Dataset(qp, obs)


def _linear_data(query_point_shape: ShapeLike, slope: str) -> Dataset:
    qp = tf.random.uniform(tf.TensorShape(query_point_shape), minval=-4, maxval=4, dtype=tf.float32)
    if slope == "pos":
        obs = tf.multiply(3, qp)
    elif slope == "neg":
        obs = tf.multiply(-3, qp)
    return Dataset(qp, obs)


def test_dropout_network_model_attribute(rate: float) -> None:
    example_data = empty_dataset([1], [1])
    model, dropout_network, optimizer = trieste_montecarlo_dropout_test(example_data, rate=rate)

    dropout_network.compile(optimizer=optimizer.optimizer, loss=optimizer.loss)

    assert model.model is dropout_network


def test_dropout_network_default_optimizer_is_correct(rate: float) -> None:
    example_data = empty_dataset([1], [1])

    dropout_network = montecarlo_dropout_test(
        example_data.query_points, example_data.observations, rate
    )
    model = TriesteMonteCarloDropout(dropout_network)
    default_loss = "mse"
    default_fit_args = {
        "batch_size": 32,
        "epochs": 1000,
        "verbose": 0,
    }
    del model.optimizer.fit_args["callbacks"]

    assert isinstance(model.optimizer, KerasOptimizer)
    assert isinstance(model.optimizer.optimizer, tf.optimizers.Optimizer)
    assert model.optimizer.fit_args == default_fit_args
    assert model.optimizer.loss == default_loss


def test_montecarlo_dropout_optimizer_changed_correctly(rate: float) -> None:
    example_data = empty_dataset([1], [1])

    custom_fit_args = {
        "verbose": 1,
        "epochs": 10,
        "batch_size": 10,
    }
    custom_optimizer = tf.optimizers.RMSprop()
    custom_loss = "mean_absolute_error"
    optimizer_wrapper = KerasOptimizer(custom_optimizer, custom_fit_args, custom_loss)

    dropout_network = montecarlo_dropout_test(
        example_data.query_points, example_data.observations, rate
    )
    model = TriesteMonteCarloDropout(dropout_network, optimizer_wrapper)

    assert model.optimizer == optimizer_wrapper
    assert model.optimizer.optimizer == custom_optimizer
    assert model.optimizer.fit_args == custom_fit_args


def test_montecarlo_dropout_is_compiled(rate: float) -> None:
    example_data = empty_dataset([1], [1])
    model, _, _ = trieste_montecarlo_dropout_test(example_data, rate)

    assert model.model.compiled_loss is not None
    assert model.model.compiled_metrics is not None
    assert model.model.optimizer is not None


@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        [[1, 1], [1, 1]],
        [[10, 5], [10, 3]],
    ],
)
def test_montecarlo_dropout_predict_call_shape(
    input_shape: List[int], output_shape: List[int], rate: float
) -> None:
    example_data = random_dataset(input_shape, output_shape)

    model, _, _ = trieste_montecarlo_dropout_test(example_data, rate)
    predicted_means, predicted_vars = model.predict(example_data.query_points)

    assert tf.is_tensor(predicted_means)
    assert predicted_means.shape == example_data.observations.shape
    assert tf.is_tensor(predicted_vars)
    assert predicted_vars.shape == example_data.observations.shape


@pytest.mark.parametrize("num_samples", [6, 12])
@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        [[1, 1], [1, 1]],
        [[10, 5], [10, 3]],
    ],
)
def test_montecarlo_dropout_sample_call_shape(
    input_shape: List[int], output_shape: List[int], num_samples: int, rate: float
) -> None:
    example_data = random_dataset(input_shape, output_shape)
    model, _, _ = trieste_montecarlo_dropout_test(example_data, rate)

    samples = model.sample(example_data.query_points, num_samples)

    assert tf.is_tensor(samples)
    assert samples.shape == [num_samples] + output_shape


@random_seed
def test_montecarlo_dropout_optimize_with_defaults(rate: float) -> None:
    example_data = _fnc_2sin_x_over_3_data([100, 1])

    model, _, _ = trieste_montecarlo_dropout_test(example_data, rate)
    model.optimize(example_data)
    loss = model.model.history.history["loss"]

    assert loss[-1] < loss[0]


def test_montecarlo_dropout_learning_rate_resets_correctly(rate: float) -> None:
    example_data = _fnc_2sin_x_over_3_data([100, 1])

    model, _, _ = trieste_montecarlo_dropout_test(example_data, rate)
    lr_init = model.optimizer.optimizer.lr

    model.optimize(example_data)
    lr1 = model.optimizer.optimizer.lr

    model.optimize(example_data)
    lr2 = model.optimizer.optimizer.lr

    assert lr1 == lr2 and lr_init == lr1


@random_seed
def test_montecarlo_dropout_optimizer_learns_new_data(rate: float) -> None:

    positive_slope = _linear_data([20, 1], "pos")
    negative_slope = _linear_data([20, 1], "neg")
    new_data = positive_slope + negative_slope
    qp = tf.constant([[1.0]])

    model, _, _ = trieste_montecarlo_dropout_test(positive_slope, rate)

    model.optimize(positive_slope)
    pred1, _ = model.predict(qp)
    model.optimize(new_data)
    pred2, _ = model.predict(qp)

    assert np.abs(pred1 - pred2) > 1


@random_seed
@pytest.mark.parametrize("num_samples", [1000, 5000, 10000])
def test_montecarlo_dropout_sample(rate: float, num_samples) -> None:
    example_data = _fnc_2sin_x_over_3_data([100, 1])
    model, _, _ = trieste_montecarlo_dropout_test(example_data, rate)

    samples = model.sample(example_data.query_points, num_samples)
    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2, axis=0)

    ref_mean, ref_variance = model.predict(example_data.query_points)

    error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))

    npt.assert_allclose(sample_mean, ref_mean, atol=4 * error)
    npt.assert_allclose(sample_variance, ref_variance, atol=8 * error)
