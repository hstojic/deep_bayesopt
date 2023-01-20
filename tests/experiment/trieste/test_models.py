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
from trieste.models import create_model
from trieste.models.keras import (
    DeepEnsemble,
    DropoutNetwork,
    KerasEnsemble,
    MonteCarloDropout,
    negative_log_likelihood,
    sample_with_replacement,
)
from trieste.models.keras.architectures import DropConnectNetwork
from trieste.models.optimizer import KerasOptimizer, TrainingData

from tests.util.misc import ShapeLike, empty_dataset, random_seed
from tests.util.models.keras.models import (
    trieste_deep_ensemble_model,
    trieste_dropout_network_model,
    trieste_keras_ensemble_model,
    trieste_mcdropout_model,
)
from tests.util.models.models import fnc_2sin_x_over_3

_ENSEMBLE_SIZE = 3


@pytest.fixture(name="ensemble_size", params=[2, 5])
def _ensemble_size_fixture(request: Any) -> int:
    return request.param


@pytest.fixture(name="independent_normal", params=[False, True])
def _independent_normal_fixture(request: Any) -> bool:
    return request.param


@pytest.fixture(name="bootstrap_data", params=[False, True])
def _bootstrap_data_fixture(request: Any) -> bool:
    return request.param


@pytest.fixture(name="rate", params=[0.1])
def _rate_fixture(request: Any) -> float:
    return request.param


@pytest.fixture(name="loss")
def _loss_fixture(request: Any) -> float:
    return "mse"


def trieste_mcdropout_model(
    example_data: Dataset, rate: float = 0.1, dropout: DropoutNetwork = DropoutNetwork
) -> MonteCarloDropout:

    dropout_network = trieste_dropout_network_model(example_data, rate=rate, dropout=dropout)

    optimizer = tf.keras.optimizers.Adam(0.01)
    fit_args = {
        "batch_size": 10,
        "epochs": 100,
        "verbose": 0,
    }
    optimizer_wrapper = KerasOptimizer(optimizer, fit_args)

    model = MonteCarloDropout(dropout_network, optimizer_wrapper)

    return model, dropout_network, optimizer_wrapper


def _get_example_data(query_point_shape: ShapeLike) -> Dataset:
    qp = tf.random.uniform(tf.TensorShape(query_point_shape), dtype=tf.float64)
    obs = fnc_2sin_x_over_3(qp)
    return Dataset(qp, obs)


def _get_linear_data(query_point_shape: ShapeLike, slope: str) -> Dataset:
    qp = tf.random.uniform(tf.TensorShape(query_point_shape), minval=-4, maxval=4, dtype=tf.float32)
    if slope == "pos":
        obs = tf.multiply(3, qp)
    elif slope == "neg":
        obs = tf.multiply(-3, qp)
    return Dataset(qp, obs)


def _ensemblise_data(
    model: KerasEnsemble, data: Dataset, ensemble_size: int, bootstrap: bool = False
) -> TrainingData:
    inputs = {}
    outputs = {}
    for index in range(ensemble_size):
        if bootstrap:
            resampled_data = sample_with_replacement(data)
        else:
            resampled_data = data
        input_name = model.model.input_names[index]
        output_name = model.model.output_names[index]
        inputs[input_name], outputs[output_name] = resampled_data.astuple()

    return inputs, outputs


@pytest.mark.parametrize("optimizer", [tf.optimizers.Adam(), tf.optimizers.RMSprop()])
def test_deep_ensemble_repr(
    optimizer: tf.optimizers.Optimizer,
    bootstrap_data: bool,
) -> None:
    example_data = empty_dataset([1], [1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE)
    keras_ensemble.model.compile(optimizer, loss=negative_log_likelihood)
    optimizer_wrapper = KerasOptimizer(optimizer, loss=negative_log_likelihood)
    model = DeepEnsemble(keras_ensemble, optimizer_wrapper, bootstrap_data)

    expected_repr = (
        f"DeepEnsemble({keras_ensemble.model!r}, {optimizer_wrapper!r}, {bootstrap_data!r})"
    )

    assert type(model).__name__ in repr(model)
    assert repr(model) == expected_repr


def test_deep_ensemble_model_attributes() -> None:
    example_data = empty_dataset([1], [1])
    model, keras_ensemble, optimizer = trieste_deep_ensemble_model(
        example_data, _ENSEMBLE_SIZE, False, False
    )

    keras_ensemble.model.compile(optimizer=optimizer.optimizer, loss=optimizer.loss)

    assert model.model is keras_ensemble.model


def test_deep_ensemble_ensemble_size_attributes(ensemble_size: int) -> None:
    example_data = empty_dataset([1], [1])
    model, _, _ = trieste_deep_ensemble_model(example_data, ensemble_size, False, False)

    assert model.ensemble_size == ensemble_size


@pytest.mark.parametrize("ensemble_size", [-1, 1])
def test_deep_ensemble_raises_for_incorrect_ensemble_size(ensemble_size: int) -> None:

    example_data = empty_dataset([1], [1])

    with pytest.raises(ValueError):
        trieste_deep_ensemble_model(example_data, ensemble_size, False, False)


def test_deep_ensemble_default_optimizer_is_correct() -> None:
    example_data = empty_dataset([1], [1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, False)
    model = DeepEnsemble(keras_ensemble)
    default_loss = negative_log_likelihood
    default_fit_args = {
        "verbose": 0,
        "epochs": 1000,
        "batch_size": 16,
    }
    del model.optimizer.fit_args["callbacks"]

    assert isinstance(model.optimizer, KerasOptimizer)
    assert isinstance(model.optimizer.optimizer, tf.optimizers.Optimizer)
    assert model.optimizer.fit_args == default_fit_args
    assert model.optimizer.loss == default_loss


def test_deep_ensemble_optimizer_changed_correctly() -> None:
    example_data = empty_dataset([1], [1])

    custom_fit_args = {
        "verbose": 1,
        "epochs": 10,
        "batch_size": 10,
    }
    custom_optimizer = tf.optimizers.RMSprop()
    custom_loss = tf.keras.losses.MeanSquaredError()
    optimizer_wrapper = KerasOptimizer(custom_optimizer, custom_fit_args, custom_loss)

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE)
    model = DeepEnsemble(keras_ensemble, optimizer_wrapper)

    assert model.optimizer == optimizer_wrapper
    assert model.optimizer.optimizer == custom_optimizer
    assert model.optimizer.fit_args == custom_fit_args


def test_deep_ensemble_is_compiled() -> None:
    example_data = empty_dataset([1], [1])
    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE)

    assert model.model.compiled_loss is not None
    assert model.model.compiled_metrics is not None
    assert model.model.optimizer is not None


@pytest.mark.skip
def test_config_builds_deep_ensemble_and_default_optimizer_is_correct() -> None:
    example_data = empty_dataset([1], [1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE)

    model_config = {"model": keras_ensemble}
    model = create_model(model_config)
    default_fit_args = {
        "verbose": 0,
        "epochs": 100,
        "batch_size": 100,
    }

    assert isinstance(model, DeepEnsemble)
    assert isinstance(model.optimizer, KerasOptimizer)
    assert isinstance(model.optimizer.optimizer, tf.keras.optimizers.Optimizer)
    assert model.optimizer.fit_args == default_fit_args


@pytest.mark.parametrize("size", [0, 1, 10])
def test_deep_ensemble_sample_index_call_shape(size: int) -> None:
    example_data = empty_dataset([1], [1])
    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE, False, False)

    network_indices = model.sample_index(size)

    assert network_indices.shape == (size,)


@random_seed
@pytest.mark.parametrize("ensemble_size", [2, 5, 10, 20])
def test_deep_ensemble_sample_index_samples_are_diverse(ensemble_size: int) -> None:
    example_data = empty_dataset([1], [1])
    model, _, _ = trieste_deep_ensemble_model(example_data, ensemble_size, False, False)

    network_indices = model.sample_index(1000)
    # breakpoint()
    assert tf.math.reduce_variance(tf.cast(network_indices, tf.float32)) > 0
    assert tf.reduce_min(network_indices) == 0
    assert tf.reduce_max(network_indices) == (ensemble_size - 1)


@pytest.mark.parametrize("dataset_size", [10, 100])
def test_deep_ensemble_predict_call_shape(ensemble_size: int, dataset_size: int) -> None:
    example_data = _get_example_data([dataset_size, 1])
    model, _, _ = trieste_deep_ensemble_model(example_data, ensemble_size, False, False)

    predicted_means, predicted_vars = model.predict(example_data.query_points)

    assert tf.is_tensor(predicted_vars)
    assert predicted_vars.shape == example_data.observations.shape
    assert tf.is_tensor(predicted_means)
    assert predicted_means.shape == example_data.observations.shape


@pytest.mark.parametrize("dataset_size", [10, 100])
def test_deep_ensemble_predict_ensemble_call_shape(ensemble_size: int, dataset_size: int) -> None:
    example_data = _get_example_data([dataset_size, 1])
    model, _, _ = trieste_deep_ensemble_model(example_data, ensemble_size, False, False)

    predicted_means, predicted_vars = model.predict_ensemble(example_data.query_points)

    assert predicted_means.shape[-3] == ensemble_size
    assert predicted_vars.shape[-3] == ensemble_size
    assert tf.is_tensor(predicted_means)
    assert tf.is_tensor(predicted_vars)
    assert predicted_means.shape[-2:] == example_data.observations.shape
    assert predicted_vars.shape[-2:] == example_data.observations.shape


@pytest.mark.parametrize("num_samples", [6, 12])
@pytest.mark.parametrize("dataset_size", [4, 8])
def test_deep_ensemble_sample_call_shape(num_samples: int, dataset_size: int) -> None:
    example_data = _get_example_data([dataset_size, 1])
    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE, False, False)

    samples = model.sample(example_data.query_points, num_samples)

    assert tf.is_tensor(samples)
    assert samples.shape == [num_samples, dataset_size, 1]


@pytest.mark.parametrize("num_samples", [6, 12])
@pytest.mark.parametrize("dataset_size", [4, 8])
def test_deep_ensemble_sample_ensemble_call_shape(num_samples: int, dataset_size: int) -> None:
    example_data = _get_example_data([dataset_size, 1])
    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE, False, False)

    samples = model.sample_ensemble(example_data.query_points, num_samples)

    assert tf.is_tensor(samples)
    assert samples.shape == [num_samples, dataset_size, 1]


@random_seed
def test_deep_ensemble_optimize_with_defaults(independent_normal: bool) -> None:
    example_data = _get_example_data([100, 1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, independent_normal)

    model = DeepEnsemble(keras_ensemble)

    model.optimize(example_data)
    loss = model.model.history.history["loss"]

    assert loss[-1] < loss[0]


@random_seed
@pytest.mark.parametrize("epochs", [5, 15])
def test_deep_ensemble_optimize(
    independent_normal: bool,
    ensemble_size: int,
    bootstrap_data: bool,
    epochs: int,
) -> None:
    example_data = _get_example_data([100, 1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, ensemble_size, independent_normal)

    custom_optimizer = tf.optimizers.RMSprop()
    custom_fit_args = {
        "verbose": 0,
        "epochs": epochs,
        "batch_size": 10,
    }
    custom_loss = tf.keras.losses.MeanSquaredError()
    optimizer_wrapper = KerasOptimizer(custom_optimizer, custom_fit_args, custom_loss)

    model = DeepEnsemble(keras_ensemble, optimizer_wrapper, bootstrap_data)

    model.optimize(example_data)
    loss = model.model.history.history["loss"]
    ensemble_losses = ["output_loss" in elt for elt in model.model.history.history.keys()]

    assert loss[-1] < loss[0]
    assert len(loss) == epochs
    assert sum(ensemble_losses) == ensemble_size


@random_seed
def test_deep_ensemble_loss(
    independent_normal: bool,
    bootstrap_data: bool,
) -> None:
    example_data = _get_example_data([100, 1])

    loss = negative_log_likelihood
    optimizer = tf.optimizers.Adam()

    model = DeepEnsemble(
        trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, independent_normal),
        KerasOptimizer(optimizer, loss=loss),
        bootstrap_data,
    )

    reference_model = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, independent_normal)
    reference_model.model.compile(optimizer=optimizer, loss=loss)
    reference_model.model.set_weights(model.model.get_weights())

    tranformed_x, tranformed_y = _ensemblise_data(
        reference_model, example_data, _ENSEMBLE_SIZE, bootstrap_data
    )
    loss = model.model.evaluate(tranformed_x, tranformed_y)
    reference_loss = reference_model.model.evaluate(tranformed_x, tranformed_y)

    npt.assert_allclose(loss, reference_loss, rtol=1e-6)


@random_seed
def test_deep_ensemble_predict_ensemble(independent_normal: bool) -> None:
    example_data = _get_example_data([100, 1])

    loss = negative_log_likelihood
    optimizer = tf.optimizers.Adam()

    model = DeepEnsemble(
        trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, independent_normal),
        KerasOptimizer(optimizer, loss=loss),
    )

    reference_model = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, independent_normal)
    reference_model.model.compile(optimizer=optimizer, loss=loss)
    reference_model.model.set_weights(model.model.get_weights())

    predicted_means, predicted_vars = model.predict_ensemble(example_data.query_points)
    tranformed_x, tranformed_y = _ensemblise_data(
        reference_model, example_data, _ENSEMBLE_SIZE, False
    )
    ensemble_distributions = reference_model.model(tranformed_x)
    reference_means = tf.convert_to_tensor([dist.mean() for dist in ensemble_distributions])
    reference_vars = tf.convert_to_tensor([dist.variance() for dist in ensemble_distributions])

    npt.assert_allclose(predicted_means, reference_means)
    npt.assert_allclose(predicted_vars, reference_vars)


@random_seed
def test_deep_ensemble_sample(independent_normal: bool) -> None:
    example_data = _get_example_data([100, 1])
    model, _, _ = trieste_deep_ensemble_model(
        example_data, _ENSEMBLE_SIZE, False, independent_normal
    )
    num_samples = 100_000

    samples = model.sample(example_data.query_points, num_samples)
    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2, axis=0)

    ref_mean, ref_variance = model.predict(example_data.query_points)

    error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean, ref_mean, atol=4 * error)
    npt.assert_allclose(sample_variance, ref_variance, atol=8 * error)


@random_seed
def test_deep_ensemble_sample_ensemble(independent_normal: bool, ensemble_size: int) -> None:
    example_data = _get_example_data([20, 1])
    model, _, _ = trieste_deep_ensemble_model(
        example_data, ensemble_size, False, independent_normal
    )
    num_samples = 2000

    samples = model.sample_ensemble(example_data.query_points, num_samples)
    sample_mean = tf.reduce_mean(samples, axis=0)

    ref_mean, _ = model.predict(example_data.query_points)

    error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean, ref_mean, atol=2.5 * error)


@random_seed
def test_deep_ensemble_prepare_data_call(
    independent_normal: bool,
    ensemble_size: int,
    bootstrap_data: bool,
) -> None:
    n_rows = 100
    x = tf.constant(np.arange(0, n_rows, 1), shape=[n_rows, 1])
    y = tf.constant(np.arange(0, n_rows, 1), shape=[n_rows, 1])
    example_data = Dataset(x, y)

    model, _, _ = trieste_deep_ensemble_model(
        example_data, ensemble_size, bootstrap_data, independent_normal
    )

    # call with whole dataset
    data = model.prepare_dataset(example_data)
    assert isinstance(data, tuple)
    for ensemble_data in data:
        assert isinstance(ensemble_data, dict)
        assert len(ensemble_data.keys()) == ensemble_size
        for member_data in ensemble_data:
            if bootstrap_data:
                assert tf.reduce_any(ensemble_data[member_data] != x)
            else:
                assert tf.reduce_all(ensemble_data[member_data] == x)
    for inp, out in zip(data[0], data[1]):
        assert "".join(filter(str.isdigit, inp)) == "".join(filter(str.isdigit, out))

    # call with query points alone
    inputs = model.prepare_query_points(example_data.query_points)
    assert isinstance(inputs, dict)
    assert len(inputs.keys()) == ensemble_size
    for member_data in inputs:
        assert tf.reduce_all(inputs[member_data] == x)


@pytest.mark.mcdropout
@pytest.mark.parametrize("optimizer", [tf.optimizers.Adam(), tf.optimizers.RMSprop()])
def test_mcdropout_repr(
    optimizer: tf.optimizers.Optimizer,
    rate: List,
    loss: str,
) -> None:
    example_data = empty_dataset([1], [1])
    dropout_network = trieste_dropout_network_model(example_data, rate)
    dropout_network.compile(optimizer, loss=loss)
    optimizer_wrapper = KerasOptimizer(optimizer, loss=loss)
    model = MonteCarloDropout(dropout_network, optimizer_wrapper)

    expected_repr = f"MonteCarloDropout({dropout_network!r}, {optimizer_wrapper!r})"

    assert type(model).__name__ in repr(model)
    assert repr(model) == expected_repr


def test_dropout_network_model_attributes(rate: List) -> None:
    example_data = empty_dataset([1], [1])
    model, dropout_network, optimizer = trieste_mcdropout_model(example_data, rate=rate)

    dropout_network.compile(optimizer=optimizer.optimizer, loss=optimizer.loss)

    assert model.model is dropout_network


@pytest.mark.mcdropout
def test_dropout_network_default_optimizer_is_correct(rate: List, loss: str) -> None:
    example_data = empty_dataset([1], [1])

    dropout_network = trieste_dropout_network_model(example_data, rate)
    model = MonteCarloDropout(dropout_network)
    default_loss = loss
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


@pytest.mark.mcdropout
def test_mcdropout_optimizer_changed_correctly(rate: List) -> None:
    example_data = empty_dataset([1], [1])

    custom_fit_args = {
        "verbose": 1,
        "epochs": 10,
        "batch_size": 10,
    }
    custom_optimizer = tf.optimizers.RMSprop()
    custom_loss = negative_log_likelihood
    optimizer_wrapper = KerasOptimizer(custom_optimizer, custom_fit_args, custom_loss)

    dropout_network = trieste_dropout_network_model(example_data, rate)
    model = MonteCarloDropout(dropout_network, optimizer_wrapper)

    assert model.optimizer == optimizer_wrapper
    assert model.optimizer.optimizer == custom_optimizer
    assert model.optimizer.fit_args == custom_fit_args


@pytest.mark.mcdropout
def test_mcdropout_is_compiled(rate: List) -> None:
    example_data = empty_dataset([1], [1])
    model, _, _ = trieste_mcdropout_model(example_data, rate)

    assert model.model.compiled_loss is not None
    assert model.model.compiled_metrics is not None
    assert model.model.optimizer is not None


@pytest.mark.skip
def test_config_builds_mcdropout_and_default_optimizer_is_correct(rate: List) -> None:
    example_data = empty_dataset([1], [1])

    dropout_network = trieste_dropout_network_model(example_data, rate)

    model_config = {"model": dropout_network}
    model = create_model(model_config)
    default_fit_args = {
        "verbose": 0,
        "epochs": 100,
        "batch_size": 100,
    }

    assert isinstance(model, MonteCarloDropout)
    assert isinstance(model.optimizer, KerasOptimizer)
    assert isinstance(model.optimizer.optimizer, tf.keras.optimizers.Optimizer)
    assert model.optimizer.fit_args == default_fit_args


@pytest.mark.mcdropout
@pytest.mark.parametrize("dataset_size", [10, 100])
def test_mcdropout_predict_call_shape(dataset_size: int, rate: List) -> None:
    example_data = _get_example_data([dataset_size, 1])
    model, _, _ = trieste_mcdropout_model(example_data, rate)

    predicted_means, predicted_vars = model.predict(example_data.query_points)

    assert tf.is_tensor(predicted_vars)
    assert predicted_vars.shape == example_data.observations.shape
    assert tf.is_tensor(predicted_means)
    assert predicted_means.shape == example_data.observations.shape


@pytest.mark.mcdropout
@pytest.mark.parametrize("num_samples", [6, 12])
@pytest.mark.parametrize("dataset_size", [4, 8])
def test_mcdropout_sample_call_shape(num_samples: int, dataset_size: int, rate: List) -> None:
    example_data = _get_example_data([dataset_size, 1])
    model, _, _ = trieste_mcdropout_model(example_data, rate)

    samples = model.sample(example_data.query_points, num_samples)

    assert tf.is_tensor(samples)
    assert samples.shape == [num_samples, dataset_size, 1]


@random_seed
@pytest.mark.mcdropout
def test_mcdropout_optimize_with_defaults(rate: List) -> None:
    example_data = _get_example_data([100, 1])

    dropout_network = trieste_dropout_network_model(example_data, rate)

    model = MonteCarloDropout(dropout_network)

    model.optimize(example_data)
    loss = model.model.history.history["loss"]

    assert loss[-1] < loss[0]


@pytest.mark.mcdropout
def test_mcdropout_learning_rate_resets(rate: List) -> None:
    example_data = _get_example_data([100, 1])

    dropout_network = trieste_dropout_network_model(example_data, rate)

    model = MonteCarloDropout(dropout_network)

    model.optimize(example_data)
    lr1 = model.model.history.history["lr"]

    model.optimize(example_data)
    lr2 = model.model.history.history["lr"]

    assert lr1[0] == lr2[0]


@random_seed
@pytest.mark.mcdropout
def test_mcdropout_optimizer_learns_new_data(rate: List) -> None:

    positive_slope = _get_linear_data([20, 1], "pos")
    negative_slope = _get_linear_data([20, 1], "neg")
    new_data = positive_slope + negative_slope
    qp = tf.constant([[1.0]])

    dropout_network = trieste_dropout_network_model(positive_slope, rate, DropoutNetwork)

    model = MonteCarloDropout(dropout_network)

    model.optimize(positive_slope)
    pred1, _ = model.predict(qp)
    model.optimize(new_data)
    pred2, _ = model.predict(qp)

    assert np.abs(pred1 - pred2) > 1


@random_seed
@pytest.mark.mcdropout
@pytest.mark.parametrize("epochs", [5, 15])
@pytest.mark.parametrize("learning_rate", [0.1, 0.01])
@pytest.mark.parametrize("rate", [0.1, 0.2])
def test_mcdropout_optimize(rate: float, epochs: int, learning_rate: float) -> None:
    example_data = _get_example_data([20, 1])

    dropout_network = trieste_dropout_network_model(example_data, rate, DropoutNetwork)

    custom_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    custom_fit_args = {
        "verbose": 0,
        "epochs": epochs,
        "batch_size": 10,
        "callbacks": [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=80, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.3, patience=15),
        ],
    }
    optimizer_wrapper = KerasOptimizer(custom_optimizer, custom_fit_args)

    model = MonteCarloDropout(
        dropout_network, optimizer=optimizer_wrapper, learning_rate=learning_rate
    )

    model.optimize(example_data)

    lr_hist = model.model.history.history["lr"]
    loss = model.model.history.history["loss"]
    # breakpoint()
    assert loss[-1] < loss[0]
    assert len(loss) == epochs
    npt.assert_almost_equal(lr_hist[0], learning_rate, decimal=3)


@random_seed
@pytest.mark.mcdropout
@pytest.mark.parametrize("layer", [DropoutNetwork, DropConnectNetwork])
def test_mcdropout_loss_with_different_layers_and_reset_lr(rate: List, loss: str, layer) -> None:
    example_data_1 = _get_example_data([200, 1])
    example_data_2 = _get_example_data([200, 1])

    model_1 = MonteCarloDropout(
        trieste_dropout_network_model(example_data_1, rate, layer),
        KerasOptimizer(tf.optimizers.Adam(), loss=loss),
        learning_rate=0.01,
    )
    model_1.optimize(example_data_1)

    model_2 = MonteCarloDropout(
        trieste_dropout_network_model(example_data_2, rate, layer),
        KerasOptimizer(tf.optimizers.Adam(), loss=loss),
        learning_rate=0.01,
    )
    model_2.optimize(example_data_2)

    loss_1 = model_1.model.evaluate(example_data_1.astuple()[0], example_data_1.astuple()[1])
    loss_2 = model_2.model.evaluate(example_data_2.astuple()[0], example_data_2.astuple()[1])

    npt.assert_almost_equal(loss_1, loss_2, decimal=3)


@random_seed
@pytest.mark.mcdropout
@pytest.mark.parametrize("num_samples", [50, 100, 200])
@pytest.mark.parametrize("num_passes", [50, 100, 200])
def test_mcdropout_predict_num_passes(rate: List, loss: str, num_samples, num_passes) -> None:
    example_data = _get_example_data([100, 1])
    transformed_x, transformed_y = example_data.astuple()

    model = MonteCarloDropout(
        trieste_dropout_network_model(example_data, rate, DropoutNetwork),
        KerasOptimizer(tf.optimizers.Adam(), loss=loss),
        learning_rate=0.01,
        num_passes=num_passes,
    )
    model.optimize(example_data)

    reference_model = trieste_dropout_network_model(example_data, rate, DropoutNetwork)
    reference_model.compile(optimizer=tf.optimizers.Adam(), loss=loss)
    reference_model.fit(transformed_x, transformed_y)
    reference_model.set_weights(model.model.get_weights())

    sample_means = tf.reduce_mean(
        model.sample(example_data.query_points, num_samples=num_samples), axis=0
    )
    predicted_means, _ = model.predict(example_data.query_points)
    reference_means = reference_model(example_data.query_points)

    npt.assert_allclose(predicted_means, sample_means, atol=0.1)
    npt.assert_allclose(predicted_means, reference_means, atol=0.1)


@random_seed
@pytest.mark.mcdropout
@pytest.mark.parametrize("num_samples", [1000, 5000, 10000])
def test_mcdropout_sample(rate: List, num_samples) -> None:
    example_data = _get_example_data([100, 1])
    model, _, _ = trieste_mcdropout_model(example_data, rate)

    samples = model.sample(example_data.query_points, num_samples)
    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2, axis=0)

    ref_mean, ref_variance = model.predict(example_data.query_points)

    error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))

    npt.assert_allclose(sample_mean, ref_mean, atol=4 * error)
    npt.assert_allclose(sample_variance, ref_variance, atol=8 * error)
