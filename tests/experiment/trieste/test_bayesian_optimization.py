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

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Type, cast

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import random_seed
from trieste.acquisition import (
    AcquisitionFunctionClass,
    ParallelContinuousThompsonSampling,
)
from trieste.acquisition.rule import (
    AcquisitionRule,
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
)
from trieste.bayesian_optimizer import (
    BayesianOptimizer,
    FrozenRecord,
    TrainableProbabilisticModelType,
    stop_at_minimum,
)
from trieste.logging import tensorboard_writer
from trieste.models import TrainableProbabilisticModel, TrajectoryFunctionClass
from trieste.models.optimizer import KerasOptimizer
from trieste.objectives import ScaledBranin, SimpleQuadratic
from trieste.objectives.utils import mk_observer
from trieste.space import SearchSpace
from trieste.types import TensorType

from experiment.trieste import TriesteMonteCarloDropout, build_montecarlo_dropout


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        pytest.param(5, EfficientGlobalOptimization(), id="EfficientGlobalOptimization"),
        pytest.param(5, DiscreteThompsonSampling(500, 1), id="DiscreteThompsonSampling"),
        pytest.param(
            5,
            EfficientGlobalOptimization(
                ParallelContinuousThompsonSampling(),
                num_query_points=3,
            ),
            id="ParallelContinuousThompsonSampling",
            marks=pytest.mark.skip(reason="sampler not ready yet"),
        ),
    ],
)
def test_bayesian_optimizer_with_mcdropout_finds_minima_of_simple_quadratic(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, TriesteMonteCarloDropout],
) -> None:
    _test_optimizer_finds_minimum(TriesteMonteCarloDropout, num_steps, acquisition_rule)


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        pytest.param(90, EfficientGlobalOptimization(), id="EfficientGlobalOptimization"),
        pytest.param(30, DiscreteThompsonSampling(500, 3), id="DiscreteThompsonSampling"),
        pytest.param(
            30,
            EfficientGlobalOptimization(
                ParallelContinuousThompsonSampling(),
                num_query_points=3,
            ),
            id="ParallelContinuousThompsonSampling",
            marks=pytest.mark.skip(reason="sampler not ready yet"),
        ),
    ],
)
def test_bayesian_optimizer_with_mcdropout_finds_minima_of_scaled_branin(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, TriesteMonteCarloDropout],
) -> None:
    _test_optimizer_finds_minimum(
        TriesteMonteCarloDropout, num_steps, acquisition_rule, optimize_branin=True
    )


def _test_optimizer_finds_minimum(
    model_type: Type[TrainableProbabilisticModelType],
    num_steps: Optional[int],
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, TrainableProbabilisticModelType],
    optimize_branin: bool = False,
    model_args: Optional[Mapping[str, Any]] = None,
    check_regret: bool = False,
) -> None:
    model_args = model_args or {}

    if optimize_branin:
        search_space = ScaledBranin.search_space
        minimizers = ScaledBranin.minimizers
        minima = ScaledBranin.minimum
        rtol_level = 0.005
    else:
        search_space = SimpleQuadratic.search_space
        minimizers = SimpleQuadratic.minimizers
        minima = SimpleQuadratic.minimum
        rtol_level = 0.05

    num_initial_query_points = 20

    initial_query_points = search_space.sample(num_initial_query_points)
    observer = mk_observer(ScaledBranin.objective if optimize_branin else SimpleQuadratic.objective)
    initial_data = observer(initial_query_points)

    model: TrainableProbabilisticModel  # (really TPMType, but that's too complicated for mypy)

    if model_type is TriesteMonteCarloDropout:
        dropout_network = build_montecarlo_dropout(initial_data, rate=0.1)
        fit_args = {
            "batch_size": 32,
            "epochs": 1000,
            "callbacks": [
                tf.keras.callbacks.EarlyStopping(monitor="loss", patience=80),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.3, patience=15),
            ],
            "verbose": 0,
        }
        mcd_optimizer = KerasOptimizer(tf.keras.optimizers.Adam(0.001), fit_args)
        model = TriesteMonteCarloDropout(
            dropout_network, mcd_optimizer, num_passes=200, **model_args
        )

    else:
        raise ValueError(f"Unsupported model_type '{model_type}'")

    with tempfile.TemporaryDirectory() as tmpdirname:
        summary_writer = tf.summary.create_file_writer(tmpdirname)
        with tensorboard_writer(summary_writer):

            result = BayesianOptimizer(observer, search_space).optimize(
                num_steps or 2,
                initial_data,
                cast(TrainableProbabilisticModelType, model),
                acquisition_rule,
                track_state=False,
                track_path=Path(tmpdirname) / "history",
                early_stop_callback=stop_at_minimum(minima, minimizers, minimum_rtol=rtol_level),
            )

            if num_steps is None:
                # this test is just being run to check for crashes, not performance
                pass
            elif check_regret:
                # this just check that the new observations are mostly better than the initial ones
                assert isinstance(result.history[0], FrozenRecord)
                initial_observations = result.history[0].load().dataset.observations
                best_initial = tf.math.reduce_min(initial_observations)
                better_than_initial = 0
                num_points = len(initial_observations)
                for i in range(1, len(result.history)):
                    step_history = result.history[i]
                    assert isinstance(step_history, FrozenRecord)
                    step_observations = step_history.load().dataset.observations
                    new_observations = step_observations[num_points:]
                    if tf.math.reduce_min(new_observations) < best_initial:
                        better_than_initial += 1
                    num_points = len(step_observations)

                assert better_than_initial / len(result.history) > 0.6
            else:
                # this actually checks that we solved the problem
                best_x, best_y, _ = result.try_get_optimal_point()
                minimizer_err = tf.abs((best_x - minimizers) / minimizers)
                assert tf.reduce_any(tf.reduce_all(minimizer_err < 0.05, axis=-1), axis=0)
                npt.assert_allclose(best_y, minima, rtol=rtol_level)

            if isinstance(acquisition_rule, EfficientGlobalOptimization):
                acq_function = acquisition_rule.acquisition_function
                assert acq_function is not None

                # check that acquisition functions defined as classes aren't retraced unnecessarily
                # they should be retraced for the optimzier's starting grid, L-BFGS, and logging
                if isinstance(acq_function, (AcquisitionFunctionClass, TrajectoryFunctionClass)):
                    assert acq_function.__call__._get_tracing_count() == 3  # type: ignore
