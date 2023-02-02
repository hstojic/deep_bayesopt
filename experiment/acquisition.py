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

from typing import Optional

import trieste
from trieste.acquisition.optimizer import (
    automatic_optimizer_selector,
    generate_continuous_optimizer,
    AcquisitionOptimizer,
)
from trieste.acquisition.rule import (
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
    RandomSampling,
)
from trieste.acquisition import ExpectedImprovement, ParallelContinuousThompsonSampling

from trieste.acquisition.utils import split_acquisition_function_calls
from trieste.observer import OBJECTIVE

def _get_af_optimizer(split_size: Optional[int],
    num_init_points: Optional[int],
) -> AcquisitionOptimizer:
    optimizer = automatic_optimizer_selector
    if num_init_points is not None:
        optimizer = generate_continuous_optimizer(num_init_points)  # type: ignore
    if split_size is not None:
        optimizer = split_acquisition_function_calls(  # type: ignore
            optimizer, split_size=split_size
        )
    return optimizer


def build_random(batch_size: int, split_size: Optional[int],
    num_init_points: Optional[int],) -> RandomSampling:
    return RandomSampling(num_query_points=batch_size)


def build_discrete_ts(batch_size: int, split_size: Optional[int],
    num_init_points: Optional[int],) -> DiscreteThompsonSampling:
    return DiscreteThompsonSampling(
        num_search_space_samples=5000, num_query_points=batch_size,)

def build_cont_ts(batch_size: int, split_size: Optional[int],
    num_init_points: Optional[int],) -> ParallelContinuousThompsonSampling:
    optimizer = _get_af_optimizer(split_size, num_init_points)
    return EfficientGlobalOptimization(
        optimizer=optimizer,
        num_query_points=batch_size,
        builder=ParallelContinuousThompsonSampling().using(OBJECTIVE),
    )

def build_ei(batch_size: int, split_size: Optional[int],
    num_init_points: Optional[int],) -> EfficientGlobalOptimization:
    optimizer = _get_af_optimizer(split_size, num_init_points)
    return EfficientGlobalOptimization(
        optimizer=optimizer,
        num_query_points=batch_size,
        builder=ExpectedImprovement().using(OBJECTIVE),
    )
