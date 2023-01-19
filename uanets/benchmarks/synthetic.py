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

from __future__ import annotations

import math
from dataclasses import dataclass
from math import pi
from typing import Callable, Sequence

import tensorflow as tf

from .interface import Benchmark


@make_unsplit_array_dataset_factory(
    tags={REGRESSION, TINY, SYNTHETIC},
    test_fraction=0.5,
    normalise=False,
)
def tiny_linear(rng: np.random.Generator) -> Tuple[AnyNDArray, AnyNDArray]:
    X = rng.random((20, 1))
    F = 0.3 * X - 0.1
    noise = rng.normal(0.0, 0.1, F.shape)
    Y = F + noise
    return X, Y


@make_unsplit_array_dataset_factory(
    tags={REGRESSION, TINY, SYNTHETIC},
    test_fraction=0.5,
    normalise=False,
)
def tiny_sine(rng: np.random.Generator) -> Tuple[AnyNDArray, AnyNDArray]:
    X = rng.random((20, 1))
    F = np.sin(10 * X)
    noise = rng.normal(0.0, 0.1, F.shape)
    Y = F + noise
    return X, Y


# We will use a simple one-dimensional toy problem introduced by <cite data-cite="hernandez2015probabilistic"/>, which was used in <cite data-cite="lakshminarayanan2016simple"/> to provide some illustrative evidence that deep ensembles do a good job of estimating uncertainty. We will replicate this exercise here.
#
# The toy problem is a simple cubic function with some Normally distributed noise around it. We will randomly sample 20 input points from [-4,4] interval that we will use as a training data later on.


# %%
from trieste.space import Box
from trieste.data import Dataset


def objective(x, error=True):
    y = tf.pow(x, 3)
    if error:
        y += tf.random.normal(x.shape, 0, 3, dtype=x.dtype)
    return y


num_points = 20

# we define the [-4,4] interval using a `Box` search space that has convenient sampling methods
search_space = Box([-4], [4])
inputs = search_space.sample_sobol(num_points)
outputs = objective(inputs)
data = Dataset(inputs, outputs)


# %% [markdown]
# Next we define a deep ensemble model and train it. Trieste supports neural network models defined as TensorFlow's Keras models. Since creating ensemble models in Keras can be somewhat involved, Trieste provides some basic architectures. Here we use the `build_keras_ensemble` function which builds a simple ensemble of neural networks in Keras where each network has the same architecture: number of hidden layers, nodes in hidden layers and activation function. It uses sensible defaults for many parameters and finally returns a model of `KerasEnsemble` class.
#
# As with other supported types of models (e.g. Gaussian process models from GPflow), we cannot use `KerasEnsemble` directly in Bayesian optimization routines, we need to pass it through an appropriate wrapper, `DeepEnsemble` wrapper in this case. One difference with respect to other model types is that we need to use a Keras specific optimizer wrapper `KerasOptimizer` where we need to specify a stochastic optimizer (Adam is used by default, but we can use other stochastic optimizers from TensorFlow), objective function (here negative log likelihood) and we can provide custom arguments for the Keras `fit` method (here we modify the default arguments; check [Keras API documentation](https://keras.io/api/models/model_training_apis/#fit-method) for a list of possible arguments).
#
# For the cubic function toy problem we use the same architecture as in <cite data-cite="lakshminarayanan2016simple"/>: ensemble size of 5 networks, where each network has one hidden layer with 100 nodes. All other implementation details were missing and we used sensible choices, as well as details about training the network.
