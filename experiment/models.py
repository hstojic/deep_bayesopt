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

import gpflow
import tensorflow as tf

from trieste.models import build_vanilla_deep_gp, DeepGaussianProcess, build_gpr, GaussianProcessRegression, build_keras_ensemble, DeepEnsemble, Optimizer, KerasOptimizer, build_svgp, SparseVariational


from trieste.data import Dataset
from trieste.space import SearchSpace

from experiment.trieste import TriesteMonteCarloDropout, build_montecarlo_dropout



def build_dgp(data: Dataset, search_space: SearchSpace) -> DeepGaussianProcess:
    """"""
    num_layers=2
    num_inducing=200
    trainable_likelihood = False
    likelihood_variance = 1e-3

    model = build_vanilla_deep_gp(data, search_space, num_layers=num_layers, num_inducing=num_inducing, likelihood_variance = likelihood_variance, trainable_likelihood=trainable_likelihood)
   
    epochs = 400
    batch_size = 1000

    def scheduler(epoch: int, lr: float) -> float:
        if epoch == epochs // 2:
            return lr * 0.1
        else:
            return lr

    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "callbacks": [tf.keras.callbacks.LearningRateScheduler(scheduler)]
        "verbose": 0,
        "shuffle": False,
    }
    optimizer = KerasOptimizer(tf.optimizers.Adam(0.005), fit_args)

    return DeepGaussianProcess(
        model = model,
        optimizer = optimizer,
        num_rff_features = 1000,
        continuous_optimisation = True,
    )


def build_svgp(data: Dataset, search_space: SearchSpace) -> SparseVariational:
    """"""
    num_inducing_points=200
    trainable_likelihood = False
    likelihood_variance = None

    model = build_svgp(data, search_space, likelihood_variance = likelihood_variance, trainable_likelihood=trainable_likelihood, num_inducing_points=num_inducing_points)
    optimizer = Optimizer(gpflow.optimizers.Scipy(), {"options": dict(maxiter=1000)}, compile=True)

    return SparseVariational(
        model = model,
        optimizer=optimizer,
        num_rff_features = 1000,
        inducing_point_selector=ConditionalImprovementReduction(),
    )


def build_gpr(data: Dataset, search_space: SearchSpace) -> GaussianProcessRegression:
    """"""
    gpflow.config.set_default_jitter(1e-5)
    trainable_likelihood = False
    likelihood_variance = None

    model = build_gpr(data, search_space, likelihood_variance = likelihood_variance, trainable_likelihood=trainable_likelihood)
    optimizer = Optimizer(gpflow.optimizers.Scipy(), {"options": dict(maxiter=1000)}, compile=True)

    return GaussianProcessRegression(
        model = model,
        optimizer=optimizer,
        num_kernel_samples = 100,
        num_rff_features = 1000,
        use_decoupled_sampler = True,
    )


def build_deep_ensemble(data: Dataset, search_space: SearchSpace) -> DeepEnsemble:
    """"""
    model = build_keras_ensemble(data, 5, 3, 25, "selu")
    fit_args = {
        "batch_size": 20,
        "epochs": 200,
        "callbacks": [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=25, restore_best_weights=True
            )
        ],
        "verbose": 0,
    }
    optimizer = KerasOptimizer(tf.keras.optimizers.Adam(0.01), fit_args)

    return DeepEnsemble(
        model = model,
        optimizer=optimizer,
        bootstrap = True,
        diversify = False,
        continuous_optimisation = True,
    )


def build_mc_dropout(data: Dataset, search_space: SearchSpace) -> TriesteMonteCarloDropout:
    """"""
    num_hidden_layers = 5
    units = 300
    activation = "relu"

    model = build_montecarlo_dropout(data, num_hidden_layers, units, activation, rate)
    optimizer = KerasOptimizer(tf.keras.optimizers.Adam(0.01), fit_args={
        "batch_size": 20,
        "epochs": 200,
        "callbacks": [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=25, restore_best_weights=True
            )
        ],
        "verbose": 0,
    })

    return TriesteMonteCarloDropout(
        model = model,
        optimizer=optimizer,
        num_passes = 100,
        continuous_optimisation = True,
    )
