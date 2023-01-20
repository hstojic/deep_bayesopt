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

from typing import Optional

import tensorflow as tf
from trieste.data import Dataset
from trieste.models.interfaces import TrainableProbabilisticModel
from trieste.models.keras.interface import KerasPredictor
from trieste.models.optimizer import KerasOptimizer
from trieste.types import TensorType

from uanets.models.mc_dropout import DropoutNetwork


class MonteCarloDropout(KerasPredictor, TrainableProbabilisticModel):
    """
    A :class:`~trieste.model.TrainableProbabilisticModel` wrapper for Monte Carlo dropout
    built using Keras.

    Monte Carlo dropout is a sampling method for approximate Bayesian computation, mathematically
    equivalent to an approximation to a probabilistic deep Gaussian Process <cite data-cite="gal2016dropout"/>
    in the sense of minimizing the Kullback-Leibler divergence between an approximate distribution
    and the posterior of a deep GP. This model is attractive due to its simplicity, as it amounts
    to a re-tooling of the dropout layers of a neural network to also be active during testing,
    and performing several forward passes through the network with the same input data. The
    resulting distribution of the outputs of the different passes are then used to estimate the
    first two moments of the predictive distribution. Note that increasing the number of passes
    increases accuracy at the cost of a higher computational burden.

    The uncertainty estimations of the original paper have been subject to extensive scrutiny, and
    it has been pointed out that the quality of the uncertainty estimates is tied to parameter
    choices which need to be calibrated to accurately account for model uncertainty. A more robust
    alternative is MC-DropConnect, an approach that generalizes the prior idea by applying dropout
    not to the layer outputs but directly to each weight (see <cite data-cite="mobiny2019"/>).

    We provide classes for constructing neural networks with Monte Carlo dropout using Keras
    (:class:`~trieste.models.keras.DropoutNetwork`) in the `architectures` package that should be
    used with the :class:`~trieste.models.keras.MonteCarloDropout` wrapper. There we also provide
    an application of MC-DropConnect, by setting the argument `dropout` to 'dropconnect'.

    Note that currently we do not support setting up the model with dictionary configs and saving
    the model during Bayesian optimization loop (``track_state`` argument in
    :meth:`~trieste.bayesian_optimizer.BayesianOptimizer.optimize` method should be set to `False`).
    """

    def __init__(
        self,
        model: DropoutNetwork,
        optimizer: Optional[KerasOptimizer] = None,
        num_passes: int = 100,
    ) -> None:
        """
        :param model: A Keras neural network model with Monte Carlo dropout layers. The
            model has to be built but not compiled.
        :param optimizer: The optimizer wrapper with necessary specifications for compiling and
            training the model. Defaults to :class:`~trieste.models.optimizer.KerasOptimizer` with
            :class:`~tf.optimizers.Adam` optimizer, mean square error loss and a dictionary
            of default arguments for Keras `fit` method: 1000 epochs, batch size 16, early stopping
            callback with patience of 50, and verbose 0.
            See https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
            arguments.
        :raise ValueError: If ``model`` is not an instance of
            :class:`~trieste.models.keras.DropoutNetwork`.
        """
        super().__init__(optimizer)

        if not self.optimizer.fit_args:
            self.optimizer.fit_args = {
                "verbose": 0,
                "epochs": 1000,
                "batch_size": 32,
                "callbacks": [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="loss", patience=100, restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.3, patience=20),
                ],
            }

        if self.optimizer.loss is None:
            self.optimizer.loss = "mse"

        self._learning_rate = self.optimizer.optimizer.learning_rate.numpy()

        model.compile(
            self.optimizer.optimizer,
            loss=[self.optimizer.loss],
            metrics=[self.optimizer.metrics],
        )

        self.num_passes = num_passes
        self._model = model

    def __repr__(self) -> str:
        """"""
        return f"MonteCarloDropout({self.model!r}, {self.optimizer!r})"

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the underlying Keras ensemble model with the specified ``dataset``.

        Optimization is performed by using the Keras `fit` method, rather than applying the
        optimizer and using the batches supplied with the optimizer wrapper. User can pass
        arguments to the `fit` method through ``minimize_args`` argument in the optimizer wrapper.
        These default to using 1000 epochs, batch size 100, and verbose 0 with an early stopping
        callback using a patience of 100 epochs and a learning rate scheduler . See
        https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
        arguments.

        Note that optimization does not return the result, instead optimization results are
        stored in a history attribute of the model object.

        :param dataset: The data with which to optimize the model.
        """
        x, y = dataset.astuple()
        self.model.fit(x=x, y=y, **self.optimizer.fit_args)
        self.optimizer.optimizer.learning_rate.assign(self._learning_rate)

    def update(self, dataset: Dataset) -> None:
        """
        Neural networks are parametric models and do not need to update data.
        `TrainableProbabilisticModel` interface, however, requires an update method, so
        here we simply pass the execution.
        """
        pass

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return ``num_samples`` samples at ``query_points``. We use the stochastic forward passes
        to simulate ``num_samples`` samples for each point of ``query_points`` points.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples, with shape [..., S, N].
        """
        return tf.stack(
            [self.model(query_points, training=True) for _ in range(num_samples)], axis=0
        )

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        r"""
        Returns mean and variance of the Monte Carlo Dropout.

        Following <cite data-cite="gal2015simple"/>, we make T stochastic forward passes
        through the trained network of L hidden layers M_l and average the results to derive
        the mean and variance. These are respectively given by

        .. math:: \mathbb{E}_{q\left(\mathbf{y}^{*} \mid \mathbf{x}^{*}\right)}
            \left(\mathbf{y}^{*}\right) \approx \frac{1}{T} \sum_{t=1}^{T}
            \widehat{\mathbf{y}}^{*}\left(\mathrm{x}^{*}, \widehat{\mathbf{M}}_{1}^{t},
            \ldots, \widehat{\mathbf{M}}_{L}^{t}\right)

        .. math:: \frac{1}{T} \operatorname{Var}_{q\left(\mathbf{y}^{*} \mid \mathbf{x}^{*}\right)}
            \left(\mathbf{y}^{*}\right) \approx\sum_{t=1}^{T} \widehat{\mathbf{y}}^{*}\left(\mathbf{x}^{*},
            \widehat{\mathbf{M}}_{1}^{t}, \ldots, \widehat{\mathbf{M}}_{L}^{t}\right)^{T}
            \widehat{\mathbf{y}}^{*}\left(\mathbf{x}^{*}, \widehat{\mathbf{M}}_{1}^{t}, \ldots,
            \widehat{\mathbf{M}}_{L}^{t}\right)-\mathbb{E}_{q\left(\mathbf{y}^{*} \mid
            \mathbf{x}^{*}\right)}\left(\mathbf{y}^{*}\right)^{T} \mathbb{E}_{q\left(\mathbf{y}^{*}
            \mid \mathbf{x}^{*}\right)}\left(\mathbf{y}^{*}\right)

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the observations at the specified
            ``query_points``.
        """

        stochastic_passes = tf.stack(
            [self.model(query_points, training=True) for _ in range(self.num_passes)], axis=0
        )
        predicted_means = tf.math.reduce_mean(stochastic_passes, axis=0)

        predicted_vars = tf.subtract(
            tf.divide(
                tf.reduce_sum(tf.math.multiply(stochastic_passes, stochastic_passes), axis=0),
                self.num_passes,
            ),
            tf.math.square(predicted_means),
        )
        return predicted_means, predicted_vars
