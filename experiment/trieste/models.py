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

from typing import Optional, Tuple

import tensorflow as tf
from trieste import logging
from trieste.data import Dataset
from trieste.models.interfaces import TrainableProbabilisticModel
from trieste.models.keras.interface import KerasPredictor
from trieste.models.optimizer import KerasOptimizer
from trieste.types import TensorType
from trieste.models.utils import write_summary_data_based_metrics

from unflow.models import MonteCarloDropout


class TriesteMonteCarloDropout(KerasPredictor, TrainableProbabilisticModel):
    """
    A :class:`~trieste.model.TrainableProbabilisticModel` wrapper for Monte Carlo dropout
    built using Keras.

    Monte Carlo dropout is a sampling method for approximate Bayesian computation, mathematically
    equivalent to an approximation to a probabilistic deep Gaussian Process
    <cite data-cite="gal2016dropout"/>
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
    alternative might be DropConnect, an approach that generalizes the prior idea by applying
    dropout not to the layer outputs but directly to each weight
    # (see <cite data-cite="mobiny2019"/>).

    A word of caution in case a learning rate scheduler is used in ``fit_args`` to
    :class:`KerasOptimizer` optimizer instance. Typically one would not want to continue with the
    reduced learning rate in the subsequent Bayesian optimization step. Hence, we reset the
    learning rate to the original one after calling the ``fit`` method. In case this is not the
    behaviour you would like, you will need to subclass the model and overwrite the
    :meth:`optimize` method.
    """

    def __init__(
        self,
        model: MonteCarloDropout,
        optimizer: Optional[KerasOptimizer] = None,
        num_passes: int = 100,
        continuous_optimisation: bool = True,
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
        :param continuous_optimisation: If True (default), the optimizer will keep track of the
            number of epochs across BO iterations and use this number as initial_epoch. This is
            essential to allow monitoring of model training across BO iterations.
        :raise ValueError: If ``model`` is not an instance of
            :class:`~unflow.models.MonteCarloDropout`.
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

        if self.optimizer.metrics is None:
            self.optimizer.metrics = ["mse"]

        model.compile(
            self.optimizer.optimizer,
            loss=[self.optimizer.loss],
            metrics=[self.optimizer.metrics],
        )

        self.original_lr = self.optimizer.optimizer.lr.numpy()
        self._absolute_epochs = 0
        self._continuous_optimisation = continuous_optimisation

        self._num_passes = num_passes
        self._model = model

    def __repr__(self) -> str:
        """"""
        return (
            f"MonteCarloDropout({self.model!r}, {self.optimizer!r}, "
            f"{self._num_passes!r}, {self._continuous_optimisation!r})"
        )

    @property
    def model(self) -> tf.keras.Model:
        """ " Returns compiled Keras model."""
        return self._model

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return ``num_samples`` samples at ``query_points``. We use the stochastic forward passes
        to simulate ``num_samples`` samples for each point of ``query_points`` points.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples, with shape [..., S, N, P], with S = num_samples and
            P is the number of outputs.
        """
        return self._model.sample(x=query_points, num_samples=num_samples)

    def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
        r"""
        Returns mean and variance of the predictions, based on ``num_passes`` forward passes.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the observations at the specified
            ``query_points``.
        """
        return self._model.predict_mean_and_var(x=query_points, num_samples=self._num_passes)

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the underlying Keras model with the specified ``dataset``.
        Optimization is performed by using the Keras `fit` method, rather than applying the
        optimizer and using the batches supplied with the optimizer wrapper. User can pass
        arguments to the `fit` method through ``minimize_args`` argument in the optimizer wrapper.
        See https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
        arguments.

        Note that optimization does not return the result, instead optimization results are
        stored in a history attribute of the model object.

        :param dataset: The data with which to optimize the model.
        """
        fit_args = dict(self.optimizer.fit_args)

        # Tell optimizer how many epochs have been used before: the optimizer will "continue"
        # optimization across multiple BO iterations rather than start fresh at each iteration.
        # This allows us to monitor training across iterations.

        if "epochs" in fit_args:
            fit_args["epochs"] = fit_args["epochs"] + self._absolute_epochs

        x, y = dataset.astuple()
        history = self.model.fit(
            x=x,
            y=y,
            **fit_args,
            initial_epoch=self._absolute_epochs,
        )
        if self._continuous_optimisation:
            self._absolute_epochs = self._absolute_epochs + len(history.history["loss"])

        # Reset lr in case there was an lr schedule: a schedule will have change the learning rate,
        # so that the next time we call `optimize` the starting learning rate would be different.
        # Therefore, we make sure the learning rate is set back to its initial value.
        self.optimizer.optimizer.lr.assign(self.original_lr)

    def update(self, dataset: Dataset) -> None:
        """
        Neural networks are parametric models and do not need to update data.
        `TrainableProbabilisticModel` interface, however, requires an update method, so
        here we simply pass the execution.
        """
        pass

    def log(self, dataset: Optional[Dataset] = None) -> None:
        """
        Log model training information at a given optimization step to the Tensorboard.
        We log several summary statistics of losses and metrics given in ``fit_args`` to
        ``optimizer`` (final, difference between inital and final loss, min and max). We also log
        epoch statistics, but as histograms, rather than time series. We also log several training
        data based metrics, such as root mean square error between predictions and observations,
        and several others.

        For custom logs user will need to subclass the model and overwrite this method.

        :param dataset: Optional data that can be used to log additional data-based model
            summaries.
        """
        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                logging.scalar("epochs/num_epochs", len(self.model.history.epoch))
                for k, v in self.model.history.history.items():
                    logging.histogram(f"{k}/epoch", lambda: v)
                    logging.scalar(f"{k}/final", lambda: v[-1])
                    logging.scalar(f"{k}/diff", lambda: v[0] - v[-1])
                    logging.scalar(f"{k}/min", lambda: tf.reduce_min(v))
                    logging.scalar(f"{k}/max", lambda: tf.reduce_max(v))
                if dataset:
                    write_summary_data_based_metrics(
                        dataset=dataset, model=self, prefix="training_"
                    )
