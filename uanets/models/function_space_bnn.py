# Copyright 2023 The deep_bayesopt Authors. All Rights Reserved.
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

import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
import tensorflow_probability as tfp


class FunctionSpaceBNN(tf.keras.Model):
    """
    A keras Bayesian Neural Network implementation with a custom loss that performs inference in the
    function space.
    """

    def __init__(
        self,
        prior: tf.keras.Sequential,
        posterior: tf.keras.Sequential,
        likelihood: tf.keras.Sequential,
        mean_function: tf.keras.Sequential,
        num_data: int,
    ) -> None:
        super().__init__()
        self.prior = prior
        self.posterior = posterior
        self.likelihood = likelihood
        self.mean_function = mean_function
        self.num_data = num_data

    def call(self, inputs):
        x = self.posterior(inputs)
        x += self.mean_function(inputs)
        return self.likelihood(x)

    def _custom_loss(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:

        inducing_points = tf.random.uniform(
            minval=-2.0, maxval=2.0, shape=x.shape, dtype=self.dtype
        )

        posterior_samples = tf.stack([self.posterior(inducing_points) for _ in range(20)], axis=0)
        variational_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.reduce_mean(posterior_samples, axis=0),
            scale_diag=1e-6 + tf.math.reduce_std(posterior_samples, axis=0),
        )
        prior_kl = tf.reduce_sum(
            tfp.distributions.kl_divergence(variational_distribution, self.prior(inducing_points))
        )

        likelihood = tf.reduce_mean(tf.stack([self(x).log_prob(y) for _ in range(20)], axis=0))

        return -likelihood * self.num_data + prior_kl

    def train_step(self, data: tf.Tensor):
        data = data_adapter.expand_1d(data)
        x, y, _ = data_adapter.unpack_x_y_sample_weight(data)

        # forward pass
        with tf.GradientTape() as tape:
            loss = self._custom_loss(x, y)

        # backwards pass
        self.optimizer.minimize(
            loss,
            self.posterior.trainable_variables
            + self.mean_function.trainable_variables
            + self.likelihood.trainable_variables,
            tape=tape,
        )
        return {"loss": loss}

    def test_step(self, data: tf.Tensor):
        data = data_adapter.expand_1d(data)
        x, y, _ = data_adapter.unpack_x_y_sample_weight(data)
        loss = self._custom_loss(x, y)
        return {"val_loss": loss}
