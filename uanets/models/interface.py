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

from abc import ABC, abstractmethod
from typing import Optional

import tensorflow as tf
from check_shapes import check_shapes

from uanets.types import MeanAndVariance


class ApproximateBayesianModel(tf.keras.Model, ABC):
    """
    This is an interface that prescribes the methods that all models need to implement.
    """

    @abstractmethod
    @check_shapes(
        "x: [batch..., N, D]",
        "return[0]: [batch..., N, P]",
        "return[1]: [batch..., N, P]",
    )
    def predict_mean_and_var(
        self, x: tf.Tensor, num_samples: Optional[int] = None, seed: Optional[int] = None
    ) -> MeanAndVariance:
        """
        Return the mean and variance of the independent marginal distributions at each point in
        ``x``.

        Some models will not have an analytical expression for computing the moments and might
        need to sample to compute them, in these cases ``num_samples`` and ``seed`` can be used.

        Models implementing this function should decorate the method with
        :meth:`~check_shapes.inherit_check_shapes`, so that check_shapes specified above are
        inherited.

        :param x:
            Input locations at which to compute mean and variance.
        :param num_samples:
            Some models might need to sample to compute mean and the variance. In these
            cases this argument can be used to set the number of samples based on which mean
            and variance will be computed.
        :param seed:
            Some models might need to sample to compute mean and the variance. In these cases
            seed can be fixed to produce deterministic results.
        :return: The mean and variance of the independent marginal distributions at each point in
            ``x``.
        """
        raise NotImplementedError

    @abstractmethod
    @check_shapes(
        "x: [batch..., N, D]",
        "return: [batch..., 1, N, P] if (num_samples is None)",
        "return: [batch..., S, N, P] if (num_samples is not None)",
    )
    def sample(
        self,
        x: tf.Tensor,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> tf.Tensor:
        """
        Return ``num_samples`` samples from the independent marginal distributions at ``x``.

        Models implementing this function should decorate the method with
        :meth:`~check_shapes.inherit_check_shapes`, so that check_shapes specified above are
        inherited.

        :param x:
            Input locations at which to draw samples.
        :param num_samples:
            Number of samples to draw.
            If `None`, a single sample is drawn and the return shape is [..., 1, N, P],
            for any positive integer the return shape is [..., S, N, P], with S = num_samples and
            P is the number of outputs.
        :param seed:
            Set the seed to produce deterministic results.
        """
        raise NotImplementedError
