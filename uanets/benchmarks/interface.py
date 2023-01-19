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
This module contains interfaces that synthetic and dataset based benchmarks have to implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from dataclasses import dataclass
import tensorflow as tf


@dataclass
class Benchmark(ABC):
    """
    This class defines a benchmark object and all the methods that need to be implemented.
    """

    name: str
    """The benchmark name"""

    @abstractmethod
    def data(self, num_samples: Optional[int], seed: Optional[int]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Returns data in the form of input and output tensors, with ``num_samples`` observations if argument is used.

        :param num_samples: The number of samples to take for generating the data. In case of
            synthetic benchmarks these might be random draws, while for datasets if used it would
            result in sampling ``num_samples`` from the dataset.
        :param seed: The seed for random samples, if used.
        :return: Tuple consisting of input tensor and output tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def input_tensor_spec(self) -> tf.TensorSpec:
        """Returns tensor specifications for inputs."""
        raise NotImplementedError

    @abstractmethod
    def output_tensor_spec(self) -> tf.TensorSpec:
        """Returns tensor specifications for outputs."""
        raise NotImplementedError
