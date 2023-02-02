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


def test_ll_vanilla_dgp(
    data: Dataset,
    model: trieste.models.gpflux.models.DeepGaussianProcess,
    num_samples: int = 100
) -> TensorType:
    samples = []
    for _ in range(num_samples):
        out = model.model_gpflux.call(data.query_points)
        y_mean, y_var = out.y_mean, out.y_var
        l = norm.logpdf(data.observations.numpy(), loc=y_mean, scale=y_var**0.5)
        samples.append(l)
    samples = np.stack(samples)
    ind_ll = tf.reduce_logsumexp(samples, axis=0) - math.log(num_samples)
    return tf.reduce_mean(ind_ll, axis=0).numpy()


def test_ll_gp(
    data: Dataset,
    model: trieste.models.gpflow.GPflowPredictor
) -> TensorType:
    return tf.reduce_mean(model.model.predict_log_density(
        (data.query_points, data.observations)
    ), keepdims=True).numpy()
