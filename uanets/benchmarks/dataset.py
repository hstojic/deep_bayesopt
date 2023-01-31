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

import pandas as pd

MOTORCYCLE_DATA_URL = (
    "https://github.com/secondmind-labs/GPflux/blob/develop/docs/notebooks/data/motor.csv"
)


def motorcycle_data():
    """
    The data comes from a motorcycle accident simulation (see ref below) and shows some
    interesting behaviour, in particular the heteroscedastic nature of the noise.

    Silverman, B. W. (1985) "Some aspects of the spline smoothing approach to non-parametric curve
    fitting". Journal of the Royal Statistical Society, series B 47, 1-52.

    Return inputs and outputs for the motorcycle dataset. We normalise the outputs.
    """
    df = pd.read_csv("./data/motor.csv", index_col=0)
    X, Y = df["times"].values.reshape(-1, 1), df["accel"].values.reshape(-1, 1)
    Y = (Y - Y.mean()) / Y.std()
    X /= X.max()
    return X, Y
