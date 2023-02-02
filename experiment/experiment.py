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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from trieste.objectives import (
    Michalewicz2,
    Michalewicz5,
    Michalewicz10,
    Hartmann6,
    Rosenbrock4,
    Ackley5,
    Shekel4,
)
from trieste.objectives.utils import mk_observer
import trieste
import pandas as pd
import time

from experiment.models import (
    build_dgp, build_svgp, build_gpr, build_deep_ensemble, build_mc_dropout
)
from experiment.acquisition import build_discrete_ts, build_random, build_cont_ts, build_ei
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('output_filename', type=str, help='output filename', nargs='?', default='test')
parser.add_argument('--function', type=str, help='objective function', nargs='?', default='michalewicz2')
parser.add_argument('--model', type=str, help='model name', nargs='?', default='deepgp')
parser.add_argument('--run', type=int, help='run number', nargs='?', default=0)
args = parser.parse_args()

function_key = args.function
model_key = args.model
run = args.run

np.random.seed(run)
tf.random.set_seed(run)

function_dict = {
    "michalewicz2": Michalewicz2,
    "michalewicz5": Michalewicz5,
    "michalewicz10": Michalewicz10,
    "hartmann": Hartmann6,
    "ackley": Ackley5,
    "rosenbrock": Rosenbrock4,
    "shekel": Shekel4,
}


model_dict = {
    "deepgp": build_dgp,
    "svgp": build_svgp,
    "gpr": build_gpr,
    "de": build_deep_ensemble,
    "mcdrop": build_mc_dropout,
}

acq_dict = {
    "discrete_ts": build_discrete_ts,
    "cont_ts": build_cont_ts,
    "random": build_random,
    "ei": build_ei,
}


if not os.path.exists(os.path.join('results', function_key)):
    os.makedirs(os.path.join('results', function_key))

pd.DataFrame({
    'function': [function_key],
    'model': [model_key],
    'run': [run]
}).to_csv(args.output_filename)

function = function_dict[function_key].objective
F_MINIMIZER = function_dict[function_key].minimizers

search_space = function_dict[function_key].search_space
observer = mk_observer(function)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_initial_points = 20
num_acquisitions = 480
batch_size = 50

split_size = None
num_init_points = None

def run_experiment(
    model_key: str,
    acq_key: str,
    initial_data: Dataset,
    search_space: SearchSpace,
):
    start_time = time.time()

    model_builder = model_dict[model_key]
    model = model_builder(initial_data, search_space)

    acq_builder = model_dict[acq_key]
    acquisition_rule = acq_builder(batch_size, )

    result = bo.optimize(num_acquisitions, initial_data, model,
                             acquisition_rule=acquisition_rule, track_state=False)

    result_dataset = result.try_get_final_dataset()
    result_query_points = result_dataset.query_points.numpy()
    result_observations = result_dataset.observations.numpy()
    result_arg_min_idx = tf.squeeze(tf.argmin(result_observations, axis=0))

    pd.DataFrame(result_query_points).to_csv(
        'results/{}/{}_query_points_{}'.format(function_key, model_key, run))
    pd.DataFrame(result_observations).to_csv(
        'results/{}/{}_observations_{}'.format(function_key, model_key, run))

    print(f"{model_key} observation "
          f"{function_key} {run}: {result_observations[result_arg_min_idx, :]}")
    print("Time: ", time.time() - start_time)


initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)


run_bayes_opt(model_key, initial_data, search_space)
