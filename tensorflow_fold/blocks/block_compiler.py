# Copyright (C) 2018 Seoul National University
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
# ==============================================================================

import argparse
import sys
import os
import json
import time

from absl import flags
import tensorflow as tf

import benchmark_cnn
import cnn_util
import parallax_config
from cnn_util import log_fn
from tensorflow.core.protobuf import config_pb2

import parallax

benchmark_cnn.define_flags()
flags.adopt_module_key_flags(benchmark_cnn)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('resource_info_file',
                           os.path.abspath(os.path.join(
                               os.path.dirname(__file__),
                               '.',
                               'resource_info')),
                           'Filename containing cluster information')
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of iterations to run for each workers.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How many steps between two runop logs.""")
tf.app.flags.DEFINE_boolean('sync', True, '')

def main(_):
    # Build single-GPU benchmark_cnn model
    with tf.Graph().as_default() as single_gpu_graph:
        ds = input_files_dataset()
        ds = parallax.shard(ds)
        images, labels = get_input_data(ds)
        
        loss = build_InceptionV3_model(images, labels)
        
        opt = tf.train.GradientDescentOptimizer(lr=0.1)
        train_op = opt.minimize(loss)

    def run(sess, num_iters, tensor_or_op_name_to_replica_names,
            num_workers, worker_id, num_replicas_per_worker):
        fetches = {
            'global_step':
                tensor_or_op_name_to_replica_names[bench.global_step.name][0],
            'cost': tensor_or_op_name_to_replica_names[bench.cost.name][0],
            'train_op':
                tensor_or_op_name_to_replica_names[bench.train_op.name][0],
        }
        
        for i in range(num_iters):
            results = sess.run(fetches)

    parallax.parallel_run(
        single_gpu_graph,
        run,
        FLAGS.resource_info_file,
        FLAGS.max_steps,
        sync=FLAGS.sync)


if __name__ == '__main__':
    tf.app.run()
