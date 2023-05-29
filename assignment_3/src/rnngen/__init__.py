__version__ = "0.1.0"

import os
import tensorflow as tf
from tensorflow.python.framework.ops import enable_eager_execution


# session: tf.compat.v1.Session


def optimize_tf_config(num_threads, gpu=True):
    """Optimize TensorFlow config for parallelism."""

    # global session
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], 'GPU')
    # disable_eager_execution()
    enable_eager_execution()

    os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
    os.environ["TF_NUM_INTRAOP_THREADS"] = f"{num_threads}"
    os.environ["TF_NUM_INTEROP_THREADS"] = f"{num_threads}"

    tf.config.threading.set_inter_op_parallelism_threads(
        num_threads
    )
    tf.config.threading.set_intra_op_parallelism_threads(
        num_threads
    )
    tf.config.set_soft_device_placement(True)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # type: ignore
    # session = tf.compat.v1.Session(config=config)  # noqa
    # tf.compat.v1.keras.backend.set_session(session)
    # return session