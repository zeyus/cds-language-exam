"""cli.py

CLI handling utilities.
"""

import logging
import typing as t
import argparse
from pathlib import Path
from .. import optimize_tf_config
from ..models import (
    rnn_model_context,
    save_best_callback,
    StabilityCallback,
    perplexity,
    generate_text_by_word,
    generate_text_sequence,
)
from .data import load_nyt_data
from .stats import (
    plot_history,
    generate_save_basename,
    save_classification_report,
    EpochTrackerCallback
)
from .. import __version__
# from . import set_tf_optim
# from ..model import get_model, save_best_callback, get_model_resnet
import tensorflow as tf
import numpy as np


def common_args(parser: argparse.ArgumentParser):
    """Add common arguments to an argument parser.

    Args:
        parser (argparse.ArgumentParser): an argument parser

    Returns:
        argparse.ArgumentParser: the argument parser
    """
    parser.add_argument(
        "task",
        help="The task to perform",
        choices=["train", "predict"],
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    parser.add_argument(
        '-s',
        '--model-save-path',
        help="Path to save the trained model(s)",
        type=Path,
        default=Path("models")
    )
    parser.add_argument(
        '-d',
        '--dataset-path',
        help="Path to the dataset",
        type=Path,
        default=Path("data/nyt_comments")
    )

    parser.add_argument(
        '-b',
        '--batch-size',
        help="The batch size",
        type=int,
        default=64
    )
    parser.add_argument(
        '-e',
        '--epochs',
        help="The number of epochs",
        type=int,
        default=10
    )
    parser.add_argument(
        '-o',
        '--out',
        help="The output path for the plots and stats",
        type=Path,
        default=Path("out"))

    parser.add_argument('-c',
                        '--from-checkpoint',
                        help="Use the checkpoint at the given path",
                        type=Path,
                        default=None)
    parser.add_argument('-p',
                        '--parallel',
                        help="Number of workers/threads for processing.",
                        type=int,
                        default=4)

    parser.add_argument('-t',
                        '--temperature',
                        help="Temperature for sampling during prediction. (1.0 is deterministic)",
                        type=float,
                        default=0.8)

    parser.add_argument('-n',
                        '--top-n',
                        help="Top N for sampling during sequence-to-sequence prediction. (1 is equivalent to argmax)",
                        type=int,
                        default=1)
    
    parser.add_argument('-m',
                        '--min-length',
                        help="Minimum length of generated text (in tokens, not characters).",
                        type=int,
                        default=0)

    # add free arg for prediction string
    parser.add_argument('prediction_string', nargs='?', default=None)

    return parser


def parse_args(
                args: t.Optional[t.List[str]] = None,
                extra_arg_func: t.Optional[t.Callable[
                    [argparse.ArgumentParser],
                    argparse.ArgumentParser]] = None) -> argparse.Namespace:
    """Parse the command line arguments.

    Args:
        args (t.Optional[t.List[str]], optional):
            Arguments to parse. Defaults to None.
        extra_arg_func (t.Optional[t.Callable[
            [argparse.ArgumentParser],
            argparse.ArgumentParser]], optional):
                Function to add extra arguments. Defaults to None.
    """
    parser = argparse.ArgumentParser(
        description="Text classification CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    if extra_arg_func:
        parser = extra_arg_func(parser)

    return common_args(parser).parse_args(args)


def run():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if not args.dataset_path.exists():
        raise FileNotFoundError(
            "Dataset path does not exist. "
            "Please download the dataset first.")
    optimize_tf_config(args.parallel)

    if args.task == "predict":
        if not args.from_checkpoint:
            default_checkpoint = Path("models/rnn/20230501_212656_rnn_2000x300_batch64_iter10.best.h5")
            logging.warning(f"No checkpoint provided. Using {default_checkpoint}")
            args.from_checkpoint = default_checkpoint
        if not args.prediction_string:
            logging.warning("No prediction string provided. Using latent space.")
            args.prediction_string = ""
        # args.batch_size = 1

    logging.info("Version: %s", __version__)
    logging.info("Dataset path: %s", args.dataset_path)
    logging.info("Model save path: %s", args.model_save_path)
    logging.info("Running...")
    max_tokens: int = 2000
    max_length: int = 300

    if args.task == "train":
        train_data, val_data, encoder = load_nyt_data(
            args.dataset_path,
            max_length=max_length,
            max_tokens=max_tokens,
            batch_size=args.batch_size,
            validation_frac=0.1)
        for example, label in train_data.take(1):
            logging.info("Example: %s", example[0])
            logging.info("Label: %s", label[0])
    else:
        _, _, encoder = load_nyt_data(
            args.dataset_path,
            max_length=max_length,
            max_tokens=max_tokens,
            batch_size=args.batch_size,
            validation_frac=0.1)

    input_shape = (max_tokens, max_length)

    model_save_path = args.model_save_path / "rnn"

    if not model_save_path.exists():
        model_save_path.mkdir(parents=True)

    if not args.out.exists():
        args.out.mkdir(parents=True)

    out_file_basename = generate_save_basename(
        args.out,
        "rnn",
        input_shape,
        args.batch_size,
        args.epochs,
    )

    model_save_basename = generate_save_basename(
        model_save_path,
        "rnn",
        input_shape,
        args.batch_size,
        args.epochs,
    )

    model: tf.keras.Model
    if args.from_checkpoint:
        logging.info("Loading model from checkpoint: %s", args.from_checkpoint)
        model_metric = {"perplexity": perplexity}
        with tf.keras.utils.custom_object_scope(model_metric):
            model = tf.keras.models.load_model(args.from_checkpoint)
        if args.task == "predict":
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        # model = tf.keras.models.load_model(args.from_checkpoint, compile=False)
        # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        # metrics = tf.metrics.Accuracy()
        # optimizer = tf.keras.optimizers.experimental.SGD(0.1, momentum=0.9)
        # model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    else:
        model = rnn_model_context(
            encoder=encoder,
            embedding_dim=256,
            rnn_units=512,
            max_length=max_length,
            batch_size=args.batch_size)

    if args.task == "train":
        tf.keras.utils.plot_model(
            model,
            to_file=out_file_basename.with_suffix(".png"),
            show_shapes=True,
            show_dtype=True,
            expand_nested=True,
            dpi=160)
        model_save_filename = model_save_basename.with_suffix(".h5")
        model_best_filename = model_save_basename.with_suffix(".best.h5")
        epoch_tracker = EpochTrackerCallback()
        stability_callback = StabilityCallback()

        H = None
        try:
            # # vectorize y as well
            # @tf.function(jit_compile=True)
            # def test_fix_ds(x: tf.Tensor, y: tf.Tensor):
            #     # Split x and y into word-level tensors
            #     x_words: tf.RaggedTensor = tf.strings.split(x, sep=' ')
            #     y_words: tf.RaggedTensor = tf.strings.split(y, sep=' ')

            #     y = y_words[:, -1:]
            #     # now make y a normal string tensor (we know the shape is (batch_size, 1))
            #     y = tf.strings.reduce_join(y, axis=1)

            #     print(y.numpy())
            #     exit()
            # logging.info("Pre-encoding data...")
            vocab_size = tf.cast(encoder.vocabulary_size(), tf.int32)
            train_data = train_data.map(lambda x, y: (
                tf.cast(
                    encoder(x),
                    tf.int16),
                tf.cast(
                    tf.one_hot(
                        tf.cast(
                            encoder(y),
                            tf.int32),
                        vocab_size,
                        dtype=tf.int32),
                    tf.int16)
                )
            )
            val_data = val_data.map(lambda x, y: (
                tf.cast(
                    encoder(x),
                    tf.int16),
                tf.cast(
                    tf.one_hot(
                        tf.cast(
                            encoder(y),
                            tf.int32),
                        vocab_size,
                        dtype=tf.int32),
                    tf.int16)
                )
            )
            # print an example input and label from the encoded dataset
            for example, label in train_data.take(1):
                logging.info("Example: %s", example[0])
                logging.info("Label: %s", label[0])
                # print shapes
                logging.info("Example shape: %s", example.shape)
                logging.info("Label shape: %s", label.shape)
            logging.info("Training...")
            H = model.fit(
                train_data,
                epochs=args.epochs,
                validation_data=val_data,
                callbacks=[
                    # stability_callback,
                    tf.keras.callbacks.BackupAndRestore(backup_dir=model_save_path / "backup", save_freq=1000),  # backup every 1000 steps
                    save_best_callback(model_best_filename),
                    epoch_tracker],
                verbose=1,
                workers=args.parallel,
                use_multiprocessing=True if args.parallel > 1 else False)
        except KeyboardInterrupt:
            logging.info(
                f"Training interrupted. Current Epoch: {epoch_tracker.EPOCH}")
            if H is None:
                H = model.history  # type: ignore
        if epoch_tracker.EPOCH > 0:
            plot_history(H, epoch_tracker.EPOCH, out_file_basename)
        logging.info(model.summary())
        logging.info(f"Saving model to {model_save_filename}...")
        model.save(model_save_filename)
        logging.info("Done.")
        exit()

    # predict from prediction string
    logging.info("Prompt string: %s", args.prediction_string)

    # make the prediction string a tensor
    # ds = tf.data.Dataset.from_tensor_slices([args.prediction_string])

    # encode the prediction string
    # x = encoder([args.prediction_string])
    logging.info("Generating sequence-to-sequence prediction...")
    seq_result = generate_text_sequence(model, encoder, args.prediction_string, temperature=args.temperature, min_length=args.min_length, max_n=args.top_n)
    logging.info("Done")
    logging.info("Generating word-by-word prediction...")
    word_result = generate_text_by_word(model, encoder, args.prediction_string, max_length, temperature=0.5)
    logging.info("Done")
    logging.info("Sequence to sequence result: \n%s\n", seq_result)
    logging.info("Word by word result: \n%s\n", word_result)

    # logging.info("Encoded Prompt: %s", x)
    # logging.info("Encoded Prompt shape: %s", x.shape)

    # # logging.info("Encoded prediction: %s", encoded_prediction)
    # # logging.info("Encoded prediction shape: %s", encoded_prediction.shape)

    # # predict
    # prediction = model.predict(
    #     x,
    #     verbose=1,
    #     workers=args.parallel,
    #     use_multiprocessing=True if args.parallel > 1 else False)
    # prediction = prediction[0]
    # logging.info("Prediction: %s", prediction)
    # logging.info("Prediction shape: %s", prediction.shape)

    # # get the last prediction

    # # decode prediction
    # # create stringlookup for decoding
    # reverse_encoder = tf.keras.layers.experimental.preprocessing.StringLookup(
    #     vocabulary=encoder.get_vocabulary(include_special_tokens=False),
    #     invert=True
    # )
    # decoded_prediction = reverse_encoder(tf.argmax(prediction, axis=-1))
    # logging.info("Decoded prediction: %s", decoded_prediction)
