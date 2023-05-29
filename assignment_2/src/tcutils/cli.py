"""cli.py

CLI handling utilities for text classification.
"""

import logging
import typing as t
import argparse
from pathlib import Path
from tcutils import __version__
from tcutils.data import load_csv_data, save_model_report
from tcutils.modelling import ClassificationModel


def common_args(parser: argparse.ArgumentParser):
    """Add common arguments to an argument parser.

    Args:
        parser (argparse.ArgumentParser): an argument parser

    Returns:
        argparse.ArgumentParser: the argument parser
    """
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Path to the CSV file containing the data",
        default=Path("in", "fake_or_real_news.csv")
    )
    parser.add_argument(
        '-m',
        '--model-save-path',
        help="Path to save the trained model(s)",
        type=Path,
        default=Path("models")
    )
    parser.add_argument(
        '-r',
        '--report-path',
        help="Path to save the classification report(s)",
        type=Path,
        default=Path("out")
    )
    parser.add_argument(
        '-v',
        '--vectorizer',
        help="Vectorizer to use",
        type=str,
        default="tfidf",
        choices=["tfidf", "count"]
    )

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


def run(classifier: t.Type[ClassificationModel]):
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    data = load_csv_data(args.file)
    clf = classifier(
        data=data,
        vectorizer=args.vectorizer)
    train_results, test_results = clf.run()
    clf.save(args.model_save_path)
    save_model_report(
        args.report_path,
        classifier.__name__,
        type(clf.vectorizer).__name__,
        train_results,
        test_results,
        clf.model_summary(),
        clf.vectorizer_summary()
    )
