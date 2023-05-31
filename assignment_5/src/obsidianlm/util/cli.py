import logging
import typing as t
import argparse
from pathlib import Path
from .. import __version__
from ..model import obsidianT5
from . import data
from .sys import print_gpu_utilization


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
        '-o',
        '--output-path',
        help="Path to save the output, figures, stats, etc.",
        type=Path,
        default=Path("out")
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        help="Batch size for training.",
        type=int,
        default=4
    )
    parser.add_argument(
        '-V',
        '--vault-path',
        help="Path to your obsidian vault.",
        type=Path,
        default=Path("vault")
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
        description="ObsidianLM: Create a model of your brain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    if extra_arg_func:
        parser = extra_arg_func(parser)

    return common_args(parser).parse_args(args)


def run():
    logging.basicConfig(level=logging.INFO)
    print_gpu_utilization()
    args = parse_args()

    # for item in dataloader:
    #     print(item)
    #     exit()
    model, tokenizer, preprocess_fn = obsidianT5.get_model(
        max_len_in=512,
        max_len_out=512,
    )

    train_ds, val_ds = data.get_datasets(
        source_dir=args.vault_path,
        preprocess_fn=preprocess_fn,
        validation_split=0.9,
        mask_lm=True,
    )

    logging.info(f"Train dataloader: {train_ds}")
    print_gpu_utilization()
    obsidianT5.train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        batch_size=args.batch_size,
        out_path=args.output_path,
        rouge=False,
        mask_lm=True,
    )

    print_gpu_utilization()
