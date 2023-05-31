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
        default=1
    )
    parser.add_argument(
        '-V',
        '--vault-path',
        help="Path to your obsidian vault.",
        type=Path,
        default=Path("vault")
    )
    parser.add_argument(
        '-i',
        '--inference',
        help="Run a test inference on the specified model checkpoint.",
        type=Path,
        default=None
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
    args = parse_args()
    print_gpu_utilization()

    if args.inference:
        obsidianT5.get_model_from_checkpoint(
            checkpoint_path=args.inference,
            max_len_in=512,
            max_len_out=512
        )

        sample_inputs = [
            "What is the meaning of life?",
            "Cognitive <extra_id_0> is the study of <extra_id_1>.",
            "summarize: Climate change is a subject that garners significant media attention, especially in the wake of natural disasters [@comfortIgnoredBannerStory2019; @weinerClimateChangeCoverage2021]. While there is ample evidence of the effects of human greenhouse gas emissions on global climate, including some natural disasters such as droughts, extreme temperaturs and flooding, the direct link to hurricane, forest-fire and earthquake frequency is more difficult to establish. Regardless of the causality of disaster events, the continued increase in global population means more people are at risk of being affected when they occur. On this basis, this paper will investigate aspects of government spending as well as population statistics as potential predictors for the number of affected individuals and number of deaths.",
            "Eye tracking is used for <extra_id_0> and <extra_id_1>.",
            "Replication bias is a <extra_id_0> in <extra_id_1>.",
            "This is a random sentence."
        ]

        for sample_input in sample_inputs:
            input_ids = tokenizer(sample_input, return_tensors="pt", padding=True).input_ids
            sequence_ids = model.generate(input_ids)
            sequences = tokenizer.batch_decode(sequence_ids)
            print(f"Input: {sample_input}")
            print(f"Output: {sequences}")



    model, tokenizer, preprocess_fn = obsidianT5.get_model(
        max_len_in=512,
        max_len_out=512,
    )

    train_ds, val_ds = data.get_datasets(
        source_dir=args.vault_path,
        preprocess_fn=preprocess_fn,
        validation_split=0.97,
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
