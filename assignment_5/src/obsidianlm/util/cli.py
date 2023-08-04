import os
# this prevents OOM (on my GPU with 8GB VRAM)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200"
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
        default=2
    )
    parser.add_argument(
        '-V',
        '--vault-path',
        help="Path to your obsidian vault.",
        type=Path,
        default=Path("vault")
    )
    parser.add_argument(
        '-r',
        '--rouge',
        help="Calculate rouge scores.",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '-e',
        '--do-eval',
        help="Run evaluation during training.",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '-i',
        '--inference',
        help="Run a test inference on the specified model checkpoint.",
        type=Path,
        default=None
    )
    parser.add_argument(
        '-p',
        '--prompt',
        help="Prompt to use for generation.",
        type=str,
        default=None,
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
        model, tokenizer, preprocess_fn = obsidianT5.get_model_from_checkpoint(
            checkpoint_path=args.inference,
            max_len_in=512,
            max_len_out=512,

        )

        sample_inputs = [
            "Question: Which models are used in this course? Context: This course wraps up the series of methods courses. We look at modelling from a birds-eye view, introducing advanced concepts and revisiting methods introduced in earlier courses from a comprehensive Bayesian perspective. We introduce causal reasoning using directed acyclic graphs (DAGs), mixture models, Gaussian processes, learn to deal with measurement error and missing data; and we revisit regression modelling, generalized linear models, multilevel modelling, Markov chain Monte Carlo sampling, learning to implement them using probabilistic programming.",
            "summarize: Climate change is a subject that garners significant media attention, especially in the wake of natural disasters [@comfortIgnoredBannerStory2019; @weinerClimateChangeCoverage2021]. While there is ample evidence of the effects of human greenhouse gas emissions on global climate, including some natural disasters such as droughts, extreme temperaturs and flooding, the direct link to hurricane, forest-fire and earthquake frequency is more difficult to establish. Regardless of the causality of disaster events, the continued increase in global population means more people are at risk of being affected when they occur. On this basis, this paper will investigate aspects of government spending as well as population statistics as potential predictors for the number of affected individuals and number of deaths.",
            "Hello, my dog is cute",
        ]

        if args.prompt:
            sample_inputs = [args.prompt]

        for sample_input in sample_inputs:
            tokenized = tokenizer(sample_input, return_tensors="pt")
            outputs = model.generate(input_ids=tokenized['input_ids'], min_length=0, max_length=512, num_beams=5, early_stopping=True)
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Input: {sample_input}")
            print(f"Output: {decoded_output}")

        return
    model, tokenizer, preprocess_fn = obsidianT5.get_model(
        max_len_in=512,
        max_len_out=512,
    )

    train_ds, val_ds = data.get_datasets(
        source_dir=args.vault_path,
        preprocess_fn=preprocess_fn,
        validation_split=0.96 if args.do_eval else 0.99999,
        mask_lm=True,
    )

    logging.info(f"Train dataloader: {train_ds}")
    print_gpu_utilization()
    obsidianT5.train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds if args.do_eval else None,
        batch_size=args.batch_size,
        out_path=args.output_path,
        rouge=args.rouge,
        mask_lm=True,
        max_len_gen=512,
        eval_batch_size=1,
    )

    print_gpu_utilization()
