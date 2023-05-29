import logging
import typing as t
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from .. import __version__
from .data import get_news_dataset
from .viz import visualize_data
from ..model import get_model_and_tokenizer, inference


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
        '-d',
        '--dataset-path',
        help="Path to the dataset",
        type=Path,
        default=Path("data")
    )
    parser.add_argument(
        '-V',
        '--visualize-data',
        help="Visualize the dataset",
        action="store_true"
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


def run():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    out_file = args.output_path / "news_emotions.csv"

    if args.visualize_data:
        if not out_file.exists():
            raise FileNotFoundError(f"File {out_file} not found, run without --visualize-data first")
        visualize_data(out_file, args.output_path)
        return

    model, tokenizer = get_model_and_tokenizer()
    dataset = get_news_dataset(
        args.dataset_path / "fake_or_real_news.csv",
        tokenizer)
    df_output = pd.DataFrame(columns=["text", "emotion", "label"])
    for item in tqdm(dataset):
        probs = inference(model, item, tokenizer)
        # get class name
        class_name = model.config.id2label[probs.argmax().item()]  # type: ignore
        # logging.info(f"Text: {item['text']} | Emotion: {class_name} | Label: {item['label']}")  # type: ignore
        df_output = pd.concat([
            df_output,
            pd.DataFrame(
                [[item['text'], class_name, item['label']]],  # type: ignore
                columns=["text", "emotion", "label"]
            )
        ])
    logging.info(f"Writing output to {out_file}")
    df_output.to_csv(out_file, index=False)
    logging.info("Done")
