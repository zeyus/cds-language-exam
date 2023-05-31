from pathlib import Path
import re
import typing as t
from datasets import Value, Features, Dataset, DatasetDict
import os
import random


def load_markdown_files(source_dir: Path) -> t.List[str]:
    """Load all markdown files from a directory.

    Args:
        source_dir (Path): directory to load from

    Returns:
        t.List[str]: list of markdown strings
    """
    files = list(source_dir.glob("**/*.md"))
    files = [str(f) for f in files]
    return files


def prepare_file_metadata(file: str) -> t.Union[t.Dict[str, t.Any], None]:
    """Prepare metadata for a file.

    Args:
        file (str): path to a file

    Returns:
        t.Dict[str, str]: metadata for the file
    """
    if not Path(file).exists():
        return None
    metadata = {}
    metadata["path"] = file
    pfile = Path(file)
    metadata["name"] = pfile.name.replace(".md", "")
    metadata["parent"] = pfile.parent.name
    metadata["tags"] = []
    metadata["links"] = []
    metadata["contents"] = pfile.read_text()
    return metadata


def get_tags(contents: str) -> t.List[str]:
    """Get tags from markdown contents. Tags are defined as ' #tag '

    Args:
        contents (str): markdown contents

    Returns:
        t.List[str]: list of tags
    """
    tags = re.findall(r"\s#(\w+)\s", contents)
    return tags


def get_links(contents: str) -> t.List[t.Dict[str, str]]:
    """Get links from markdown.

    Links are defined as
        - external [link title](link)
        - internal [[link]]

    Args:
        contents (str): markdown contents

    Returns:
        t.List[t.Dict[str, str]]: list of links
    """
    links = []
    # external links
    external_links = re.findall(r"\[(.*?)\]\((.*?)\)", contents)
    for link in external_links:
        links.append({"title": link[0], "link": link[1], "type": "external"})
    # internal links
    internal_links = re.findall(r"\[\[(.*?)\]\]", contents)
    for link in internal_links:
        links.append({"title": link, "link": link, "type": "internal"})

    return links


def parse_markdown_files(
        files: t.List[str],
        shuffle: bool = False,
        mask_lm: bool = True) -> t.Dict:
    """Parse markdown files into a list of strings.

    Args:
        files (t.List[str]): list of markdown files

    Returns:
        t.Dict: dictionary of parsed files for loading into a Dataset
    """
    if shuffle:
        random.shuffle(files)

    data = {
        "idx": [],
        "text": [],
    }
    if not mask_lm:
        data["context"] = []
        data["summary"] = []
    for i, file in enumerate(files):
        metadata = prepare_file_metadata(file)
        if metadata is None:
            continue
        metadata["tags"] = get_tags(metadata["contents"])
        metadata["links"] = get_links(metadata["contents"])
        context = f"Name: {metadata['name']}\n"
        context += f"Parent: {metadata['parent']}\n"
        context += f"Tags: {', '.join(metadata['tags'])}\n"
        context += "Links: "
        context = ', '.join([link['title'] for link in metadata['links']])
        context += "\n"
        text = metadata["contents"]
        summary = text.split("\n")[0]
        data["idx"].append(i)
        data["text"].append(text)
        if not mask_lm:
            data["context"].append(context)
            data["summary"].append(summary)

    return data


def get_dataset(
        files: t.List[str],
        preprocess_fn: t.Callable[[t.Any], t.Dict],
        nproc: t.Optional[int] = None,
        mask_lm: bool = True) -> Dataset:
    """Get a dataset from a directory of markdown files.

    Args:
        source_dir (Path): directory to load from
        preprocess_fn (t.Callable[[t.Any], t.Dict]): preprocessing function

    Returns:
        IterableDataset: dataset
    """

    ds_feature_spec = {
        "idx": Value("int32", id=None),
        "text": Value("string", id=None),
    }
    if not mask_lm:
        ds_feature_spec["context"] = Value("string", id=None)
        ds_feature_spec["summary"] = Value("string", id=None)
    ds_features = Features(ds_feature_spec)
    ds = Dataset.from_dict(
        parse_markdown_files(files, mask_lm=mask_lm),
        features=ds_features,
    )

    ds = ds.map(
        preprocess_fn,
        batched=False,
        keep_in_memory=True,
        num_proc=nproc)
    return ds.with_format("torch")


def get_datasets(
        source_dir: Path,
        preprocess_fn: t.Callable[[t.Any], t.Dict],
        validation_split: float = 0.9,
        mask_lm: bool = True) -> t.Tuple[Dataset, Dataset]:
    """Create datasets from a directory of markdown files.

    Args:
        source_dir (Path): directory to load from
        preprocess_fn (t.Callable[[t.Any], t.Dict]): preprocessing function
        validation_split (float, optional): validation split. Defaults to 0.8.

    Returns:
        t.Tuple[Dataset, Dataset]: train and validation datasets
    """
    files = load_markdown_files(source_dir)

    n_logical_cores = os.cpu_count()
    ds = get_dataset(
        files=files,
        preprocess_fn=preprocess_fn,
        nproc=n_logical_cores,
        mask_lm=mask_lm
    )

    ds = ds.train_test_split(
        train_size=validation_split,
        keep_in_memory=True)

    return ds["train"], ds["test"]
