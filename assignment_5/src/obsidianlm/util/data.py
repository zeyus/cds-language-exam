from pathlib import Path
import re
import typing as t
from datasets import Value, Features, Dataset
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
        shuffle: bool = True) -> t.Dict:
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
        "context": [],
        "text": [],
        "summary": []
    }
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
        data["context"].append(context)
        data["text"].append(text)
        data["summary"].append(summary)
    return data


def get_dataset(
        files: t.List[str],
        preprocess_fn: t.Callable[[t.Any], t.Dict],
        nproc: t.Optional[int] = None) -> Dataset:
    """Get a dataset from a directory of markdown files.

    Args:
        source_dir (Path): directory to load from
        preprocess_fn (t.Callable[[t.Any], t.Dict]): preprocessing function

    Returns:
        IterableDataset: dataset
    """
    ds_features = Features({
        "idx": Value("int32", id=None),
        "context": Value("string", id=None),
        "text": Value("string", id=None),
        "summary": Value("string", id=None),
    })
    ds = Dataset.from_dict(
        parse_markdown_files(files),
        features=ds_features,
    )

    ds = ds.map(
        preprocess_fn,
        batched=False,
        keep_in_memory=True,
        num_proc=nproc)
    return ds.with_format("torch")


def get_dataloader(
        source_dir: Path,
        preprocess_fn: t.Callable[[t.Any], t.Dict]) -> t.Tuple[Dataset, int]:
    """Get a dataloader from a directory of markdown files.

    Args:
        source_dir (Path): directory to load from
        preprocess_fn (t.Callable[[t.Any], t.Dict]): preprocessing function
        batch_size (int, optional): batch size. Defaults to 1.

    Returns:
        DataLoader: dataloader
        int: number of files loaded
    """
    files = load_markdown_files(source_dir)
    ds_length = len(files)
    n_logical_cores = os.cpu_count()
    ds = get_dataset(files, preprocess_fn, n_logical_cores)
    # ds.__class__.__len__ = MethodType(lambda self: ds_length, ds)
    print(len(ds))

    # dl = DataLoader(
    #     ds,  # type: ignore
    #     batch_size=batch_size,
    #     pin_memory=True,
    #     shuffle=False,
    #     num_workers=n_logical_cores if n_logical_cores else 1
    # )
    return ds, ds_length
