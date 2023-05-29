import pandas as pd
import logging
import typing as t
from pathlib import Path
from matplotlib import pyplot as plt
import math


def plot_emotional_distribution(
        data: pd.DataFrame,
        out_path: Path,
        which: t.Optional[str] = None):
    """Plot the emotional distribution of the dataset.

    Args:
        data (pd.DataFrame): the dataset
        out_path (Path): the path to save the figure
        which (str, optional): which news to plot. None plots all news.
    """
    if which is not None:
        data = data[data["label"] == which.upper()]
    else:
        which = "All"
    # color by emotion
    colors = {
        "anger": "red",
        "disgust": "brown",
        "fear": "orange",
        "joy": "green",
        "neutral": "gray",
        "sadness": "blue",
        "surprise": "purple"
    }
    fig, ax = plt.subplots(
        figsize=(10, 5),
        dpi=300
    )
    emotions = data["emotion"].unique()
    emotions.sort()
    emotions = emotions.tolist()
    counts = data["emotion"].value_counts()
    max_count = counts.max()
    total = counts.sum()
    counts = counts[emotions]
    # add bar with counts
    ax.bar(
        emotions,
        counts,
        color=[colors[emotion] for emotion in emotions]
    )

    # add text with counts
    for emotion, count in zip(emotions, counts):
        ax.text(
            emotion,
            count + math.ceil(max_count/100*1.1),
            f"{count/total*100:.2f}% ({count})",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontdict={"size": 8}
        )

    ax.set_title(f"Emotional Distribution for {which} News")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")

    fig.savefig(
        str(out_path / f"emotional_distribution_{which}.png"),
        bbox_inches="tight")


def visualize_data(data: Path, out_path: Path):
    logging.info("Visualizing the dataset")
    logging.info("Plotting emotional distribution for all news")
    plot_emotional_distribution(pd.read_csv(data), out_path, which=None)
    logging.info("Plotting emotional distribution for fake news")
    plot_emotional_distribution(pd.read_csv(data), out_path, which="Fake")
    logging.info("Plotting emotional distribution for real news")
    plot_emotional_distribution(pd.read_csv(data), out_path, which="Real")
