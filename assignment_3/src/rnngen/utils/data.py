from pathlib import Path
import pickle
import tensorflow as tf
import pandas as pd
import logging
import re


def text_encoder(
        ds: tf.data.Dataset,
        max_tokens: int = 5000,
        max_length: int = 100,
        **kwargs) -> tf.keras.layers.experimental.preprocessing.TextVectorization:
    """Create a text encoder.

    Args:
        ds (tf.data.Dataset): The text to encode.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 5000.
        max_length (int, optional): The maximum length of a sequence. Defaults to 100.

    Returns:
        tf.keras.layers.experimental.preprocessing.TextVectorization: The text encoder
    """
    # def preprocess_text(text):
    #     # Remove [ARTICLE], [COMMENT], and [REPLY] tags
    #     text = tf.strings.regex_replace(text, r"\[ARTICLE\]|\[COMMENT\]|\[REPLY\]", "")
    #     # Apply any other preprocessing steps you want, like removing punctuation or lowercasing
    #     # ...
    #     return text
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=max_tokens,
        standardize=None,  # We do our own standardization
        split="whitespace",
        ngrams=None,
        pad_to_max_tokens=True,
        output_mode="int",
        output_sequence_length=max_length,
        **kwargs
    )
    encoder.adapt(ds)
    return encoder


def save_text_vectorizer(path: Path, vectorizer: tf.keras.layers.experimental.preprocessing.TextVectorization):
    """Save the text vectorizer to disk.
    Args:
        path (Path): The path to save the text vectorizer to
        vectorizer (text_encoder.TextEncoder): The text vectorizer to save
    """
    path.mkdir(parents=True, exist_ok=True)
    pickle.dump({
        "config": vectorizer.get_config(),
        "weights": vectorizer.get_weights()
    }, open(path / "text_vectorizer.pkl", "wb"))


def load_text_vectorizer(path: Path) -> tf.keras.layers.experimental.preprocessing.TextVectorization:
    """Load the text vectorizer from disk.
    Args:
        path (Path): The path to load the text vectorizer from
    Returns:
        text_encoder.TextEncoder: The loaded text vectorizer

    Raises:
        FileNotFoundError: If the text vectorizer does not exist at the given path
    """
    if not (path / "text_vectorizer.pkl").exists():
        logging.error("The text vectorizer does not exist at %s", path / "text_vectorizer.pkl")
        raise FileNotFoundError("The text vectorizer does not exist at %s" % (path / "text_vectorizer.pkl"))
    data = pickle.load(open(path / "text_vectorizer.pkl", "rb"))
    if "batch_input_shape" in data["config"]:
        if data["config"]["batch_input_shape"] is None:
            del data["config"]["batch_input_shape"]
    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization.from_config(data["config"])
    vectorizer.set_weights(data["weights"])
    return vectorizer


def strip_standardize_tensor(text: tf.Tensor) -> tf.Tensor:
    """Strip and standardize text.
    Args:
        text (tf.Tensor): The text to strip and standardize
    Returns:
        tf.Tensor: The stripped and standardized text
    """
    # Replace line breaks with spaces
    text = tf.strings.regex_replace(text, r"[\n\r]+", " ")
    # Lowercase text
    text = tf.strings.lower(text)
    # Remove non-alphanumeric characters and punctuation we don't want
    text = tf.strings.regex_replace(text, r"[^a-zA-Z0-9\s,.\-!?]", "")
    # add spaces around punctuation
    text = tf.strings.regex_replace(text, r"([.,!?\-])", r" \1 ")
    # Remove multiple spaces
    text = tf.strings.regex_replace(text, r"\s{2,}", " ")
    text = tf.strings.strip(text)
    return text


def strip_standardize_series(text: pd.Series) -> pd.Series:
    """Strip and standardize text.
    Args:
        text (pd.Series): The text to strip and standardize
    Returns:
        pd.Series: The stripped and standardized text
    """
    # if the contents equals "Unknown", return an empty string
    text = text.str.replace("Unknown", "")
    # Replace line breaks with spaces
    text = text.str.replace(r"[\n\r]+", " ", regex=True)
    # Lowercase text
    text = text.str.lower()
    # Replace <br>, <br />, <br/> with spaces
    text = text.str.replace(r"<br\s*/?>", " ", regex=True)
    # Remove non-alphanumeric characters and punctuation we don't want
    text = text.str.replace(r"[^a-zA-Z0-9\s,.\-!?]", "", regex=True)
    # add spaces around punctuation
    text = text.str.replace(r"([\-.,!?])", r" \1 ", regex=True)
    # Remove multiple spaces
    text = text.str.replace(r"\s{2,}", " ", regex=True)
    text = text.str.strip()
    return text


def strip_standardize_py(text: str) -> str:
    """Strip and standardize text.
    Args:
        text (str): The text to strip and standardize
    Returns:
        str: The stripped and standardized text
    """
    # if the contents equals "Unknown", return an empty string
    text = text.replace("Unknown", "")
    # Replace line breaks with spaces
    text = re.sub(r"[\n\r]+", " ", text)
    # Lowercase text
    text = text.lower()
    # Replace <br>, <br />, <br/> with spaces
    text = re.sub(r"<br\s*/?>", " ", text)
    # Remove non-alphanumeric characters and punctuation we don't want
    text = re.sub(r"[^a-zA-Z0-9\s,.\-!?]", "", text)
    # add spaces around punctuation
    text = re.sub(r"([\-.,!?])", r" \1 ", text)
    # Remove multiple spaces
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    return text


def load_nyt_data(
        path: Path,
        batch_size: int = 256,
        validation_frac: float = 0.1,
        max_tokens: int = 5000,
        max_length: int = 1000) -> tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        tf.keras.layers.experimental.preprocessing.TextVectorization]:
    """Load the New York Times comments dataset.
    Args:
        path (Path): The path to the dataset
        batch_size (int, optional): The batch size. Defaults to 256.
        validation_pct (float, optional): The percentage of the dataset to use for validation. Defaults to 0.2.
    Returns:
        tuple[
            tf.data.Dataset: The training dataset
            tf.data.Dataset: The validation dataset
            tf.keras.layers.experimental.preprocessing.TextVectorization: The text encoder
        ]
    """

    ITEM_TOKEN = "<ITEM>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"

    def cap_at_max_length(text: str, max_len=max_length):
        if type(text) != str:
            return str
        text = text.split(" ")
        if len(text) > max_len:
            text = text[:max_len]
        return " ".join(text)

    def configure_for_performance(
            ds: tf.data.Dataset,
            shuffle: bool = False) -> tf.data.Dataset:
        if shuffle:
            logging.info("Shuffling dataset")
            ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        ds = ds.cache()
        logging.info("Batching dataset")
        ds = ds.batch(
            batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds
    
    # def load_data_with_element_spec(path: Path) -> tf.data.Dataset:
    #     tf.io.gfile()

    logging.info("Loading the New York Times comments dataset")
    logging.info(f"Dataset path: {path}")
    logging.info("Trying to load a cached dataset from disk...")

    if (path / "train_ds").exists() and (path / "val_ds").exists() and (path / "encoder" / "text_vectorizer.pkl").exists():
        logging.info("Cached dataset found, loading from disk...")

        # # load spec
        # with open(path / "train_ds_spec.pkl", "rb") as f:
        #     train_spec = pickle.load(f)
        # logging.info("Train data spec: %s", train_spec)
        # train_ds = tf.data.Dataset.load(str(path / "train_ds"), element_spec=train_spec)

        # with open(path / "val_ds_spec.pkl", "rb") as f:
        #     val_spec = pickle.load(f)
        # logging.info("Validation data spec: %s", val_spec)
        # val_ds = tf.data.Dataset.load(str(path / "val_ds"), element_spec=val_spec)
        train_ds = tf.data.Dataset.load(str(path / "train_ds"))
        val_ds = tf.data.Dataset.load(str(path / "val_ds"))
        text_vectorizer = load_text_vectorizer(path / "encoder")
        if text_vectorizer._output_sequence_length != max_length:
            logging.warning(f"Overriding encoder output sequence length from {text_vectorizer._output_sequence_length} to {max_length}")
            text_vectorizer._output_sequence_length = max_length
            vocab = text_vectorizer.get_vocabulary()
            if len(vocab) > max_tokens:
                logging.warning(f"Overriding encoder vocabulary size from {len(vocab)} to {max_tokens}")
                vocab = vocab[:max_tokens]
                text_vectorizer.set_vocabulary(vocab)
        logging.info("Loaded cached dataset from disk, configuring for performance...")
        train_ds = configure_for_performance(train_ds, shuffle=True)
        val_ds = configure_for_performance(val_ds, shuffle=False)
        return train_ds, val_ds, text_vectorizer
    logging.info("No cached dataset found, loading from raw data")

    article_prefix = "Articles"  # contains title, tags, metadata
    # abstract,articleID,articleWordCount,byline,documentType,headline,keywords,multimedia,newDesk,printPage,pubDate,sectionName,snippet,source,typeOfMaterial,webURL
    article_relevant_columns = [
        "articleID",  # for linking comments to articles
        "keywords",
        "headline",
        "snippet"]

    comments_prefix = "Comments"  # contains comments
    # approveDate,commentBody,commentID,commentSequence,commentTitle,commentType,createDate,depth,editorsSelection,parentID,parentUserDisplayName,permID,picURL,recommendations,recommendedFlag,replyCount,reportAbuseFlag,sharing,status,timespeople,trusted,updateDate,userDisplayName,userID,userLocation,userTitle,userURL,inReplyTo,articleID,sectionName,newDesk,articleWordCount,printPage,typeOfMaterial
    comments_relevant_columns = [
        "commentBody",
        "commentID",
        "parentID",
        "articleID",
        "depth"
    ]

    logging.info("Loading articles")
    # Load the articles {article_prefix}*.csv
    articles = pd.concat(
        pd.read_csv(f, usecols=article_relevant_columns)
        for f in path.glob(f"{article_prefix}*.csv")
    )
    logging.info(f"Loaded {len(articles)} articles")

    logging.info("Loading comments")
    # Load the comments {comments_prefix}*.csv
    comments = pd.concat(
        pd.read_csv(f, usecols=comments_relevant_columns)
        for f in path.glob(f"{comments_prefix}*.csv")
    )
    logging.info(f"Loaded {len(comments)} comments")

    # preprocess the text
    articles["headline"] = strip_standardize_series(articles["headline"])
    articles["snippet"] = strip_standardize_series(articles["snippet"])

    # keywords are like: "['Television', 'The Good Fight (TV Program)']"
    articles["keywords"] = strip_standardize_series(articles["keywords"])
    comments["commentBody"] = strip_standardize_series(comments["commentBody"])

    # log some examples
    logging.info("Article examples:")
    for i in range(5):
        logging.info(f"Article headline {i}: {articles.iloc[i]['headline']}")
        logging.info(f"Article snippet {i}: {articles.iloc[i]['snippet']}")
    logging.info("Comment examples:")
    for i in range(5):
        logging.info(f"Comment {i}: {comments.iloc[i]['commentBody']}")
    # Join the comments to the articles
    comments = comments.join(
        articles.set_index("articleID"),
        on="articleID",
        how="inner",
        rsuffix="_article"
    )

    del articles
    # Drop the articleID column
    comments.drop(columns=["articleID"], inplace=True)

    # cut commentBody to at most max_length - 2 (for start/end tokens)
    comments["commentBody"] = comments["commentBody"].apply(cap_at_max_length)

    # Turn into a row per comment thread (article + top-level comment, reply to that comment)
    # parentId = 0.0 for top-level comments and depth = 1
    # commentID is <some_number>.x
    # parentId = commentId for replies and depth = 2 (I can't see any depth > 2)

    # Get the top-level comments
    top_level_comments = comments[comments["depth"] == 1]
    # Get the replies
    replies = comments[comments["depth"] == 2]
    logging.info(f"Found {len(top_level_comments)} top-level comments and {len(replies)} replies")
    # Join the replies to the top-level comments, but we don't want to lose comments that don't have replies
    comments = top_level_comments.join(
        replies.set_index("parentID"),
        on="commentID",
        how="left",
        rsuffix="_reply"
    )
    del top_level_comments
    del replies

    # Drop the parentID column
    comments.drop(columns=["parentID"], inplace=True)

    # Now we want a single column for the text like:
    # '[TITLE] commentBody [KEYWORDS] keywords [COMMENT] commentBody [REPLY] commentBody_reply (if it exists)'
    # We'll use the headline and keywords from the article, and the comment and reply

    # Add the [TITLE] and [KEYWORDS] tags
    # comments["headline"] = (
    #     "[TITLE] " + comments["headline"] + " [KEYWORDS] " + comments["keywords"]
    # )
    # Drop the keywords column
    # comments.drop(columns=["keywords"], inplace=True)


    comments_toplevel = comments[comments["depth"] == 1]
    comments_reply = comments[comments["depth"] == 2]
    del comments
    comments_toplevel["text"] = (
        f"{ITEM_TOKEN} " + comments_toplevel["headline"] + comments_toplevel["snippet"] + comments_toplevel["keywords"])
    # comments_toplevel["text_no_tokens"] = comments_toplevel["headline"] + " " + comments_toplevel["keywords"]
    # ensure text is not longer than max_length
    comments_toplevel["text"] = comments_toplevel["text"].apply(cap_at_max_length, args=(max_length - 2,))
    comments_toplevel["label"] = (
        f"{START_TOKEN} " + comments_toplevel["commentBody"] + f" {END_TOKEN}"
    )
    # comments_toplevel["label_no_tokens"] = comments_toplevel["commentBody"]
    comments_reply["text"] = (
        f"{ITEM_TOKEN} " + comments_reply["commentBody"]
    )
    # ensure text is not longer than max_length
    comments_reply["text"] = comments_reply["text"].apply(cap_at_max_length, args=(max_length - 2,))
    # comments_reply["text_no_tokens"] = comments_reply["commentBody"]
    comments_reply["label"] = (
        f"{START_TOKEN} " + comments_reply["commentBody_reply"] + f" {END_TOKEN}"
    )
    # comments_reply["label_no_tokens"] = comments_reply["commentBody_reply"]

    # comments["text"] = (
    #     "[ARTICLE] " + comments["headline"] + comments["snippet"] + comments["keywords"] + " [COMMENT] " + comments["commentBody"]
    # )
    # comments[""]

    comments = pd.concat([comments_toplevel, comments_reply])
    comments.drop(columns=["headline", "snippet", "keywords", "commentBody", "commentBody_reply"], inplace=True)
    del comments_toplevel
    del comments_reply
    logging.info(f"Loaded {len(comments)} comments")
    # preshusffle the dataset
    logging.info("Pre-shuffling the entire dataset")
    comments = comments.sample(frac=1)

    # Split into train and validation based on validation_frac
    train_ds = comments[:int(len(comments) * (1 - validation_frac))]
    val_ds = comments[int(len(comments) * (1 - validation_frac)):]
    del comments

    # now create the datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_ds["text"], train_ds["label"]))
    val_ds = tf.data.Dataset.from_tensor_slices((val_ds["text"], val_ds["label"]))
    logging.info("Saving datasets")
    train_ds.save(str(path / "train_ds"))
    logging.info("Saved train_ds")

    train_spec = tf.data.DatasetSpec.from_value(train_ds)
    logging.info("Train data spec: %s", train_spec)
    # save spec
    with open(path / "train_ds_spec.pkl", "wb") as f:
        pickle.dump(train_spec, f)

    val_ds.save(str(path / "val_ds"))
    logging.info("Saved val_ds")

    val_spec = tf.data.DatasetSpec.from_value(val_ds)
    logging.info("Val data spec: %s", val_spec)
    # save spec
    with open(path / "val_ds_spec.pkl", "wb") as f:
        pickle.dump(val_spec, f)

    # log some examples from train_ds
    logging.info("Train_ds examples:")
    for i, (text, label) in enumerate(train_ds.take(5)):
        logging.info(f"Example {i}: {text.numpy()}")
        logging.info(f"Label {i}: {label.numpy()}")
    # log some examples from val_ds
    logging.info("Val_ds examples:")
    for i, (text, label) in enumerate(val_ds.take(5)):
        logging.info(f"Example {i}: {text.numpy()}")
        logging.info(f"Label {i}: {label.numpy()}")
    logging.info("Preparing encoder")
    encoder = text_encoder(train_ds, max_tokens=max_tokens, max_length=max_length)
    logging.info(f"Loaded encoder with {encoder.vocabulary_size()} tokens, saving encoder...")
    # save / cache the encoder
    save_text_vectorizer((path / "encoder"), encoder)

    train_ds = configure_for_performance(train_ds, shuffle=True)
    val_ds = configure_for_performance(val_ds, shuffle=False)

    return train_ds, val_ds, encoder
