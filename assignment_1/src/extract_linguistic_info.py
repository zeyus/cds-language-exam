"""extract_linguistic_info.py

Assignment 1: Cultural Data Science: Language Analytics
Luke Ring

Assignment:
Loop over each text file in the folder called "in"
Extract the following information:
Relative frequency of Nouns, Verbs, Adjective, and Adverbs per 10,000 words
Total number of unique PER, LOC, ORGS

For each sub-folder (a1, a2, a3, ...)
save a table which shows the following information:

Filename
RelFreq NOUN
RelFreq VERB
RelFreq ADJ
RelFreq ADV
Unique PER
Unique LOC
Unique ORG

"""


import os
import re
import pathlib
import typing as t
import argparse
import logging
import spacy
import tqdm


def remove_xml_tags(text: str) -> str:
    """Removes XML tags from the text."""

    # the tags we want to match do not have spaces in them
    text = re.sub(r"<[^\s>]*>", "", text)

    return text


def get_token_counts(doc) -> t.Dict[str, float]:
    """Returns the relative frequency of each part of speech."""
    # first get the frequency of each part of speech and total number of words
    total_words = len(doc)
    freq = {
        "NOUN": 0,
        "VERB": 0,
        "ADJ": 0,
        "ADV": 0,
    }

    ents = {
        "PERSON": [],
        "LOC": [],
        "ORG": []
    }

    # get the parts of text
    for token in doc:
        if token.pos_ in freq:
            freq[token.pos_] += 1

    # get the entities
    for ent in doc.ents:
        if ent.label_ in ents:
            ents[ent.label_].append(ent.text)

    # now calculate the relative frequency
    # and unique entities
    rel_freq = {
        "NOUN": freq["NOUN"] / total_words * 10000,
        "VERB": freq["VERB"] / total_words * 10000,
        "ADJ": freq["ADJ"] / total_words * 10000,
        "ADV": freq["ADV"] / total_words * 10000,
        "PER": len(set(ents["PERSON"])),
        "LOC": len(set(ents["LOC"])),
        "ORG": len(set(ents["ORG"]))
    }

    return rel_freq


def main(
        in_dir: pathlib.Path,
        out_dir: pathlib.Path,
        encoding: str,
        spacy_model: str) -> None:
    """Main function for extracting linguistic information from text files."""

    logging.info("Starting linguistic information extraction")

    # first make sure in and out directories exist
    if not os.path.exists(in_dir) or not os.path.isdir(in_dir):
        raise ValueError("The input directory does not exist")
    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        raise ValueError("The output directory does not exist")

    # Load the spacy model
    logging.info(f"Loading spacy model {spacy_model}")

    # Check if GPU is available
    using_gpu: bool = spacy.prefer_gpu()  # type: ignore
    if using_gpu:
        logging.info("Using GPU")
    else:
        logging.info("Using CPU")
    nlp = spacy.load(spacy_model)

    # Get all files in the input directory and subdirectories
    logging.info("Getting list of files")
    files = []
    for root, _, file_names in os.walk(in_dir):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))

    logging.info(f"Found {len(files)} files")

    # Loop over each file, extract info and write summary table for each folder
    # The summary table will be saved with the name of the folder and will
    # contain a row for each file in the folder
    logging.info("Extracting linguistic information")
    rel_freqs = {}
    # iterate with progress bar
    for file in tqdm.tqdm(files):
        # get the folder name
        folder = os.path.basename(os.path.dirname(file))

        # add the folder to the dictionary if it doesn't exist
        if folder not in rel_freqs:
            rel_freqs[folder] = {}

        # get the file name
        file_name = os.path.basename(file)

        # open the file and extract the linguistic information
        with open(file, "r", encoding=encoding) as f:
            text = f.read()
            # remove XML metadata
            text = remove_xml_tags(text)
            doc = nlp(text)
            rel_freqs[folder][file_name] = get_token_counts(doc)

    # Now write the summary tables to the output directory
    logging.info("Writing summary tables")
    for folder, file_rel_freqs in rel_freqs.items():
        # create the output file name
        out_file = os.path.join(out_dir, folder + ".csv")

        # write the summary table to the output file
        with open(out_file, "w") as f:
            # write the header
            f.write("Filename,RelFreq NOUN,RelFreq VERB,RelFreq ADJ,RelFreq ADV,Unique PER,Unique LOC,Unique ORG\n")  # noqa: E501

            # write the data
            for file_name, rel_freq in file_rel_freqs.items():
                f.write(f"{file_name},{rel_freq['NOUN']},{rel_freq['VERB']},{rel_freq['ADJ']},{rel_freq['ADV']},{rel_freq['PER']},{rel_freq['LOC']},{rel_freq['ORG']}\n")  # noqa: E501


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(
        description="Extracts linguistic information from text files")
    parser.add_argument(
        "-i",
        type=pathlib.Path,
        default="in/USEcorpus",
        help="The directory containing the text files to be processed")
    parser.add_argument(
        "-o",
        type=pathlib.Path,
        default="out",
        help="The directory to write the output files to")
    parser.add_argument(
        "-e",
        type=str,
        default="latin-1",
        help="The encoding of the input files")
    parser.add_argument(
        "-m",
        type=str,
        default="en_core_web_lg",
        help="The spacy model to use"
    )
    args = parser.parse_args()

    # set up logging to terminal
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # print the arguments
    logging.info("Arguments:")
    logging.info(f"Input directory: {args.i}")
    logging.info(f"Output directory: {args.o}")

    # run the main function
    main(args.i, args.o, args.e, args.m)
