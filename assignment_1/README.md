# Extract linguistic features from text files

This repository contains a script for the Cultural Data Science: Language Analytics course at Aarhus University. The script recursively extracts linguistic features from text files in an input folder and saves them in CSV files in an output folder.

## Setup

Using anaconda:

```bash
conda env create -f environment.yml
conda activate cds-lang-1
```

Using pip:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/extract_linguistic_info.py -i input_folder -o output_folder <-e encoding> <-m spacy_model>
```

By default, the spacy models `en_core_web_md`and `en_core_web_lg` are included and can be used for the `-m` flag. If you want to use a different model, you need to install it first:

```bash
python -m spacy download <model_name>
```

You can run the script with the `-h` flag to see the available options:

```
usage: extract_linguistic_info.py [-h] [-i I] [-o O] [-e E] [-m M]

Extracts linguistic information from text files

options:
  -h, --help  show this help message and exit
  -i I        The directory containing the text files to be processed
  -o O        The directory to write the output files to
  -e E        The encoding of the input files
  -m M        The spacy model to use
```

### Example

```bash
python src/extract_linguistic_info.py -i in/USEcorpus -o out -e latin-1 -m en_core_web_md
```

This will provide a progress bar and output something like the following to the terminal:

```
2023-02-25 11:11:38 - INFO - Arguments:
2023-02-25 11:11:38 - INFO - Input directory: in\USEcorpus
2023-02-25 11:11:38 - INFO - Output directory: out
2023-02-25 11:11:38 - INFO - Starting linguistic information extraction
2023-02-25 11:11:38 - INFO - Loading spacy model en_core_web_lg
2023-02-25 11:11:38 - INFO - Using GPU
2023-02-25 11:11:41 - INFO - Getting list of files
2023-02-25 11:11:41 - INFO - Found 1497 files
2023-02-25 11:11:41 - INFO - Extracting linguistic information
100%|██████████████████████████████████████| 1497/1497 [02:28<00:00, 10.06it/s]
2023-02-25 11:14:10 - INFO - Writing summary tables
```

## Output

The script creates a CSV file for each text subfolder in the input folder. The CSV files are named after the subfolder.


Each CSV file contains the following columns:

| Column | Description |
| --- | --- |
| Filename | The name of the text file |
| RelFreq NOUN | The relative frequency of nouns in the text |
| RelFreq VERB | The relative frequency of verbs in the text |
| RelFreq ADJ | The relative frequency of adjectives in the text |
| RelFreq ADV | The relative frequency of adverbs in the text |
| Unique PER | The number of unique named entities of type PERSON |
| Unique LOC | The number of unique named entities of type LOCATION |
| Unique ORG | The number of unique named entities of type ORGANIZATION |

