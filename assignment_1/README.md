# Assignment 1 - Extracting linguistic features using spaCy

## Original Assignment Description 

This assignment concerns using ```spaCy``` to extract linguistic information from a corpus of texts.

The corpus is an interesting one: *The Uppsala Student English Corpus (USE)*. All of the data is included in the folder called ```in``` but you can access more documentation via [this link](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457).

For this exercise, you should write some code which does the following:

- Loop over each text file in the folder called ```in```
- Extract the following information:
    - Relative frequency of Nouns, Verbs, Adjective, and Adverbs per 10,000 words
    - Total number of *unique* PER, LOC, ORGS
- For each sub-folder (a1, a2, a3, ...) save a table which shows the following information:

|Filename|RelFreq NOUN|RelFreq VERB|RelFreq ADJ|RelFreq ADV|Unique PER|Unique LOC|Unique ORG|
|---|---|---|---|---|---|---|---|
|file1.txt|---|---|---|---|---|---|---|
|file2.txt|---|---|---|---|---|---|---|
|etc|---|---|---|---|---|---|---|

### Objective

This assignment is designed to test that you can:

1. Work with multiple input data arranged hierarchically in folders;
2. Use ```spaCy``` to extract linguistic information from text data;
3. Save those results in a clear way which can be shared or used for future analysis

### Some notes

- The data is arranged in various subfolders related to their content (see the [README](in/README.md) for more info). You'll need to think a little bit about how to do this. You should be able do it using a combination of things we've already looked at, such as ```os.listdir()```, ```os.path.join()```, and for loops.
- The text files contain some extra information that such as document ID and other metadata that occurs between pointed brackets ```<>```. Make sure to remove these as part of your preprocessing steps!
- There are 14 subfolders (a1, a2, a3, etc), so when completed the folder ```out``` should have 14 CSV files.

### Additional comments

Your code should include functions that you have written wherever possible. Try to break your code down into smaller self-contained parts, rather than having it as one long set of instructions.

For this assignment, you are welcome to submit your code either as a Jupyter Notebook, or as ```.py``` script. If you do not know how to write ```.py``` scripts, don't worry - we're working towards that!

Lastly, you are welcome to edit this README file to contain whatever informatio you like. Remember - documentation is important!

## Assignment 1, Luke Ring

Repository: [https://github.com/zeyus/cds-language-exam/tree/main/assignment_1](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1)

This repository contains a script for the Cultural Data Science: Language Analytics course at Aarhus University. The script recursively extracts linguistic features from text files in an input folder and saves them in CSV files in an output folder.

### Contribution

This assignment was completed by me individually and independently, the code contained in this repository is my own work.

### Setup

Using anaconda:

```bash
conda env create -f environment.yml
conda activate cds-lang-1
```

Using pip:

```bash
pip install -r requirements.txt
```

### Usage

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

#### Example

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
100%|--------------------------------------| 1497/1497 [02:28<00:00, 10.06it/s]
2023-02-25 11:14:10 - INFO - Writing summary tables
```

### Output

The script creates a CSV file for each text subfolder in the input folder. The CSV files are named after the subfolder.


Each CSV file contains the following columns:

| Column | Description |
| :--- | :--- |
| Filename | The name of the text file |
| RelFreq NOUN | The relative frequency of nouns in the text |
| RelFreq VERB | The relative frequency of verbs in the text |
| RelFreq ADJ | The relative frequency of adjectives in the text |
| RelFreq ADV | The relative frequency of adverbs in the text |
| Unique PER | The number of unique named entities of type PERSON |
| Unique LOC | The number of unique named entities of type LOCATION |
| Unique ORG | The number of unique named entities of type ORGANIZATION |


The followg output CSV files are available in the `out` folder:

- [a1.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/a1.csv)
- [a2.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/a2.csv)
- [a3.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/a3.csv)
- [a4.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/a4.csv)
- [a5.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/a5.csv)
- [b1.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/b1.csv)
- [b2.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/b2.csv)
- [b3.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/b3.csv)
- [b4.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/b4.csv)
- [b5.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/b5.csv)
- [b6.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/b6.csv)
- [b7.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/b7.csv)
- [b8.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/b8.csv)
- [c1.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/c1.csv)

#### Example Output

The following is the contents of [b6.csv](https://github.com/zeyus/cds-language-exam/tree/main/assignment_1/out/b6.csv) with the values rounded to 2 decimal places:

| Filename | RelFreq NOUN | RelFreq VERB | RelFreq ADJ | RelFreq ADV | Unique PER | Unique LOC | Unique ORG |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|0107.b6.txt|1724.14|1238.83|855.68|421.46|1|0|1|
|0137.b6.txt|1735.65|1241.66|934.58|534.05|1|0|0|
|0151.b6.txt|1491.23|1353.38|651.63|538.85|3|0|0|
|0157.b6.txt|1215.47|1381.22|718.23|607.73|2|0|0|
|0158.b6.txt|1522.49|1257.21|761.25|657.44|2|0|0|
|0178.b6.txt|1742.34|1140.44|876.45|549.10|2|0|1|
|0185.b6.txt|1609.20|1379.31|675.29|416.67|2|0|0|
|0198.b6.txt|1542.94|1222.71|669.58|465.79|2|0|0|
|0219.b6.txt|1701.53|1311.02|543.93|362.62|2|0|0|
|0223.b6.txt|1731.01|1232.88|660.02|622.67|3|0|0|
|0238.b6.txt|1400.97|1417.07|772.95|402.58|2|0|0|
|0318.b6.txt|1764.71|980.39|813.73|460.78|3|0|0|


