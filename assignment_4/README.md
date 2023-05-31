# Assignment 4 - Using finetuned transformers via HuggingFace

## Original Assignment Description 

In previous assignments, you've done a lot of model training of various kinds of complexity, such as training document classifiers or RNN language models. This assignment is more like Assignment 1, in that it's about *feature extraction*.

For this assignment, you should use ```HuggingFace``` to extract information from the *Fake or Real News* dataset that we've worked with previously.

You should write code and documentation which addresses the following tasks:

- Initalize a ```HuggingFace``` pipeline for emotion classification
- Perform emotion classification for every *headline* in the data
- Assuming the most likely prediction is the correct label, create tables and visualisations which show the following:
  - Distribution of emotions across all of the data
  - Distribution of emotions across *only* the real news
  - Distribution of emotions across *only* the fake news
- Comparing the results, discuss if there are any key differences between the two sets of headlines


### Tips
- I recommend using ```j-hartmann/emotion-english-distilroberta-base``` like we used in class.
- Spend some time thinking about how best to present you results, and how to make your visualisations appealing and readable.
- **MAKE SURE TO UPDATE YOUR README APPROPRIATELY!**

## Assignment 4, Luke Ring

Repository: [https://github.com/zeyus/cds-language-exam/tree/main/assignment_4](https://github.com/zeyus/cds-language-exam/tree/main/assignment_4)

### Contribution

This assignment was completed by me individually and independently, the code contained in this repository is my own work.

### Setup

This assignment uses PyTorch and HuggingFace Transformers. Fine tuning was done using CUDA 11.8 on an NVIDIA GeForce GTX 1070 GPU with 8GB VRAM on a system with 24GB RAM.

#### Prerequisites

- Python 3.11

#### Installation

Clone the repository:
  
```bash
git clone https://github.com/zeyus/cds-language-exam
cd cds-language-exam/assignment_4
```

Install requirements:

```bash
pip install -r requirements.txt
```

### Usage

The script can be run from the command line as follows:

```bash
python3 src/ftt.py
```

It is also possible to specify arguments to the script, which can be seen by running:

```bash
python3 src/ftt.py --help
```

Output:

```text
usage: ftt.py [-h] [--version] [-o OUTPUT_PATH] [-d DATASET_PATH] [-V]

Text classification CLI

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Path to save the output, figures, stats, etc. (default: out)
  -d DATASET_PATH, --dataset-path DATASET_PATH
                        Path to the dataset (default: data)
  -V, --visualize-data  Visualize the dataset (default: False)
```

### Implementation

The headlines from the `fake_or_real_news.csv` file are loaded into a `pandas.DataFrame` object. A `Dataset` is created from using the `Dataset.from_pandas` method. The model used is the recommended `j-hartmann/emotion-english-distilroberta-base` model. The model is then used to predict the emotion of each headline in the dataset using an `inference` function that tokenizes the input and returns a `softmax` of the predictions. The predictions are then saved to a `pandas.DataFrame` object and saved to the `out/news_emotions.csv` file.

After the csv file has been created, the script can be run with the `-V` argument to create visualizations of the emotional distribution of the headlines. The visualizations are saved to the `out` directory.

### Results

![emotional distribution for all news](./out/emotional_distribution_All.png)

![emotional distribution for real news](./out/emotional_distribution_Real.png)

![emotional distribution for fake news](./out/emotional_distribution_Fake.png)

Sample of the predictions (complete predictions can be found in the [news_emotions.csv](./out/news_emotions.csv) file):

| text | emotion | label |
| --- | --- | --- |
|You Can Smell Hillary’s Fear|fear|FAKE|
|Watch The Exact Moment Paul Ryan Committed Political Suicide At A Trump Rally (VIDEO)|sadness|FAKE|
|Kerry to go to Paris in gesture of sympathy|joy|REAL|
|Bernie supporters on Twitter erupt in anger against the DNC: 'We tried to warn you!'|anger|FAKE|
|The Battle of New York: Why This Primary Matters|neutral|REAL|
|Tehran USA|neutral|FAKE|
|Girl Horrified At What She Watches Boyfriend Do After He Left FaceTime On|fear|FAKE|
|‘Britain’s Schindler’ Dies at 106|sadness|REAL|
|Fact check: Trump and Clinton at the 'commander-in-chief' forum|neutral|REAL|
|Iran reportedly makes new push for uranium concessions in nuclear talks|neutral|REAL|


The results look quite good, apart from the third sample "Kerry to go to Paris in gesture of sympathy" which has been clasified as "joy", but in my opinion would be better classified as sadness, although it's easy to imagine that "going to Paris" is usually something associated with holidays and pleasant experiences.

