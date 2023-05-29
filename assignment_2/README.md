[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10420172&assignment_repo_type=AssignmentRepo)
# Assignment 2 - Text classification benchmarks

## Setup

Clone the repository and install the requirements:

```bash
git clone https://github.com/AU-CDS/assignment-2---text-classification-zeyus
cd assignment-2---text-classification-zeyus
pip install -r requirements.txt
```

## Running text classification benchmarks

There are two main scripts for running the text classification benchmarks:

- `src/txt-benchmark-lr.py` runs a Linear Regression model on the text classification task
- `src/txt-benchmark-nn.py` runs a Neural Network model on the text classification task

By default, the script uses the paths required for the assignment, but can be customized.

Both scripts support the following arguments:

```
usage:  [-h] [--version] [-f FILE] [-m MODEL_SAVE_PATH] [-r REPORT_PATH] [-v {tfidf,count}]

Text classification CLI

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -f FILE, --file FILE  Path to the CSV file containing the data (default: in\fake_or_real_news.csv)
  -m MODEL_SAVE_PATH, --model-save-path MODEL_SAVE_PATH
                        Path to save the trained model(s) (default: models)
  -r REPORT_PATH, --report-path REPORT_PATH
                        Path to save the classification report(s) (default: out)
  -v {tfidf,count}, --vectorizer {tfidf,count}
                        Vectorizer to use (default: tfidf)
```

The reports contain the following information/columns:

- model: the name of the model
- timestamp: the timestamp of the run
- vectorizer: the name of the vectorizer
- train_accuracy: the accuracy of the model on the training set
- train_precision: the precision of the model on the training set
- train_recall: the recall of the model on the training set
- train_f1: the F1 score of the model on the training set
- test_accuracy: the accuracy of the model on the test set
- test_precision: the precision of the model on the test set
- test_recall: the recall of the model on the test set
- test_f1: the F1 score of the model on the test set
- model_params: the parameters of the model
- vectorizer_params: the parameters of the vectorizer
- train_metrics_report: the classification report of the model on the training set
- test_metrics_report: the classification report of the model on the test set

It's not as pretty as I'd have liked but it can be read into a pandas dataframe for further analysis/summary.

I also was planning to add a script for doing a parameter sweep and add arguments for model and vectorizer parameters, but this is already overengineered (although, it's quite easy to add that in now!)
