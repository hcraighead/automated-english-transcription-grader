# Investigating the effect of auxiliary objectives for the automated grading of learner English speech transcriptions (ACL 2020)


This repository includes code to replicate experiments in the [Investigating the effect of auxiliary objectives for the automated grading of learner English speech transcriptions (ACL2020)](https://www.aclweb.org/anthology/2020.acl-main.206.pdf) paper. It provides the LSTM and BERT speech grader models with sequence labelling auxiliary objective prediction heads, code for training and evaluating the models and code for data preparation from TSV files. Please cite our paper when using our code.

## Installation
This project was developed using Python 3.6.9, [PyTorch](https://pytorch.org/) and [Transformers](https://github.com/huggingface/transformers). To install the project, use the following commands:
```
$ git clone https://github.com/hcraighead/automated-english-transcription-grader.git
$ cd automated-english-transcription-grader
$ pip install -r requirements.txt
```

## Data preparation
The experiments in our work were performed on responses collected to Cambridge Assessment's [BULATs examination](https://www.cambridgeenglish.org/exams-and-tests/bulats), which is not publicly available. However, you can provide any TSV file (containing a header) of transcriptions containing the following columns:
- text (required): the transcription of the speech (spaces are assumed to signify tokenization)
- score (required): the numerical score assigned to the speech (by default, a scale between 0 - 6 is used to match CEFR proficiency levels)
- pos (optional): Penn Treebank part of speech tags. These should be space-separated and aligned with a token in text (i.e. there should be an identical number of tokens and POS tags)
- deprel (optional): Universal Dependency relation to head/parent token. These should be space-separated and aligned with a token in text (i.e. there should be an identical number of tokens and Universal Dependency relation labels)
- l1 (optional): native language/L1 of the speaker. Our experiments included L1 speakers of Arabic, Dutch, French, Polish, Thai and Vietnamese.

Your TSV file can include other columns, which will be ignored. Your data directory should contain TSVs named `train.tsv`, `valid.tsv` and `test.tsv`.

## Train a model
To train a model, you must provide a `--model` (lstm or bert), `--data_dir` (a data directory containing your TSVs named train.tsv, valid.tsv and test.tsv) and `--output_dir` (where your model should be stored). If you are training a BERT model, you may additionally provide a `--model_path` to indicate the BERT model checkpoint to initialise from (this is set to `bert-base-uncased` by default).

Example of training an LSTM model:
```
python3 run_speech_grader.py \
    --do_train \
    --model lstm \
    --output_dir lstm-model/ 
    --data_dir data/ 
```

Example of training a BERT model:
```
python3 run_speech_grader.py \ 
    --do_train \
    --model bert \
    --output_dir bert-model/ \
    --data_dir data/ \
    --model_path bert-base-uncased
```
To log performance on a validation set, use the `--evaluate_during_training` option.

To add special tokens use the `--special_tokens` option. In our experiments, we used this to treat hesitation tokens in the BULATs transcriptions as special tokens, meaning that they are not be split by the BERT tokenzier and will not contribute to loss functions for the sequence tagging auxiliary objectives.

To customise to your particular grading scale (which this work assumes begins at 0), provide your maximum numerical grade using `--max_score`.

Example of customised grader options:
```
python3 run_speech_grader.py \ 
    --do_train \
    --model bert \
    --output_dir bert-model/ \
    --data_dir data/ \
    --model_path bert-base-uncased \
    --evalute_during_training \
    --max_score 100 \
    --special_tokens "%hesitation%"
```

### Training with auxiliary objectives
To train the graders with auxiliary objectives, the auxiliary objective flag(s) needs to be provided: `--use_{auxiliary objective}_obj`. Each objective, including scoring, must have a provided alpha value to indicate how much weight should be given to it in the combined loss function. These alpha values must combine to equal 1. Below is an example of training a BERT model using all four possible auxiliary objectives:
```
python3 run_speech_grader.py \ 
    --do_train \
    --model bert \
    --output_dir bert-model/ \
    --data_dir data/ \
    --model_path bert-base-uncased \
    --use_mlm_obj \
    --use_pos_obj \
    --use_deprel_obj \
    --use_native_language_obj \
    --mlm_alpha 0.1 \
    --pos_alpha 0.1 \
    --deprel_alpha 0.1 \
    --native_language_alpha 0.1 \
    --score_alpha 0.6
```

Note that the BERT speech grader uses --use_mlm_obj for masked language modeling and the LSTM speech grader uses --use_lm_obj for language modeling.

### Saving models
Training arguments are saved in the `--output_dir` provided during training. A model will always be saved at the end of the training process in a folder within `output_dir` named `final`. If you are using `--evaluate_during_training`, you can also use `--save_best_on_evaluate` to store the model that achieves that best result on the validation set or `--save_all_checkpoints` to save a checkpoint at every evaluation.

## Test a model
To test a model, you must provide`--model_args_dir` (directory to where the training arguments for your model are stored, which will be `--output_dir`),`--model_dir` (directory where the model is stored, which will be a `final`, `best` or `checkpoint` directory within `--output_dir`) and `--data_dir` (directory to TSV data to test the model on). An example of how to test a model is shown below:
```
python3 run_speech_grader.py \
    --do_test \
    --model_args_dir lstm-model/ \
    --model_dir lst-model/best/ \
    --data_dir data/
```

This will log various statistics about the performance of the model on the test set. Additionally, `--prediction-file` can be used to provide a directory to store the predictions of the grader.

## Citation
```
@inproceedings{craighead-etal-2020-investigating,
    title = "Investigating the effect of auxiliary objectives for the automated grading of learner {E}nglish speech transcriptions",
    author = "Craighead, Hannah  and
      Caines, Andrew  and
      Buttery, Paula  and
      Yannakoudakis, Helen",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.206",
    doi = "10.18653/v1/2020.acl-main.206",
    pages = "2258--2269",
}
```
