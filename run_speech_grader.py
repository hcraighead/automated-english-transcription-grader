import argparse
import logging
import os

import torch
from transformers import BertTokenizer, BertConfig

import data
import train
from models import bert_model, lstm_model

logger = logging.getLogger(__name__)
# The auxiliary objectives used to train the speech grader.
# Note that mlm (masked language modelling) can only be used for the BERT speech grader
# and lm (language modelling) for the LSTM speech grader.
aux_objs = ['pos', 'deprel', 'native_language', 'mlm', 'lm']


def get_auxiliary_objectives(args, vocab_size):
    training_objectives = {'score': (1, args.score_alpha)}
    total_weighting = args.score_alpha
    for obj in aux_objs:
        alpha = getattr(args, '{}_alpha'.format(obj))
        if getattr(args, 'use_{}_objective'.format(obj)):
            if alpha == 0.0:
                logger.info('Must set an alpha value for {} objective'.format(obj))
                return
            if obj == 'mlm' or obj == 'lm':
                num_predictions = vocab_size
            else:
                num_predictions = len(getattr(data, 'get_{}_labels'.format(obj))())
            training_objectives[obj] = (num_predictions, alpha)
            total_weighting += alpha

    if total_weighting != 1.0:
        logger.info('Weighting values for objectives must add up to 1, total was {}'.format(total_weighting))
        return
    return training_objectives


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    ## Other parameters
    parser.add_argument("--model_path", default="bert-base-uncased", type=str,
                        help="Pre-trained BERT model to extend e.g. bert-base-uncased.")
    parser.add_argument("--model", default=None, type=str,
                        help="Type of speech grader model: ['lstm', 'bert']")
    parser.add_argument("--max_score", default=6, type=float,
                         help="Maximum score that an example can be awarded (the default value used is 6, inline with CEFR levels).")
    parser.add_argument('--special_tokens', default=[], type=str, action='append',
                        help='Special tokens to mask when making auxiliary objective predictions. These are also denoted as special tokens for the BERT tokenizer.')

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Train a model.")
    parser.add_argument("--do_test", action='store_true',
                        help="Evaluate the model at --model_dir on the test set.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument('--save_best_on_evaluate', action='store_true',
                        help="Save best model based on evaluation scoring loss.")
    parser.add_argument("--save_all_checkpoints", action='store_true',
                       help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written during training.")
    parser.add_argument("--model_dir", default=None, type=str,
                        help="The directory where the model files are stored (used for testing).")
    parser.add_argument("--model_args_dir", default=None, type=str,
                        help="The directory where the model args are stored (used for testing).")
    parser.add_argument('--overwrite_cache', action='store_true',
                       help="Overwrite the cached training, validation and testing sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--predictions-file', type=str, default=None)

    # Auxiliary objectives
    for obj in aux_objs:
        parser.add_argument('--use_{}_objective'.format(obj), action='store_true',
                            help='Use {} objective during training. --{}_alpha must also be set'.format(obj, obj))
        parser.add_argument('--{}_alpha'.format(obj), type=float, default=0.0,
                            help='Weighting of {} objective in loss score. All alpha values must add to 1'.format(obj))
    parser.add_argument('--score_alpha', type=float, default=1.0,
                        help='Weighting of scoring objective in loss score. All alpha values must add to 1')


    # Parsing args
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.save_best_on_evaluate and not args.evaluate_during_training:
        args.logger.info('Cannot save best model if not evaluating')
        return

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    # Train a model
    if args.do_train:
        # Store training arguments to facilitate reloading a model.
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        args.logger = logger

        if not args.model:
            args.logger.info("--model must be provided for training (['lstm', 'bert'])")
            return
        if args.model == 'lstm':
            vocab = data.load_and_cache_vocab(args.data_dir, logger)
            training_objectives = get_auxiliary_objectives(args, len(vocab))
            grader = lstm_model.SpeechGraderModel(args, vocab, training_objectives).to(args.device)
            train_data = data.load_and_cache_examples(
                args.model, args.data_dir, args.max_seq_length, args.special_tokens,
                logger, vocab=vocab, reload=args.overwrite_cache)
            dev_data = data.load_and_cache_examples(
                args.model, args.data_dir, args.max_seq_length, args.special_tokens,
                logger, vocab=vocab, evaluate=True, reload=args.overwrite_cache)
            trainer = train.Trainer(args, grader, training_objectives)
        elif args.model == 'bert':
            tokenizer = BertTokenizer.from_pretrained(args.model_path, additional_special_tokens=args.special_tokens)
            config = BertConfig.from_pretrained(args.model_path)
            training_objectives = get_auxiliary_objectives(args, tokenizer.vocab_size)
            config.training_objectives = training_objectives
            config.max_score = args.max_score
            grader = bert_model.SpeechGraderModel(config=config).to(args.device)
            train_data = data.load_and_cache_examples(
                args.model, args.data_dir, args.max_seq_length, args.special_tokens,
                logger, tokenizer=tokenizer, reload=args.overwrite_cache)
            dev_data = data.load_and_cache_examples(
                args.model, args.data_dir, args.max_seq_length, args.special_tokens,
                logger, tokenizer=tokenizer, evaluate=True, reload=args.overwrite_cache)
            trainer = train.Trainer(args, grader, training_objectives, bert_tokenizer=tokenizer)
        else:
            args.logger.info("--model must be either 'lstm' or 'bert'")
            return
        trainer.train(train_data, dev_data)

    # Test a model
    if args.do_test:
        # Retrieve training arguments to facilitate reloading the model.
        args.logger = logger
        train_args = torch.load(os.path.join(args.model_args_dir, 'training_args.bin'))
        train_args.predictions_file = args.predictions_file
        train_args.logger = logger

        if train_args.model == 'lstm':
            # use the vocabulary from train time
            vocab = data.load_and_cache_vocab(train_args.data_dir, logger)
            training_objectives = get_auxiliary_objectives(train_args, len(vocab))
            grader = lstm_model.SpeechGraderModel(args, vocab, training_objectives).to(args.device)
            grader.load_state_dict(torch.load(os.path.join(args.model_dir, 'lstm.model')))
            test_data = data.load_and_cache_examples(
                train_args.model, args.data_dir, train_args.max_seq_length, train_args.special_tokens,
                logger, vocab=vocab, test=True, reload=args.overwrite_cache)
            trainer = train.Trainer(train_args, grader, training_objectives)

        else:
            tokenizer = BertTokenizer.from_pretrained(args.model_dir, do_lower_case=True)
            training_objectives = get_auxiliary_objectives(train_args, tokenizer.vocab_size)
            config = BertConfig.from_pretrained(args.model_dir)
            grader = bert_model.SpeechGraderModel.from_pretrained(args.model_dir, config=config).to(args.device)
            test_data = data.load_and_cache_examples(
                train_args.model, args.data_dir, train_args.max_seq_length, train_args.special_tokens,
                logger, tokenizer=tokenizer, test=True, reload=args.overwrite_cache)
            trainer = train.Trainer(train_args, grader, training_objectives, bert_tokenizer=tokenizer)
        trainer.test(test_data)

if __name__ == "__main__":
    main()
