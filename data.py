import csv
import os
import pickle
import sys
import torch
from collections import Counter
from torch.utils.data import TensorDataset

pos_tag_labels = ['X', '.', '\'\'', 'ADD', 'AFX', 'CC', 'CD', 'DT', 'EX', 'FW', 'GW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
                  'MD', 'NFP', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
                  'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
deprel_labels = ['X', 'nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp', 'obl', 'vocative', 'expl', 'dislocated',
                 'advcl', 'advmod', 'discourse', 'aux', 'cop', 'mark', 'nmod', 'appos', 'nummod', 'acl', 'amod', 'det',
                 'clf', 'case', 'conj', 'cc', 'fixed', 'flat', 'compound', 'list', 'parataxis', 'orphan', 'goeswith',
                 'reparandum', 'punct', 'root', 'dep']
native_language_labels = ['X', 'French', 'Dutch', 'Polish', 'Vietnamese', 'Thai', 'Arabic']

def get_pos_labels():
    return pos_tag_labels

def get_deprel_labels():
    return deprel_labels

def get_native_language_labels():
    return native_language_labels


class InputFeatures(object):
    """A single set of features for a single InputExample."""

    def __init__(self, input_ids, input_mask, score, segment_ids=None, pos_tags=None, dep_rels=None, native_language=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.score = score
        self.segment_ids = segment_ids
        self.pos_tags = pos_tags
        self.dep_rels = dep_rels
        self.native_language = native_language

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, id, tokens, score, pos_tags=None, dep_rels=None, native_language=None):
        """Constructs a InputExample.
        Args:
            id: Unique id for the example.
            tokens: string. The tokenized text.
            pos_tags: (Optional) string. The part of speech tags, space separated aligned with tokens
            dep_rels: (Optional) string. The universal dependency relation labels, space separated aligned with tokens
            native_language: (Optional) string. Native language of the speaker
            score: float. Score that was assigned to the example.
        """
        self.id = id
        self.tokens = tokens
        self.pos_tags = pos_tags
        self.dep_rels = dep_rels
        self.native_language = native_language
        self.score = score

class TSVProcessor(object):
    """Processor for TSV data set"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.tsv")), "valid")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        columns = {}
        for (i, line) in enumerate(lines):
            if i == 0:
                columns = {key:header_index for header_index, key in enumerate(line)}
            id = "%s-%s" % (set_type, i)
            tokens = line[columns['text']]
            pos_tags = line[columns['pos']].split(' ') if 'pos' in columns else ['X'] * len(tokens)
            dep_rels = line[columns['deprel']].split(' ') if 'deprel' in columns else ['X'] * len(tokens)
            dep_rels = [dep_rel.split(':')[0] for dep_rel in dep_rels]
            native_language = line[columns['pos']].split(' ') if 'l1' in columns else 'X'
            score = line[columns['score']]
            examples.append(
                InputExample(id=id, tokens=tokens, score=score, pos_tags=pos_tags,
                             dep_rels=dep_rels, native_language=native_language))
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        print(input_file)
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

def convert_examples_to_features(examples, model, max_seq_length, special_tokens, logger,
                                 vocab=None, tokenizer=None):
    special_tokens_count = 2 if model == 'bert' else 0
    pos_tag_label_map = {label: i for i, label in enumerate(pos_tag_labels)}
    dep_rel_label_map = {label: i for i, label in enumerate(deprel_labels)}
    native_language_label_map = {label: i for i, label in enumerate(native_language_labels)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens, pos_tags, dep_rels, native_language = [], [], [], []
        segment_ids = None

        for i, word in enumerate(example.tokens.split(' ')):
            if len(tokens) == max_seq_length - special_tokens_count:
                break
            word_pieces = tokenizer.tokenize(word) if tokenizer else [word]
            tokens.extend(word_pieces)
            # don't predict on special tokens for auxiliary objjectives
            if word in special_tokens:
                pos_tags.append(-1)
                dep_rels.append(-1)
                native_language.append(-1)
            else:
                pos_tags.append(pos_tag_label_map.get(example.pos_tags[i], 0))
                dep_rels.append(dep_rel_label_map.get(example.dep_rels[i], 0))
                native_language.append(native_language_label_map.get(example.native_language, 0))
                if model == 'bert':
                    token_padding = [-1] * (len(word_pieces) - 1)
                    pos_tags.extend(token_padding)
                    dep_rels.extend(token_padding)
                    native_language.extend(token_padding)


        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            pos_tags = pos_tags[:(max_seq_length - special_tokens_count)]
            dep_rels = dep_rels[:(max_seq_length - special_tokens_count)]
            native_language = native_language[:(max_seq_length - special_tokens_count)]


        if model == 'lstm':
            input_ids = [vocab.get(token, 1) for token in tokens]
        else:
            input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
            pos_tags = [-1] + pos_tags + [-1]
            dep_rels = [-1] + dep_rels + [-1]
            native_language = [-1] + native_language + [-1]
            segment_ids = [0] * len(input_ids)


        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        example_padding = max_seq_length - len(input_ids)
        input_ids = input_ids + ([tokenizer.pad_token_id if model == 'bert' else 0] * example_padding)
        input_mask = input_mask + ([0] * example_padding)
        pos_tags = pos_tags + ([-1] * example_padding)
        dep_rels = dep_rels + ([-1] * example_padding)
        native_language = native_language + ([-1] * example_padding)
        if model == 'bert':
            segment_ids = segment_ids + ([0] * example_padding)
            assert len(segment_ids) == max_seq_length


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(pos_tags) == max_seq_length
        assert len(dep_rels) == max_seq_length
        assert len(native_language) == max_seq_length

        score = float(example.score)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("id: %s" % (example.id))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            if model == 'bert':
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("score: %d)" % (score))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              score=score,
                              pos_tags=pos_tags,
                              dep_rels=dep_rels,
                              native_language=native_language
                              ))
    return features

def load_and_cache_vocab(data_dir, logger):
    vocab_file = os.path.join(data_dir, 'cached_vocab.p')
    # Load vocab from cache
    if os.path.exists(vocab_file):
        logger.info("Loading vocab from cached file %s", vocab_file)
        return pickle.load(open(vocab_file, 'rb'))

    # Create tokenizer
    logger.info("Creating vocab from dataset file at %s", data_dir)
    processor = TSVProcessor()
    examples = processor.get_train_examples(data_dir)
    word_counts = Counter([word for example in examples for word in example.tokens.split(' ')])
    sorted_vocab = ['[PAD]', 'UNK'] + sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = {w: k for k, w in enumerate(sorted_vocab)}
    logger.info("Saving vocab of length %d into cached file %s", len(vocab_to_int), vocab_file)
    pickle.dump(vocab_to_int, open(vocab_file, 'wb'))
    return vocab_to_int

def load_and_cache_examples(model, data_dir, max_seq_length, special_tokens, logger, vocab=None, tokenizer=None, evaluate=False, test=False, reload=False):
    assert not (evaluate and test), "Cannot load validation data and test data at the same time."
    processor = TSVProcessor()
    # Load data features from cache or dataset file
    file_type = 'valid' if evaluate else 'train'
    file_type = 'test'  if test else file_type
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}'.format(
        model,
        file_type,
        str(max_seq_length)))
    if not reload and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        if evaluate:
            examples = processor.get_dev_examples(data_dir)
        elif test:
            examples = processor.get_test_examples(data_dir)
        else:
            examples = processor.get_train_examples(data_dir)

        features = convert_examples_to_features(examples, model, max_seq_length, special_tokens, logger, vocab, tokenizer)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_pos_tags = torch.tensor([f.pos_tags for f in features], dtype=torch.long)
    all_dep_rels = torch.tensor([f.dep_rels for f in features], dtype=torch.long)
    all_native_languages = torch.tensor([f.native_language for f in features], dtype=torch.long)
    all_scores = torch.tensor([f.score for f in features], dtype=torch.float)

    if model == 'lstm':
        return TensorDataset(all_input_ids, all_input_mask, all_scores, all_pos_tags, all_dep_rels, all_native_languages)

    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_scores, all_pos_tags, all_dep_rels, all_native_languages)
