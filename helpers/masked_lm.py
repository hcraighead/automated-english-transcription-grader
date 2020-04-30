import torch

def get_special_tokens_mask(tokenizer, labels):
    """ Returns a mask for special tokens that should be ignored for sampling during masked language modelling. """
    return list(map(lambda x: 1 if x in [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id] else 0,
                    labels))

def mask_tokens(tokenizer, inputs, device):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # Sample tokens at 0.15 probability each.
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [get_special_tokens_mask(tokenizer, val) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    # Only compute loss on tokens that are masked out
    labels[~masked_indices] = -1

    # Replace 80% of sampled tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # Replace 10% of sampled tokens with a random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=device)
    inputs[indices_random] = random_words[indices_random]

    # Leave remaining 10% of tokens as is
    return inputs, labels
