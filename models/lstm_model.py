import torch
import torch.nn as nn
from transformers.modeling_bert import gelu


class PredictionHead(nn.Module):
    """
    A prediction head for a single objective of the SpeechGraderModel.

    Args:
        hidden_size (int): the size of the hidden layer (i.e. LSTM output size and decoder input size)
        num_labels (int): the number of labels that can be predicted

    Attributes:
        dense (torch.nn.Dense): a dense linear layer
        transform_act_fn: the activation function to apply to the output of the dense layer
        layer_norm (torch.nn.LayerNorm): a normalization layer
        decoder (torch.nn.Linear): a linear layer that makes predictions across the labels
        bias (torch.nn.Parameter): biases per label
    """

    def __init__(self, hidden_size, num_labels):
        super(PredictionHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = gelu
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        self.decoder = nn.Linear(hidden_size, num_labels)
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class SpeechGraderModel(nn.Module):
    """
    Bidirectional LSTM model for automated speech scoring of transcripts.

    Args:
        vocab (): the vocabulary to use
        training_objectives (dict of str: int): mapping of training objectives to the number of labels
        len_embedding (int): the length of the vocabulary embeddings
        num_layers (int): number of layers in the bi-directional LSTM
        hidden_size (int): the size of the hidden layer (i.e. LSTM output size and decoder input size)

    Attributes:
        args (): arguments used to train/evaluate the model
        hidden_size(int): the size of the hidden layer (i.e. LSTM output size and decoder input size)
        encoder (torch.nn.Embedding): the encoder of input vocabulary
        rnn (torch.nn.LSTM): the bidirectional LSTM
        decoder_objectives(list of str): the objectives to be predicted
        {objective}_decoder(PredictionHead): a prediction head for the given objective
        score_scaler(torch.nn.HardTanh): the scoring prediction head for scoring, which scales the score to the accepted
            bounds
    """

    def __init__(self, args, vocab, training_objectives, len_embedding=300, num_layers=3, hidden_size=384):
        super(SpeechGraderModel, self).__init__()
        self.args = args
        self.encoder = nn.Embedding(len(vocab), len_embedding)
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(len_embedding, hidden_size, num_layers=num_layers, bidirectional=True)

        self.decoder_objectives = training_objectives.keys()
        # Creates a prediction head per objective.
        for objective, objective_params in training_objectives.items():
            num_predictions, _ = objective_params
            # Language modelling requires two prediction heads as the next and previous words are predicted using the
            # forward and backward passes of the LSTM respectively.
            if objective == 'lm':
                decoder = PredictionHead(hidden_size, num_predictions)
                setattr(self, objective + '_forward_decoder', decoder)
                decoder = PredictionHead(hidden_size, num_predictions)
                setattr(self, objective + '_backward_decoder', decoder)
            else:
                decoder = PredictionHead(hidden_size * 2, num_predictions)
                setattr(self, objective + '_decoder', decoder)

        # The score scaler is used to force the result of the score prediction head to be within the range of possible
        # scores.
        self.score_scaler = nn.Hardtanh(min_val=0, max_val=args.max_score)

    def forward(self, batch):
        """
        Returns:
            training_objective_predictions (dict of str: [float]): mapping of training objective to the predicted label
        """
        embeddings = self.encoder(batch)
        # The final output of the rnn used by sequence labelling auxiliary objectives.
        rnn_sequence_output, (hidden, cell) = self.rnn(embeddings)
        # The final output of the forward and backward passes are concatenated together and used by the scoring layer.
        # [sent len, batch size, num directions, hid dim]
        a = rnn_sequence_output.view(rnn_sequence_output.size(0), rnn_sequence_output.size(1), 2, self.hidden_size)
        forward = a[-1, :, 0, :].squeeze()
        backward = a[0, :, 1, :].squeeze()
        rnn_output = torch.cat((forward, backward), 1)

        # Predictions per training objective
        training_objective_predictions = {}
        for objective in self.decoder_objectives:
            input = rnn_output if objective is 'score' else rnn_sequence_output
            # Language modelling requires makes both a next and previous word prediction based on the forward and
            # backward passes respectively.
            if objective == 'lm':
                lm_f_decoded = getattr(self, objective + '_forward_decoder')(a[:, :, 0, :].squeeze())
                lm_f_decoded = lm_f_decoded.view(-1, lm_f_decoded.shape[2])
                lm_b_decoded = getattr(self, objective + '_backward_decoder')(a[:, :, 1, :].squeeze())
                lm_b_decoded = lm_b_decoded.view(-1, lm_b_decoded.shape[2])
                decoded_objective = torch.cat((lm_f_decoded, lm_b_decoded), 0)
            else:
                decoded_objective = getattr(self, objective + '_decoder')(input)
                if objective == 'score':
                    decoded_objective = self.score_scaler(decoded_objective)
                decoded_objective = decoded_objective.view(-1, decoded_objective.shape[
                    2]) if objective is not 'score' else decoded_objective.squeeze()
            training_objective_predictions[objective] = decoded_objective

        return training_objective_predictions
