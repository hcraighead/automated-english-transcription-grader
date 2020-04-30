import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
from transformers.modeling_bert import BertPreTrainedModel, BertPredictionHeadTransform

class PredictionHead(nn.Module):
    '''
    A prediction head for a single objective of the SpeechGraderModel.

    Args:
        config (BertConfig): the config for the the pre-trained BERT model
        num_labels (int): the number of labels that can be predicted

    Attributes:
        transform (transformers.modeling_bert.BertPredictionHeadTransform): a dense linear layer with gelu activation
            function
        decoder (torch.nn.Linear): a linear layer that makes predictions across the labels
        bias (torch.nn.Parameter): biases per label
    '''
    def __init__(self, config, num_labels):
        super(PredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, num_labels)
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class SpeechGraderModel(BertPreTrainedModel):
    '''
    BERT Model for automated speech scoring of transcripts.

    Args:
        training_objectives (dict of str: int): mapping of training objectives to the number of predictions required

    Attributes:
        model_dir (string) : directory to store/load this model to/from
        args : arguments from argparse
        bert_tokenizer : the tokenizer to use for the BERT model
        training_objectives_decoders (dict of str: (int, float)): a mapping of training objectives to their training
            parameters (a tuple containing the number of labels (i.e. the decoder output size) and the weight to give to
            this objective in the combined weighted loss function)
    '''
    def __init__(self, args, training_objectives):
        super(SpeechGraderModel, self).__init__(BertConfig.from_pretrained(args.model_dir if args.model_dir else args.model_path))
        self.args = args

        self.bert_config = BertConfig.from_pretrained(args.model_dir if args.model_dir else args.model_path)
        self.bert = BertModel.from_pretrained(args.model_dir if args.model_dir else args.model_path)

        # Creates a prediction head per objective.
        self.decoder_objectives = training_objectives.keys()
        for objective, objective_params in training_objectives.items():
            num_predictions, _ = objective_params
            decoder = PredictionHead(self.bert_config, num_predictions)
            setattr(self, objective + '_decoder', decoder)

        # The score scaler is used to force the result of the score prediction head to be within the range of possible
        # scores.
        self.score_scaler = nn.Hardtanh(min_val=0, max_val=args.max_score)

    def forward(self, batch):
        """
        Returns:
        training_objective_predictions (dict of str: [float]): mapping of training objective to the predicted label
        """
        bert_sequence_output, bert_pooled_output = self.bert(**batch)
        training_objective_predictions = {}

        for objective in self.decoder_objectives:
            input = bert_pooled_output if objective is 'score' else bert_sequence_output
            decoded_objective = getattr(self, objective + '_decoder')(input)
            if objective == 'score':
                decoded_objective = self.score_scaler(decoded_objective)
            training_objective_predictions[objective] = decoded_objective.view(-1, decoded_objective.shape[2]) \
                if objective is not 'score' else decoded_objective.squeeze()

        return training_objective_predictions
