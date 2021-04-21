#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import torch
import torch.nn as nn
from typing import List

from ..extractive_attention_mask import ExtractiveAttentionMask
from ..tokenizers import BartTokenizerWithMapping, PegasusTokenizerWithMapping
from ..base_models import BartForConditionalGenerationEAM, PegasusForConditionalGenerationEAM


class AbstractiveModel(nn.Module):
    datasets_mapping = {
        'bart': {
            'cnn': 'facebook/bart-large-cnn',
            'xsum': 'facebook/bart-large-xsum'
        },
        'pegasus': {
            'cnn': 'google/pegasus-cnn_dailymail',
            'xsum': 'google/pegasus-xsum',
            'gigaword': 'google/pegasus-gigaword'
        }
    }

    setup_mapping = {
        'bart': {
            'base_model': BartForConditionalGenerationEAM,
            'tokenizer': BartTokenizerWithMapping
        },
        'pegasus': {
            'base_model': PegasusForConditionalGenerationEAM,
            'tokenizer': PegasusTokenizerWithMapping
        }
    }

    def __init__(self, base_model_name, dataset):
        super(AbstractiveModel, self).__init__()
        self.base_model_name = base_model_name.lower()
        self.dataset = dataset.lower()

        self.tokenizer = self.setup_mapping[self.base_model_name]['tokenizer']()

        self.base_model_class = self.setup_mapping[self.base_model_name]['base_model']
        self.base_model = self.base_model_class.from_pretrained(
            self.datasets_mapping[self.base_model_name][self.dataset]
        )

        self.extractive_attention_mask = ExtractiveAttentionMask()

    def geneate(self, sentences: List[str], sentence_scores: List[int], **base_model_params):
        tokenized_sequence, mapping = self.tokenizer.tokenize(sentences)
        attention_mask = self.extractive_attention_mask(mapping, sentence_scores)

        summary = self.base_model.generate(input_ids=tokenized_sequence, attention_mask=attention_mask,
                                           decoder_start_token_id=self.tokenizer.bos_token_id, **base_model_params)
        return self.tokenizer.decode(summary)

    def forward(self, batch_sentences: List[List[str]], sentence_scores_batch: List[List[str]], **base_model_params):
        tokenized_batch = []
        mapping_batch = []
        attention_mask_batch = []

        for sentences in batch_sentences:
            tokenized_sequence, mapping = self.tokenizer.tokenize(sentences)
            tokenized_batch += [tokenized_sequence]
            mapping_batch += [mapping]

        for i, sentence_scores in enumerate(sentence_scores_batch):
            mapping = mapping_batch[i]
            attention_mask = self.extractive_attention_mask(mapping, sentence_scores)
            attention_mask_batch += [attention_mask]

        return self.base_model(torch.Tensor(tokenized_batch), torch.Tensor(attention_mask_batch), **base_model_params)
