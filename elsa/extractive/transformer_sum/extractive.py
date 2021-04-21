#
# Source: https://github.com/HHousen/TransformerSum
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import sys
import logging

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

from argparse import Namespace
import pytorch_lightning as pl
import torch
from .pooling import Pooling
from .data import SentencesProcessor
from .classifier import (
    LinearClassifier,
    SimpleLinearClassifier,
    TransformerEncoderClassifier,
)

logger = logging.getLogger(__name__)


class ExtractiveSummarizer(pl.LightningModule):
    def __init__(self, hparams, embedding_model_config=None, classifier_obj=None):
        super(ExtractiveSummarizer, self).__init__()

        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)

        hparams.gradient_checkpointing = getattr(
            hparams, "gradient_checkpointing", False
        )
        hparams.tokenizer_no_use_fast = getattr(hparams, "tokenizer_no_use_fast", False)

        self.hparams = hparams
        self.forward_modify_inputs_callback = None

        if not embedding_model_config:
            embedding_model_config = AutoConfig.from_pretrained(
                hparams.model_name_or_path,
                gradient_checkpointing=hparams.gradient_checkpointing,
            )

        self.word_embedding_model = AutoModel.from_config(embedding_model_config)

        if (
            "roberta" in hparams.model_name_or_path
            or "distil" in hparams.model_name_or_path
        ) and not hparams.no_use_token_type_ids:
            logger.warning(
                (
                    "You are using a %s model but did not set "
                    + "--no_use_token_type_ids. This model does not support `token_type_ids` so "
                    + "this option has been automatically enabled.",
                    hparams.model_type,
                )
            )
            self.hparams.no_use_token_type_ids = True

        if hparams.pooling_mode == "sent_rep_tokens":
            self.pooling_model = Pooling(
                sent_rep_tokens=True, mean_tokens=False, max_tokens=False
            )
        elif hparams.pooling_mode == "max_tokens":
            self.pooling_model = Pooling(
                sent_rep_tokens=False, mean_tokens=False, max_tokens=True
            )
        else:
            self.pooling_model = Pooling(
                sent_rep_tokens=False, mean_tokens=True, max_tokens=False
            )

        if classifier_obj:
            self.encoder = classifier_obj

        else:
            classifier_exists = getattr(hparams, "classifier", False)
            if (not classifier_exists) or (hparams.classifier == "linear"):
                self.encoder = LinearClassifier(
                    self.word_embedding_model.config.hidden_size,
                    dropout=hparams.classifier_dropout,
                )
            elif hparams.classifier == "simple_linear":
                self.encoder = SimpleLinearClassifier(
                    self.word_embedding_model.config.hidden_size
                )
            elif hparams.classifier == "transformer":
                self.encoder = TransformerEncoderClassifier(
                    self.word_embedding_model.config.hidden_size,
                    dropout=hparams.classifier_dropout,
                    num_layers=hparams.classifier_transformer_num_layers,
                )
            elif hparams.classifier == "transformer_linear":
                linear = LinearClassifier(
                    self.word_embedding_model.config.hidden_size,
                    dropout=hparams.classifier_dropout,
                )
                self.encoder = TransformerEncoderClassifier(
                    self.word_embedding_model.config.hidden_size,
                    dropout=hparams.classifier_dropout,
                    num_layers=hparams.classifier_transformer_num_layers,
                    custom_reduction=linear,
                )
            else:
                logger.error(
                    "%s is not a valid value for `--classifier`. Exiting...",
                    hparams.classifier,
                )
                sys.exit(1)

        self.hparams.no_test_block_trigrams = getattr(
            hparams, "no_test_block_trigrams", False
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams.tokenizer_name
            if hparams.tokenizer_name
            else hparams.model_name_or_path,
            use_fast=(not self.hparams.tokenizer_no_use_fast),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        sent_rep_mask=None,
        token_type_ids=None,
        sent_rep_token_ids=None,
        sent_lengths=None,
        sent_lengths_mask=None,
        **kwargs,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if not self.hparams.no_use_token_type_ids:
            inputs["token_type_ids"] = token_type_ids

        if self.forward_modify_inputs_callback:
            inputs = self.forward_modify_inputs_callback(inputs)  # skipcq: PYL-E1102

        outputs = self.word_embedding_model(**inputs, **kwargs)
        word_vectors = outputs[0]

        sents_vec, mask = self.pooling_model(
            word_vectors=word_vectors,
            sent_rep_token_ids=sent_rep_token_ids,
            sent_rep_mask=sent_rep_mask,
            sent_lengths=sent_lengths,
            sent_lengths_mask=sent_lengths_mask,
        )

        sent_scores = self.encoder(sents_vec, mask)
        return sent_scores, mask

    def predict(self, src_txt):

        input_ids = SentencesProcessor.get_input_ids(
            self.tokenizer,
            src_txt,
            sep_token=self.tokenizer.sep_token,
            cls_token=self.tokenizer.cls_token,
            bert_compatible_cls=True,
        )

        input_ids = torch.tensor(input_ids)
        attention_mask = [1] * len(input_ids)
        attention_mask = torch.tensor(attention_mask)

        sent_rep_token_ids = [
            i for i, t in enumerate(input_ids) if t == self.tokenizer.cls_token_id
        ]
        sent_rep_mask = torch.tensor([1] * len(sent_rep_token_ids))

        input_ids.unsqueeze_(0)
        attention_mask.unsqueeze_(0)
        sent_rep_mask.unsqueeze_(0)

        scores, _ = self.forward(
            input_ids,
            attention_mask,
            sent_rep_mask=sent_rep_mask,
            sent_rep_token_ids=sent_rep_token_ids,
        )

        return scores
