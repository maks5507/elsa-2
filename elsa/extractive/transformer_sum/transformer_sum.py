#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import torch.nn as nn
from spacy.lang.en import English
from .extractive import ExtractiveSummarizer


class TransformerSum(nn.Module):
    def __init__(self, pretrained_model_path):
        super(TransformerSum, self).__init__()
        self.model = ExtractiveSummarizer.load_from_checkpoint(pretrained_model_path, strict=False)
        self.tokenizer = English()

    def summarize(self, sentences, embeddings=None, factor=None):
        src_txt = [
            " ".join([token.text for token in self.tokenizer(sent)
                      if str(token) != "."]) + "."
            for sent in sentences
        ]

        scores = self.model.predict(src_txt)
        return scores.detach().numpy().flatten().tolist()

    def forward(self, tokenized_texts, **model_params):
        return self.model.predict(tokenized_texts, **model_params)