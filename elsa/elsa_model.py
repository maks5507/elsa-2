#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import torch.nn as nn
from extractive import TransformerSum, Centroid, Textrank
from elsa import preprocessing, AbstractiveModel


class Elsa(nn.Module):
    extractive_models = {
        'bertsum': TransformerSum,
        'transformersum': TransformerSum,
        'centroid': Centroid,
        'textrank': Textrank
    }

    def __init__(self, extractive_model, extractive_init_params, abstractive_model, dataset, hidden_size):
        super(Elsa, self).__init__()
        self.extractive = self.extractive_models[extractive_model](**extractive_init_params)
        self.abstractive = AbstractiveModel(abstractive_model, dataset)

        self.sentence_tokenizer = preprocessing.SentenceTokenizer()

        self.linear1 = nn.Linear(512, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 512)
        self.relu2 = nn.ReLU()

    def forward(self, texts):
        scores = []
        for text in texts:
            current_scores = self.extractive(text)
            current_scores = self.linear1(current_scores)
            current_scores = self.relu1(current_scores)
            current_scores = self.linear2(current_scores)
            current_scores = self.relu2(current_scores)
            scores += current_scores

        sentences = []
        for text in texts:
            sentences += [self.sentence_tokenizer.tokenize(text)]

        return self.abstractive_model(sentences, scores)

    def generate(self, texts, **generation_params):
        scores = []
        for text in texts:
            current_scores = self.extractive(text)
            current_scores = self.linear1(current_scores)
            current_scores = self.relu1(current_scores)
            current_scores = self.linear2(current_scores)
            current_scores = self.relu2(current_scores)
            scores += current_scores

        sentences = []
        for text in texts:
            sentences += [self.sentence_tokenizer.tokenize(text)]

        return self.abstractive_model.generate(sentences, scores, **generation_params)

