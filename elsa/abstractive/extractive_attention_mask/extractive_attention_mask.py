#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import torch
from torch.nn import functional as F
from typing import List


class ExtractiveAttentionMask:
    def __call__(self, mapping: List[int], sentences_scores: List[float]) -> torch.Tensor:
        attention_mask = torch.zeros(512)
        mapping_tensor = torch.Tensor(mapping)

        for i, score in enumerate(sentences_scores):
            mask = torch.where(mapping_tensor == i, torch.tensor(1), torch.tensor(0)).to(bool)
            attention_mask[mask] = score

        mask = torch.where(mapping_tensor == -1, torch.tensor(1), torch.tensor(0)).to(bool)
        attention_mask[mask] = 0.

        mask = torch.where(mapping_tensor == -2, torch.tensor(1), torch.tensor(0)).to(bool)
        attention_mask[mask] = 0

        attention_mask = F.softmax(attention_mask, dim=-1)
        attention_mask[mask] = float('inf')

        return attention_mask.unsqueeze(0)
