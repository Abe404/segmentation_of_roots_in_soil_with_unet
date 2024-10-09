"""
Copyright (C) 2019 Abraham Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
from torch.nn.functional import sigmoid
from torch.nn.functional import binary_cross_entropy_with_logits


def dice_loss(predictions, labels):
    """ based on loss function from V-Net paper """
    predictions = sigmoid(predictions).squeeze(1)
    labels = labels.float()
    preds = predictions.contiguous().view(-1)
    labels = labels.view(-1)
    intersection = torch.sum(torch.mul(preds, labels))
    union = torch.sum(preds) + torch.sum(labels)
    return 1 - ((2 * intersection) / (union))


def combined_loss(predictions, labels):
    """ mix of dice and cross entropy """
    ce = 0.3 * binary_cross_entropy_with_logits(predictions.squeeze(1),
                                                labels.to(torch.float32))
    if torch.sum(labels) > 0:
        return dice_loss(predictions, labels) + ce
    else:
        # When no roots use only cross entropy as dice is undefined.
        return ce
