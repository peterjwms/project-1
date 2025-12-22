"""
The utility module contains helper functions/classes that are not related
to internal functionalities and attributes of core modules. That means, the
functions/classes within this module can only be imported from other modules.

Instructions:
---
* The only provided function is ``to_level`` which normalizes the string of "sense"
within the PDTB dataset by reducing its sense level from a higher level to a lower
one. Usage is described in its docstring.
* Other than that, you're welcome to add any functionalities in this module
"""
import torch
import torch.nn as nn
from tqdm import tqdm


def to_level(sense: str, level: int = 2) -> str:
    """converts a sense in string to a desired level

    There are 3 sense levels in PDTB:
        Level 1 senses are the single-word senses like `Temporal` and `Contingency`.
        Level 2 senses add an additional sub-level sense on top of Level 1 senses, as in `Expansion.Exception`
        Level 3 senses adds yet another sub-level sense, as in `Temporal.Asynchronous.Precedence`.

    This function is used to ensure that all senses do not exceed the desired
    sense level provided as the argument `level`. For example,
    >>> to_level('Expansion.Restatement', level=1)
    'Expansion'
    >>> to_level('Temporal.Asynchronous.Succession', level=2)
    'Temporal.Asynchronous'

    When the input sense has a lower sense level than the desired sense level,
    this function will retain the original sense string. For example,

    >>> to_level('Expansion', level=2)
    'Expansion'
    >>> to_level('Comparison.Contrast', level=3)
    'Comparison.Contrast'

    Args:
        sense (str): a sense as given in any of the PDTB data files
        level (int): a desired sense level

    Returns:
        str: a sense below or at the desired sense level
    """
    s_split = sense.split(".")
    s_join = ".".join(s_split[:level])
    return s_join


def accuracy(y, preds):
    """computes the accuracy given the true labels and predicted labels"""
    num_correct = (y == preds).sum().item()
    total = y.size(0)
    return num_correct / total


def precision(y, preds):
    """computes the precision given the true labels and predicted labels"""
    # TODO: make sure this works for multi-class classification
    true_positives = ((y == 1) & (preds == 1)).sum().item()
    predicted_positives = (preds == 1).sum().item()
    if predicted_positives == 0:
        return 0.0
    return true_positives / predicted_positives


def recall(y, preds):
    """computes the recall given the true labels and predicted labels"""
    # TODO: make sure this works for multi-class classification
    true_positives = ((y == 1) & (preds == 1)).sum().item()
    actual_positives = (y == 1).sum().item()
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives


def f1_score(y, preds):
    """computes the F1 score given the true labels and predicted labels"""
    p = precision(y, preds)
    r = recall(y, preds)
    if (p + r) == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def grid_search():
    pass  # Placeholder for actual implementation
