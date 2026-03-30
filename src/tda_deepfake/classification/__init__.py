"""Classification module.

Trains and evaluates simple, interpretable classifiers (SVM, logistic regression)
on topological feature vectors. Deliberate simplicity isolates the contribution
of topological features from classifier complexity.
"""

from .classifier import Classifier

__all__ = ["Classifier"]
