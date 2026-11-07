"""Measuring model quality by log-likelihood.

likelihood -- how much probability the model gives held-out text (val NLL,
              perplexity, bits/token, and loss by position in the context)
hellaswag  -- whether higher likelihood tracks the *sensible* continuation:
              multiple-choice endings ranked by their log-likelihood
"""

from evaluation.hellaswag import evaluate_hellaswag
from evaluation.likelihood import val_nll

__all__ = ['evaluate_hellaswag', 'val_nll']
