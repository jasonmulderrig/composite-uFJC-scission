"""The single-chain module for the composite uFJC scission model"""

# Import external modules
from __future__ import division

# Import internal modules
from .rate_dependence_scission import RateDependenceScissionCompositeuFJC


class CompositeuFJC(RateDependenceScissionCompositeuFJC):
    """The composite uFJC scission single-chain model class."""
    def __init__(self, **kwargs):
        """
        Initializes the ``CompositeuFJC`` class, producing a composite
        uFJC scission single chain model instance.
        """
        RateDependenceScissionCompositeuFJC.__init__(self, **kwargs)