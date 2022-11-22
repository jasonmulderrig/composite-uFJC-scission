"""The module for the composite uFJC single chain scission model"""

# Import external modules
from __future__ import division

# Import internal modules
from .rate_dependence_scission import RateDependenceScissionCompositeuFJC


class CompositeuFJC(RateDependenceScissionCompositeuFJC):
    """The composite uFJC single-chain scission model class.
    
    This class is a representation of the composite uFJC single-chain
    scission model; an instance of this class is a composite uFJC
    single-chain scission model instance. It inherits all attributes and
    methods from the ``RateDependenceScissionCompositeuFJC`` class,
    which inherits all attributes and methods from the 
    ``ScissionModelCompositeuFJC`` class, which inherits all attributes
    and methods from the ``CoreCompositeuFJC`` class.
    """
    def __init__(self, **kwargs):
        """
        Initializes the ``CompositeuFJC`` class, producing a composite
        uFJC scission single chain model instance. 
        
        Initialize and inherit all attributes and methods from the
        ``RateDependenceScissionCompositeuFJC`` class instance
        """
        RateDependenceScissionCompositeuFJC.__init__(self, **kwargs)