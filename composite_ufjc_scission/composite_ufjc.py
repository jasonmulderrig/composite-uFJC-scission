"""The module for the composite uFJC single chain scission model."""

# Import external modules
from __future__ import division

# Import internal modules
from .rate_dependence_scission import (
    RateIndependentScission, RateDependentScission
)
from .scission_model import AnalyticalScissionCompositeuFJC


class RateIndependentScissionCompositeuFJC(
        RateIndependentScission, AnalyticalScissionCompositeuFJC):
    """The composite uFJC single-chain model class with rate-independent
    stochastic scission.
    
    This class is a representation of the composite uFJC single-chain
    model with rate-independent stochastic scission; an instance of this
    class is a composite uFJC single-chain model instance with
    rate-independent stochastic scission. It inherits all attributes and
    methods from the ``RateIndependentScission`` class. It also inherits
    all attributes and methods from the
    ``AnalyticalScissionCompositeuFJC`` class, which inherits all
    attributes and methods from the ``CompositeuFJC`` class.
    """
    def __init__(self, **kwargs):
        """
        Initializes the ``RateIndependentScissionCompositeuFJC`` class,
        producing a composite uFJC single-chain model instance with
        rate-independent stochastic scission.
        
        Initialize and inherit all attributes and methods from the
        ``RateIndependentScission`` class instance and the
        ``AnalyticalScissionCompositeuFJC`` class instance.
        """
        RateIndependentScission.__init__(self)
        AnalyticalScissionCompositeuFJC.__init__(self, **kwargs)


class RateDependentScissionCompositeuFJC(
        RateDependentScission, AnalyticalScissionCompositeuFJC):
    """The composite uFJC single-chain model class with rate-dependent
    stochastic scission.
    
    This class is a representation of the composite uFJC single-chain
    model with rate-dependent stochastic scission; an instance of this
    class is a composite uFJC single-chain model instance with
    rate-dependent stochastic scission. It also inherits all attributes
    and methods from the ``RateDependentScission`` class. It also
    inherits all attributes and methods from the
    ``AnalyticalScissionCompositeuFJC`` class, which inherits all
    attributes and methods from the ``CompositeuFJC`` class.
    """
    def __init__(self, **kwargs):
        """
        Initializes the ``RateDependentScissionCompositeuFJC`` class,
        producing a composite uFJC single-chain model instance with
        rate-dependent stochastic scission.
        
        Initialize and inherit all attributes and methods from the
        ``RateDependentScission`` class instance and the
        ``AnalyticalScissionCompositeuFJC`` class instance.
        """
        RateDependentScission.__init__(self, **kwargs)
        AnalyticalScissionCompositeuFJC.__init__(self, **kwargs)
