# import necessary libraries
from __future__ import division
from .default_parameters import default_parameters
from .utility import generate_savedir

class CompositeuFJCScissionCharacterizer(object):
    
    """
    Characterizer class for composite uFJCs with scission
    """

    def __init__(self):

        # Parameters
        self.parameters = default_parameters()
        self.set_user_parameters()

        # Setup filesystem
        self.savedir = generate_savedir(self.prefix())
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """
        pass

    def prefix(self):
        return "characterizer"

    def characterization(self):
        pass

    def finalization(self):
        pass