"""The module for the composite uFJC scission model specifying the
fundamental scission model
"""

# Import external modules
from __future__ import division
import numpy as np
from scipy import integrate

# Import internal modules
from .core import CoreCompositeuFJC


class ScissionModelCompositeuFJC(CoreCompositeuFJC):
    """The composite uFJC scission model class specifying the
    fundamental scission model.

    This class contains methods specifying the fundamental scission 
    model, which involve defining both energetic and probabilistic
    quantities. It inherits all attributes and methods from the 
    ``CoreCompositeuFJC`` class.
    """
    def __init__(self, **kwargs):
        """
        Initializes the ``ScissionModelCompositeuFJC`` class. 
        
        Initialize and inherit all attributes and methods from the
        ``CoreCompositeuFJC`` class instance. Calculate and retain
        parameters that intrinsically depend on the fundamental scission
        model in the composite uFJC scission model.
        """
        CoreCompositeuFJC.__init__(self, **kwargs)

        # Parameters needed for numerical calculations
        self.lmbda_nu_hat_inc = 0.0005

        # Calculate and retain numerically calculated parameters
        self.epsilon_nu_diss_hat_crit = self.epsilon_nu_diss_hat_crit_func()
        self.A_nu                     = self.A_nu_func()
        self.Lambda_nu_ref            = self.lmbda_nu_func(self.A_nu)
    
    def p_nu_sci_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of segment scission
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch
        """
        return np.exp(-self.e_nu_sci_hat_func(lmbda_nu_hat))

    def p_nu_sur_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of segment survival
        
        This function computes the rate-independent probability of
        segment survival as a function of the applied segment stretch
        """
        return 1. - self.p_nu_sci_hat_func(lmbda_nu_hat)
    
    def p_c_sur_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of chain survival
        
        This function computes the rate-independent probability of chain
        survival as a function of the applied segment stretch
        """
        return self.p_nu_sur_hat_func(lmbda_nu_hat)**self.nu
    
    def p_c_sci_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of chain scission
        
        This function computes the rate-independent probability of chain
        scission as a function of the applied segment stretch
        """
        return 1. - self.p_c_sur_hat_func(lmbda_nu_hat)
    
    def epsilon_nu_diss_hat_crit_func(self):
        """Nondimensional rate-independent dissipated segment scission
        energy for a chain at the critical state
        
        This function computes the nondimensional rate-independent
        dissipated segment scission energy for a chain at the critical
        state
        """
        def rate_independent_epsilon_nu_diss_hat_func(
                lmbda_nu_hat_max, lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
                rate_independent_epsilon_nu_diss_hat_val_prior):
            """Nondimensional rate-independent dissipated segment
            scission energy
            
            This function computes the nondimensional rate-independent
            dissipated segment scission energy as a function of its
            prior value and the current, prior, and maximum applied
            segment stretch values
            """
            # dissipated energy cannot be destroyed
            if lmbda_nu_hat_val < lmbda_nu_hat_max:
                return rate_independent_epsilon_nu_diss_hat_val_prior
            # dissipated energy from fully broken segments remains fixed
            elif lmbda_nu_hat_max > self.lmbda_nu_crit:
                return rate_independent_epsilon_nu_diss_hat_val_prior
            else:
                if (lmbda_nu_hat_val-1.) <= self.lmbda_nu_hat_inc:
                    epsilon_nu_diss_hat_prime_val = 0
                else:
                    cbrt_arg = (self.zeta_nu_char**2 * self.kappa_nu
                                / (lmbda_nu_hat_val-1.))
                    epsilon_nu_diss_hat_prime_val = (
                        self.p_nu_sci_hat_func(lmbda_nu_hat_val) 
                        * (np.cbrt(cbrt_arg)-self.kappa_nu*(lmbda_nu_hat_val-1.))
                        * self.epsilon_nu_sci_hat_func(lmbda_nu_hat_val)
                    )
                
                # the lmbda_nu_hat increment here is guaranteed to be 
                # non-negative
                return (rate_independent_epsilon_nu_diss_hat_val_prior 
                        + epsilon_nu_diss_hat_prime_val
                        * (lmbda_nu_hat_val-lmbda_nu_hat_val_prior))
        
        # Define the values of the applied segment stretch to 
        # calculate over
        lmbda_nu_hat_num_steps = (int(
            np.around(
                (self.lmbda_nu_crit-self.lmbda_nu_ref)/self.lmbda_nu_hat_inc))
            + 1)
        lmbda_nu_hat_steps = np.linspace(
            self.lmbda_nu_ref, self.lmbda_nu_crit, lmbda_nu_hat_num_steps)

        # initialization
        lmbda_nu_hat_max                   = 0
        epsilon_nu_diss_hat_crit_val_prior = 0
        epsilon_nu_diss_hat_crit_val       = 0

        for lmbda_nu_hat_indx in range(lmbda_nu_hat_num_steps):
            lmbda_nu_hat_val = lmbda_nu_hat_steps[lmbda_nu_hat_indx]
            lmbda_nu_hat_max = max([lmbda_nu_hat_max, lmbda_nu_hat_val])
            
            # preserve initialization and continue on
            if lmbda_nu_hat_indx == 0:
                epsilon_nu_diss_hat_crit_val = 0
            else:
                epsilon_nu_diss_hat_crit_val = (
                    rate_independent_epsilon_nu_diss_hat_func(
                        lmbda_nu_hat_max, lmbda_nu_hat_val,
                        lmbda_nu_hat_steps[lmbda_nu_hat_indx-1],
                        epsilon_nu_diss_hat_crit_val_prior)
                )
            
            # update values
            epsilon_nu_diss_hat_crit_val_prior = epsilon_nu_diss_hat_crit_val
        
        return epsilon_nu_diss_hat_crit_val
    
    def I_intgrnd_func(self, lmbda_c_eq, n, nu):
        """Integrand involved in the intact equilibrium chain
        configuration partition function integration
        
        This function computes the integrand involved in the intact 
        equilibrium chain configuration partition function integration
        as a function of the equilibrium chain stretch, integer n, and
        segment number nu
        """
        lmbda_nu = self.lmbda_nu_func(lmbda_c_eq)
        psi_cnu  = self.psi_cnu_func(lmbda_nu, lmbda_c_eq)
        
        return np.exp(-nu*(psi_cnu+self.zeta_nu_char)) * lmbda_c_eq**(n-1)
    
    def I_func(self, n, nu):
        """Intact equilibrium chain configuration partition function
        integration through all admissible end-to-end chain distances up
        to the critical point
        
        This function numerically computes the intact equilibrium chain
        configuration partition function through all admissible
        end-to-end chain distances up to the critical point as a 
        function of integer n and segment number nu
        """
        return integrate.quad(
            self.I_intgrnd_func, self.lmbda_c_eq_ref, self.lmbda_c_eq_crit,
            args=(n,nu), epsabs=1.0e-12, epsrel=1.0e-12)[0]
    
    # Reference equilibrium chain stretch
    def A_nu_func(self):
        """Reference equilibrium chain stretch
        
        This function numerically computes the reference equilibrium
        chain stretch
        """
        I_5 = self.I_func(5, self.nu)
        I_3 = self.I_func(3, self.nu)
        sqrt_arg = (1. / (1.+self.nu*np.exp(-self.epsilon_nu_diss_hat_crit))
                    * I_5 / I_3)
        
        return np.sqrt(sqrt_arg)
