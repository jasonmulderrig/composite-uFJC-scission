"""The module for the composite uFJC scission model specifying the
rate-dependent nature of scission
"""

# Import external modules
from __future__ import division
import numpy as np
from scipy import integrate

# Import internal modules
from .scission_model import ScissionModelCompositeuFJC


class RateDependenceScissionCompositeuFJC(ScissionModelCompositeuFJC):
    """The composite uFJC scission model class specifying the
    rate-dependent nature of scission.

    This class contains methods specifying the rate-dependent nature of
    scission for the composite uFJC chain model, which involve defining
    both energetic and probabilistic quantities. It inherits all
    attributes and methods from the ``ScissionModelCompositeuFJC``
    class, which inherits all attributes and methods from the
    ``CoreCompositeuFJC`` class.
    """
    def __init__(self, **kwargs):
        """
        Initializes the ``RateDependenceScissionCompositeuFJC`` class. 
        
        Initialize and inherit all attributes and methods from the
        ``ScissionModelCompositeuFJC`` class instance
        """
        ScissionModelCompositeuFJC.__init__(self, **kwargs)
    
    def p_nu_sci_hat_cum_intgrl_func(
            self, p_nu_sci_hat_val, t_val, p_nu_sci_hat_prior, t_prior, 
            p_nu_sci_hat_cum_intgrl_val_prior):
        """History-dependent time integral of the probability of segment
        scission
        
        This function computes the history-dependent time integral of
        the probability of segment scission as a function of its prior
        value and the current and prior values of both the probability
        of segment scission and time
        """
        return p_nu_sci_hat_cum_intgrl_val_prior + integrate.trapezoid(
            [p_nu_sci_hat_prior, p_nu_sci_hat_val], x=[t_prior, t_val])
    
    def gamma_nu_func(self, p_nu_sci_hat_cum_intgrl_val):
        """Rate-dependent probability of segment scission
        
        This function computes the rate-dependent probability of segment
        scission as a function of the history-dependent time integral of
        the probability of segment scission
        """
        return 1. - np.exp(-self.omega_0*p_nu_sci_hat_cum_intgrl_val)
    
    def rho_nu_func(self, p_nu_sci_hat_cum_intgrl_val):
        """Rate-dependent probability of segment survival
        
        This function computes the rate-dependent probability of segment
        survival as a function of the history-dependent time integral of
        the probability of segment scission
        """
        return 1. - self.gamma_nu_func(p_nu_sci_hat_cum_intgrl_val)
    
    def rate_dependent_epsilon_nu_diss_hat_func(
            self, p_nu_sci_hat_val, p_nu_sci_hat_cum_intgrl_val, t_val,
            lmbda_nu_hat_val, t_prior, epsilon_nu_diss_hat_val_prior):
        """Nondimensional rate-dependent dissipated segment scission
        energy
        
        This function computes the nondimensional rate-dependent
        dissipated segment scission energy as a function of its prior
        value, the current applied segment stretch, the current and
        prior values of time, the current probability of segment
        scission, and the history-dependent time integral of the
        probability of segment scission
        """
        epsilon_nu_sci_hat_val = self.epsilon_nu_sci_hat_func(lmbda_nu_hat_val)
        
        epsilon_nu_diss_hat_dot_val = (
            np.exp(-self.omega_0*p_nu_sci_hat_cum_intgrl_val) * self.omega_0
            * p_nu_sci_hat_val * epsilon_nu_sci_hat_val
        )
        
        return (epsilon_nu_diss_hat_val_prior + epsilon_nu_diss_hat_dot_val
                * (t_val-t_prior))
    
    def rate_dependent_expctd_val_epsilon_nu_sci_hat_intgrnd_func(
            self, p_nu_sci_hat_val, p_nu_sci_hat_cum_intgrl_val, t_val,
            lmbda_nu_hat_val, t_prior, epsilon_nu_sci_hat_val_prior):
        """Integrand of the statistical expected value of the
        rate-dependent nondimensional segment scission energy
        
        This function computes the integrand of the statistical expected
        value of the rate-dependent nondimensional segment scission
        energy as a function of the prior value of the nondimensional
        segment scission energy, the current applied segment stretch,
        the current and prior values of time, the current probability of
        segment scission, and the history-dependent time integral of the
        probability of segment scission
        """
        epsilon_nu_sci_hat_val = self.epsilon_nu_sci_hat_func(lmbda_nu_hat_val)

        epsilon_nu_sci_hat_dot_val = (
            (epsilon_nu_sci_hat_val-epsilon_nu_sci_hat_val_prior)
            / (t_val-t_prior))
        
        return (epsilon_nu_sci_hat_val
                * np.exp(-self.omega_0*p_nu_sci_hat_cum_intgrl_val)
                * self.omega_0 * p_nu_sci_hat_val / epsilon_nu_sci_hat_dot_val)
    
    def rate_dependent_expctd_val_epsilon_nu_sci_hat_cum_intgrl_func(
            self, expctd_val_epsilon_nu_sci_hat_intgrnd_val,
            epsilon_nu_sci_hat_val,
            expctd_val_epsilon_nu_sci_hat_intgrnd_val_prior,
            epsilon_nu_sci_hat_prior, expctd_val_epsilon_nu_sci_hat_val_prior):
        """History-dependent integral of the statistical expected value
        of the rate-dependent nondimensional segment scission energy
        
        This function computes the history-dependent integral of the
        statistical expected value of the rate-dependent 
        nondimensional segment scission energy as a function of its
        prior value and the current and prior values of both the
        nondimensional segment scission energy and the integrand of the
        statistical expected value of the rate-dependent
        nondimensional segment scission energy
        """
        return (
            expctd_val_epsilon_nu_sci_hat_val_prior + integrate.trapezoid(
                [expctd_val_epsilon_nu_sci_hat_intgrnd_val_prior,
                expctd_val_epsilon_nu_sci_hat_intgrnd_val],
                x=[epsilon_nu_sci_hat_prior, epsilon_nu_sci_hat_val])
        )
    
    def gamma_c_func(self, p_nu_sci_hat_cum_intgrl_val):
        """Rate-dependent probability of chain scission
        
        This function computes the rate-dependent probability of chain
        scission as a function of the history-dependent time integral of
        the probability of segment scission
        """
        return 1. - np.exp(-self.nu*self.omega_0*p_nu_sci_hat_cum_intgrl_val)
    
    def rho_c_func(self, p_nu_sci_hat_cum_intgrl_val):
        """Rate-dependent probability of chain survival
        
        This function computes the rate-dependent probability of chain
        survival as a function of the history-dependent time integral of
        the probability of segment scission
        """
        return 1. - self.gamma_c_func(p_nu_sci_hat_cum_intgrl_val)
    
    def chi_c_func(self, k_cond_val, p_nu_sci_hat_cum_intgrl_val):
        """Rate-dependent chain degradation
        
        This function computes the rate-dependent chain degradation as
        a function of an additive conditioning parameter and the
        history-dependent time integral of the probability of segment
        scission
        """
        return ((1.-k_cond_val) * self.rho_c_func(p_nu_sci_hat_cum_intgrl_val)
                + k_cond_val)
    
    def omicron_c_func(self, k_cond_val, p_nu_sci_hat_cum_intgrl_val):
        """Rate-dependent chain damage
        
        This function computes the rate-dependent chain damage as a
        function of an additive conditioning parameter and the
        history-dependent time integral of the probability of segment
        scission
        """
        return 1. - self.chi_c_func(k_cond_val, p_nu_sci_hat_cum_intgrl_val)
    
    def rate_dependent_epsilon_cnu_diss_hat_func(
            self, p_nu_sci_hat_val, p_nu_sci_hat_cum_intgrl_val, t_val,
            lmbda_nu_hat_val, t_prior, epsilon_cnu_diss_hat_val_prior):
        """Nondimensional rate-dependent dissipated chain scission
        energy
        
        This function computes the nondimensional rate-dependent
        dissipated chain scission energy as a function of its prior
        value, the current applied segment stretch, the current and
        prior values of time, the current probability of segment
        scission, and the history-dependent time integral of the
        probability of segment scission
        """
        epsilon_nu_sci_hat_val = self.epsilon_nu_sci_hat_func(lmbda_nu_hat_val)
        
        epsilon_cnu_diss_hat_dot_val = (
            np.exp(-self.nu*self.omega_0*p_nu_sci_hat_cum_intgrl_val) * self.nu
            * self.omega_0 * p_nu_sci_hat_val * epsilon_nu_sci_hat_val
        )
        
        return (epsilon_cnu_diss_hat_val_prior + epsilon_cnu_diss_hat_dot_val
                * (t_val-t_prior))
    
    def rate_dependent_expctd_val_epsilon_cnu_sci_hat_intgrnd_func(
            self, p_nu_sci_hat_val, p_nu_sci_hat_cum_intgrl_val, t_val,
            lmbda_nu_hat_val, t_prior, epsilon_cnu_sci_hat_val_prior):
        """Integrand of the statistical expected value of the
        rate-dependent nondimensional chain scission energy
        
        This function computes the integrand of the statistical expected
        value of the rate-dependent nondimensional chain scission
        energy as a function of the prior value of the nondimensional
        chain scission energy, the current applied segment stretch,
        the current and prior values of time, the current probability of
        segment scission, and the history-dependent time integral of the
        probability of segment scission
        """
        epsilon_cnu_sci_hat_val = self.epsilon_cnu_sci_hat_func(
            lmbda_nu_hat_val)

        epsilon_cnu_sci_hat_dot_val = (
            (epsilon_cnu_sci_hat_val-epsilon_cnu_sci_hat_val_prior)
            / (t_val-t_prior))
        
        return (epsilon_cnu_sci_hat_val
                * np.exp(-self.nu*self.omega_0*p_nu_sci_hat_cum_intgrl_val)
                * self.nu * self.omega_0 * p_nu_sci_hat_val
                / epsilon_cnu_sci_hat_dot_val)
    
    def rate_dependent_expctd_val_epsilon_cnu_sci_hat_cum_intgrl_func(
            self, expctd_val_epsilon_cnu_sci_hat_intgrnd_val,
            epsilon_cnu_sci_hat_val,
            expctd_val_epsilon_cnu_sci_hat_intgrnd_val_prior,
            epsilon_cnu_sci_hat_prior,
            expctd_val_epsilon_cnu_sci_hat_val_prior):
        """History-dependent integral of the statistical expected value
        of the rate-dependent nondimensional chain scission energy
        
        This function computes the history-dependent integral of the
        statistical expected value of the rate-dependent 
        nondimensional chain scission energy as a function of its
        prior value and the current and prior values of both the
        nondimensional chain scission energy and the integrand of the
        statistical expected value of the rate-dependent
        nondimensional chain scission energy
        """
        return (
            expctd_val_epsilon_cnu_sci_hat_val_prior + integrate.trapezoid(
                [expctd_val_epsilon_cnu_sci_hat_intgrnd_val_prior,
                expctd_val_epsilon_cnu_sci_hat_intgrnd_val],
                x=[epsilon_cnu_sci_hat_prior, epsilon_cnu_sci_hat_val])
        )
    
    def rate_independent_epsilon_nu_diss_hat_func(
            self, lmbda_nu_hat_max, lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
            epsilon_nu_diss_hat_val_prior):
        """Nondimensional rate-independent dissipated segment scission
        energy
        
        This function computes the nondimensional rate-independent
        dissipated segment scission energy as a function of its prior
        value and the current, prior, and maximum applied segment
        stretch values
        """
        # dissipated energy cannot be destroyed
        if lmbda_nu_hat_val < lmbda_nu_hat_max:
            return epsilon_nu_diss_hat_val_prior
        # dissipated energy from fully broken segments remains fixed
        elif lmbda_nu_hat_max > self.lmbda_nu_crit:
            return epsilon_nu_diss_hat_val_prior
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
            return (epsilon_nu_diss_hat_val_prior 
                    + epsilon_nu_diss_hat_prime_val
                    * (lmbda_nu_hat_val-lmbda_nu_hat_val_prior))
    
    def rate_independent_expctd_val_epsilon_nu_sci_hat_intgrnd_func(
            self, lmbda_nu_hat_max, lmbda_nu_hat_val, 
            expctd_val_epsilon_nu_sci_hat_intgrnd_val_prior):
        """Integrand of the statistical expected value of the
        rate-independent nondimensional segment scission energy
        
        This function computes the integrand of the statistical expected
        value of the rate-independent nondimensional segment scission
        energy as a function of its prior value and the current and
        maximum applied segment stretch values
        """
        # statistical expected value of the nondimensional segment
        # scission energy cannot be destroyed
        if lmbda_nu_hat_val < lmbda_nu_hat_max:
            return (
                expctd_val_epsilon_nu_sci_hat_intgrnd_val_prior
            )
        # statistical expected value of the nondimensional segment
        # scission energy from fully broken segments remains fixed
        elif lmbda_nu_hat_max > self.lmbda_nu_crit:
            return (
                expctd_val_epsilon_nu_sci_hat_intgrnd_val_prior
            )
        else:
            if (lmbda_nu_hat_val-1.) <= self.lmbda_nu_hat_inc:
                return 0
            else:
                epsilon_nu_sci_hat_val = (
                    self.epsilon_nu_sci_hat_func(lmbda_nu_hat_val)
                )
                cbrt_arg = (self.zeta_nu_char**2 * self.kappa_nu
                            / (lmbda_nu_hat_val-1.))
                p_nu_sci_hat_prime_val = (
                    self.p_nu_sci_hat_func(lmbda_nu_hat_val)
                    * (np.cbrt(cbrt_arg)-self.kappa_nu*(lmbda_nu_hat_val-1.))
                )
                sinh_arg = self.kappa_nu*(lmbda_nu_hat_val-1.)
                epsilon_nu_sci_hat_prime_val = (
                    1. / (lmbda_nu_hat_val-1.)
                    - (self.kappa_nu/np.sinh(sinh_arg))**2
                    * (lmbda_nu_hat_val-1.) 
                    + self.kappa_nu * (lmbda_nu_hat_val-1.)
                )
                return (epsilon_nu_sci_hat_val * p_nu_sci_hat_prime_val
                        / epsilon_nu_sci_hat_prime_val)
    
    def rate_independent_expctd_val_epsilon_nu_sci_hat_cum_intgrl_func(
            self, expctd_val_epsilon_nu_sci_hat_intgrnd_val,
            epsilon_nu_sci_hat_val,
            expctd_val_epsilon_nu_sci_hat_intgrnd_val_prior,
            epsilon_nu_sci_hat_prior, expctd_val_epsilon_nu_sci_hat_val_prior):
        """History-dependent integral of the statistical expected value
        of the rate-independent nondimensional segment scission energy
        
        This function computes the history-dependent integral of the
        statistical expected value of the rate-independent 
        nondimensional segment scission energy as a function of its
        prior value and the current and prior values of both the
        nondimensional segment scission energy and the integrand of the
        statistical expected value of the rate-independent
        nondimensional segment scission energy
        """
        return (
            expctd_val_epsilon_nu_sci_hat_val_prior + integrate.trapezoid(
                [expctd_val_epsilon_nu_sci_hat_intgrnd_val_prior,
                expctd_val_epsilon_nu_sci_hat_intgrnd_val],
                x=[epsilon_nu_sci_hat_prior, epsilon_nu_sci_hat_val])
        )
    
    def upsilon_c_func(self, k_cond_val, lmbda_nu_hat):
        """Rate-independent chain degradation
        
        This function computes the rate-independent chain degradation as
        a function of an additive conditioning parameter and the applied
        segment stretch
        """
        return ((1.-k_cond_val) * self.p_c_sur_hat_func(lmbda_nu_hat)
                + k_cond_val)
    
    def d_c_func(self, k_cond_val, lmbda_nu_hat):
        """Rate-independent chain damage
        
        This function computes the rate-independent chain damage as a
        function of an additive conditioning parameter and the applied
        segment stretch
        """
        return 1. - self.upsilon_c_func(k_cond_val, lmbda_nu_hat)
    
    def rate_independent_epsilon_cnu_diss_hat_func(
            self, lmbda_nu_hat_max, lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
            epsilon_cnu_diss_hat_val_prior):
        """Nondimensional rate-independent dissipated chain scission
        energy
        
        This function computes the nondimensional rate-independent
        dissipated chain scission energy as a function of its prior
        value, as well as the current, prior, and maximum applied
        segment stretch values.
        """
        # dissipated energy cannot be destroyed
        if lmbda_nu_hat_val < lmbda_nu_hat_max:
            return epsilon_cnu_diss_hat_val_prior
        # dissipated energy from fully broken chains remains fixed
        elif lmbda_nu_hat_max > self.lmbda_nu_crit:
            return epsilon_cnu_diss_hat_val_prior
        else:
            if (lmbda_nu_hat_val-1.) <= self.lmbda_nu_hat_inc:
                epsilon_cnu_diss_hat_prime_val = 0
            else:
                cbrt_arg = (self.zeta_nu_char**2 * self.kappa_nu 
                            / (lmbda_nu_hat_val-1.))
                epsilon_cnu_diss_hat_prime_val = (
                    self.nu 
                    * (1.
                    -self.p_nu_sci_hat_func(lmbda_nu_hat_val))**(self.nu-1)
                    * self.p_nu_sci_hat_func(lmbda_nu_hat_val)
                    * (np.cbrt(cbrt_arg)-self.kappa_nu
                    *(lmbda_nu_hat_val-1.))
                    * self.epsilon_nu_sci_hat_func(lmbda_nu_hat_val)
                )
            
            # the lmbda_nu_hat increment here is guaranteed to be 
            # non-negative
            return (epsilon_cnu_diss_hat_val_prior 
                    + epsilon_cnu_diss_hat_prime_val
                    * (lmbda_nu_hat_val-lmbda_nu_hat_val_prior))
    
    def rate_independent_expctd_val_epsilon_cnu_sci_hat_intgrnd_func(
            self, lmbda_nu_hat_max, lmbda_nu_hat_val,
            expctd_val_epsilon_cnu_sci_hat_intgrnd_val_prior):
        """Integrand of the statistical expected value of the
        rate-independent nondimensional chain scission energy
        
        This function computes the integrand of the statistical expected
        value of the rate-independent nondimensional chain scission
        energy as a function of its prior value and the current and
        maximum applied segment stretch values
        """
        # statistical expected value of the nondimensional chain
        # scission energy cannot be destroyed
        if lmbda_nu_hat_val < lmbda_nu_hat_max:
            return (
                expctd_val_epsilon_cnu_sci_hat_intgrnd_val_prior
            )
        # statistical expected value of the nondimensional chain
        # scission energy from fully broken chain remains fixed
        elif lmbda_nu_hat_max > self.lmbda_nu_crit:
            return (
                expctd_val_epsilon_cnu_sci_hat_intgrnd_val_prior
            )
        else:
            if (lmbda_nu_hat_val-1.) <= self.lmbda_nu_hat_inc:
                return 0
            else:
                epsilon_cnu_sci_hat_val = (
                    self.epsilon_cnu_sci_hat_func(lmbda_nu_hat_val)
                )
                cbrt_arg = (self.zeta_nu_char**2 * self.kappa_nu
                            / (lmbda_nu_hat_val-1.))
                p_c_sci_hat_prime_val = (self.nu
                    * (1.
                    -self.p_nu_sci_hat_func(lmbda_nu_hat_val))**(self.nu-1)
                    * self.p_nu_sci_hat_func(lmbda_nu_hat_val)
                    * (np.cbrt(cbrt_arg)-self.kappa_nu*(lmbda_nu_hat_val-1.))
                )
                sinh_arg = self.kappa_nu*(lmbda_nu_hat_val-1.)
                epsilon_cnu_sci_hat_prime_val = (
                    1. / (lmbda_nu_hat_val-1.)
                    - (self.kappa_nu/np.sinh(sinh_arg))**2
                    * (lmbda_nu_hat_val-1.) 
                    + self.kappa_nu * (lmbda_nu_hat_val-1.)
                )
                return (epsilon_cnu_sci_hat_val * p_c_sci_hat_prime_val
                        / epsilon_cnu_sci_hat_prime_val)
    
    def rate_independent_expctd_val_epsilon_cnu_sci_hat_cum_intgrl_func(
            self, expctd_val_epsilon_cnu_sci_hat_intgrnd_val,
            epsilon_cnu_sci_hat_val,
            expctd_val_epsilon_cnu_sci_hat_intgrnd_val_prior,
            epsilon_cnu_sci_hat_prior,
            expctd_val_epsilon_cnu_sci_hat_val_prior):
        """History-dependent integral of the statistical expected value
        of the rate-independent nondimensional chain scission energy
        
        This function computes the history-dependent integral of the
        statistical expected value of the rate-independent
        nondimensional chain scission energy as a function of its prior
        value and the current and prior values of both the
        nondimensional chain scission energy and the integrand of the
        statistical expected value of the rate-independent
        nondimensional chain scission energy
        """
        return (
            expctd_val_epsilon_cnu_sci_hat_val_prior + integrate.trapezoid(
                [expctd_val_epsilon_cnu_sci_hat_intgrnd_val_prior,
                expctd_val_epsilon_cnu_sci_hat_intgrnd_val],
                x=[epsilon_cnu_sci_hat_prior, epsilon_cnu_sci_hat_val])
        )