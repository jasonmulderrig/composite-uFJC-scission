"""The module for the composite uFJC scission model specifying the
fundamental scission model
"""

# Import external modules
from __future__ import division
import sys
import numpy as np
from scipy import integrate

# Import internal modules
from .core import CompositeuFJC


class AnalyticalScissionCompositeuFJC(CompositeuFJC):
    """The composite uFJC scission model class specifying the
    fundamental analytical scission model.

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
        CompositeuFJC.__init__(self, **kwargs)

        k_cond = kwargs.get("k_cond", None)

        if k_cond is None:
            error_message = """\
                Error: Need to specify the additive conditioning
                parameter for the rate-independent chain degradation and
                chain damage functions.\
                """
            sys.exit(error_message)
        
        # Retain specified parameters
        self.k_cond = k_cond

        # Parameters needed for numerical calculations
        self.lmbda_nu_hat_inc = 0.0005

        # Calculate and retain numerically calculated parameters
        self.epsilon_nu_diss_hat_crit  = self.epsilon_nu_diss_hat_crit_func()
        self.epsilon_cnu_diss_hat_crit = self.epsilon_cnu_diss_hat_crit_func()
        self.A_nu          = self.A_nu_func()
        self.Lambda_nu_ref = self.lmbda_nu_func(self.A_nu)
    
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
    
    def e_nu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional segment scission
        activation energy barrier function for segment stretches less
        than the critical segment stretch
        
        This function computes the nondimensional segment scission
        activation energy barrier as a function of the applied segment
        stretch according to its analytical form for segment stretches
        less than the critical segment stretch
        """
        if lmbda_nu_hat <= self.lmbda_nu_crit:
            cbrt_arg = (self.zeta_nu_char**2 * self.kappa_nu
                        * (lmbda_nu_hat-1.)**2)
            
            return (0.5 * self.kappa_nu * (lmbda_nu_hat-1.)**2
                    - 1.5*np.cbrt(cbrt_arg) + self.zeta_nu_char)
        else:
            return 0.
    
    def e_nu_sci_hat_prime_rate_independent_analytical_func(self, lmbda_nu_hat):
        """Rate-independent derivative of the analytical form of the
        nondimensional segment scission activation energy barrier
        function for segment stretches less than the critical segment
        stretch. The rate-independent derivative is taken with respect
        to the applied segment stretch.
        
        This function computes the rate-independent derivative of the
        analytical form of the nondimensional segment scission
        activation energy barrier function for segment stretches less
        than the critical segment stretch. The rate-independent
        derivative is taken with respect to the applied segment stretch.
        """
        if lmbda_nu_hat <= 1.:
            return -np.inf
        elif lmbda_nu_hat <= self.lmbda_nu_crit:
            cbrt_arg = self.zeta_nu_char**2 * self.kappa_nu / (lmbda_nu_hat-1.)
            
            return self.kappa_nu * (lmbda_nu_hat-1.) - np.cbrt(cbrt_arg)
        else:
            return 0.
    
    def u_nu_analytical_func(self, lmbda_nu_hat):
        return self.u_nu_func(lmbda_nu_hat)
    
    def s_cnu_analytical_func(self, lmbda_nu_hat):
        def L_func(x):
            return 1. / np.tanh(x) - 1. / x
        if lmbda_nu_hat <= 1.:
            return 0.
        elif lmbda_nu_hat <= self.lmbda_nu_crit:
            xi_c_hat = self.kappa_nu * (lmbda_nu_hat-1.)
            return (L_func(xi_c_hat) * xi_c_hat
                    + np.log(xi_c_hat/np.sinh(xi_c_hat)))
        else:
            xi_c_hat = (
                self.zeta_nu_char**2 / (self.kappa_nu*(lmbda_nu_hat-1.)**3)
            )
            return (L_func(xi_c_hat) * xi_c_hat
                    + np.log(xi_c_hat/np.sinh(xi_c_hat)))
    
    def psi_cnu_analytical_func(self, lmbda_nu_hat):
        return (self.u_nu_analytical_func(lmbda_nu_hat)
                + self.s_cnu_analytical_func(lmbda_nu_hat))
    
    def epsilon_nu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional segment scission
        energy function for segment stretches less than the critical
        segment stretch
        
        This function computes the nondimensional segment scission
        energy as a function of the applied segment stretch according to
        its analytical form for segment stretches less than the critical
        segment stretch
        """
        return (self.psi_cnu_analytical_func(lmbda_nu_hat) 
                + self.zeta_nu_char)
    
    def epsilon_cnu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional chain scission
        energy function for segment stretches less than the critical
        segment stretch
        
        This function computes the nondimensional chain scission
        energy as a function of the applied segment stretch according to
        its analytical form for segment stretches less than the critical
        segment stretch
        """
        return self.epsilon_nu_sci_hat_analytical_func(lmbda_nu_hat)
    
    def u_nu_prime_rate_independent_analytical_func(self, lmbda_nu_hat):
        if lmbda_nu_hat <= 1.:
            return 0.
        elif lmbda_nu_hat <= self.lmbda_nu_crit:
            return self.kappa_nu * (lmbda_nu_hat-1.)
        else:
            return (
                self.zeta_nu_char**2 / (self.kappa_nu*(lmbda_nu_hat-1.)**3)
            )
    
    def s_cnu_prime_rate_independent_analytical_func(self, lmbda_nu_hat):
        def L_prime_func(x):
            return -(1./np.sinh(x))**2 + 1. / x**2
        if lmbda_nu_hat <= 1.:
            return 0.
        elif lmbda_nu_hat <= self.lmbda_nu_crit:
            xi_c_hat = self.kappa_nu * (lmbda_nu_hat-1.)
            xi_c_hat_prime = self.kappa_nu
            return xi_c_hat_prime*xi_c_hat*L_prime_func(xi_c_hat)
        else:
            xi_c_hat = (
                self.zeta_nu_char**2 / (self.kappa_nu*(lmbda_nu_hat-1.)**3)
            )
            xi_c_hat_prime = (
                -3*self.zeta_nu_char**2 / (self.kappa_nu*(lmbda_nu_hat-1.)**4)
            )
            return xi_c_hat_prime*xi_c_hat*L_prime_func(xi_c_hat)
    
    def psi_cnu_prime_rate_independent_analytical_func(self, lmbda_nu_hat):
        return (self.u_nu_prime_rate_independent_analytical_func(lmbda_nu_hat)
                + self.s_cnu_prime_rate_independent_analytical_func(lmbda_nu_hat))
    
    def epsilon_nu_sci_hat_prime_rate_independent_analytical_func(
            self, lmbda_nu_hat):
        """Analytical form of the nondimensional segment scission
        energy function for segment stretches less than the critical
        segment stretch
        
        This function computes the nondimensional segment scission
        energy as a function of the applied segment stretch according to
        its analytical form for segment stretches less than the critical
        segment stretch
        """
        return self.psi_cnu_prime_rate_independent_analytical_func(lmbda_nu_hat)
    
    def epsilon_cnu_sci_hat_prime_rate_independent_analytical_func(
            self, lmbda_nu_hat):
        """Analytical form of the nondimensional chain scission
        energy function for segment stretches less than the critical
        segment stretch
        
        This function computes the nondimensional chain scission
        energy as a function of the applied segment stretch according to
        its analytical form for segment stretches less than the critical
        segment stretch
        """
        return (
            self.epsilon_nu_sci_hat_prime_rate_independent_analytical_func(lmbda_nu_hat)
        )
    
    def p_nu_sci_hat_prime_rate_independent_analytical_func(self, lmbda_nu_hat):
        """Rate-independent derivative of the probability of segment
        scission according to the fundamental analytical scission model
        for segment stretches less than the critical segment stretch.
        The rate-independent derivative is taken with respect to the
        applied segment stretch.
        
        This function computes the rate-independent derivative of the
        probability of segment scission according to the fundamental
        analytical scission model for segment stretches less than the
        critical segment stretch. The rate-independent derivative is
        taken with respect to the applied segment stretch.
        """
        if lmbda_nu_hat <= 1.:
            return 0.
        else:
            return (
                -self.p_nu_sci_hat_func(lmbda_nu_hat)
                * self.e_nu_sci_hat_prime_rate_independent_analytical_func(lmbda_nu_hat)
            )
    
    def p_c_sci_hat_prime_rate_independent_analytical_func(self, lmbda_nu_hat):
        """Rate-independent derivative of the probability of chain
        scission according to the fundamental analytical scission model
        for segment stretches less than the critical segment stretch.
        The rate-independent derivative is taken with respect to the
        applied segment stretch.
        
        This function computes the rate-independent derivative of the
        probability of chain scission according to the fundamental
        analytical scission model for segment stretches less than the
        critical segment stretch. The rate-independent derivative is
        taken with respect to the applied segment stretch.
        """
        if lmbda_nu_hat <= 1.:
            return 0.
        else:
            return (
                self.nu
                * (1.-self.p_nu_sci_hat_func(lmbda_nu_hat))**(self.nu-1)
                * self.p_nu_sci_hat_prime_rate_independent_analytical_func(lmbda_nu_hat)
            )
    
    def epsilon_nu_diss_hat_prime_rate_independent_analytical_func(
            self, lmbda_nu_hat):
        """Rate-independent derivative of the nondimensional
        rate-independent dissipated segment scission energy according to
        the fundamental analytical scission model for segment stretches
        less than the critical segment stretch. The rate-independent
        derivative is taken with respect to the applied segment stretch.
        
        This function computes the rate-independent derivative of the
        nondimensional rate-independent dissipated segment scission
        energy according to the fundamental analytical scission model
        for segment stretches less than the critical segment stretch.
        The rate-independent derivative is taken with respect to the
        applied segment stretch.
        """
        return (
            self.p_nu_sci_hat_prime_rate_independent_analytical_func(lmbda_nu_hat)
            * self.epsilon_nu_sci_hat_func(lmbda_nu_hat)
        )
    
    def epsilon_nu_diss_hat_rate_independent_analytical_func(
            self, lmbda_nu_hat_max, lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
            epsilon_nu_diss_hat_val_prior):
        """Nondimensional rate-independent dissipated segment scission
        energy according to the fundamental analytical scission model
        
        This function computes the nondimensional rate-independent
        dissipated segment scission energy according to the fundamental
        analytical scission model as a function of its prior value and
        the current, prior, and maximum applied segment stretch values.
        """
        # dissipated energy cannot be destroyed
        if lmbda_nu_hat_val < lmbda_nu_hat_max:
            return epsilon_nu_diss_hat_val_prior
        # dissipated energy from fully broken segments remains fixed
        elif lmbda_nu_hat_max > self.lmbda_nu_crit:
            return epsilon_nu_diss_hat_val_prior
        else:
            if (lmbda_nu_hat_val-1.) <= self.lmbda_nu_hat_inc:
                epsilon_nu_diss_hat_prime_val = 0.
            else:
                epsilon_nu_diss_hat_prime_val = (
                    self.epsilon_nu_diss_hat_prime_rate_independent_analytical_func(lmbda_nu_hat_val)
                )
            
            # the lmbda_nu_hat increment here is guaranteed to be 
            # non-negative
            return (epsilon_nu_diss_hat_val_prior 
                    + epsilon_nu_diss_hat_prime_val
                    * (lmbda_nu_hat_val-lmbda_nu_hat_val_prior))
    
    def epsilon_nu_diss_hat_crit_func(self):
        """Nondimensional rate-independent dissipated segment scission
        energy for a chain at the critical state
        
        This function computes the nondimensional rate-independent
        dissipated segment scission energy for a chain at the critical
        state
        """
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
                epsilon_nu_diss_hat_crit_val = 0.
            else:
                epsilon_nu_diss_hat_crit_val = (
                    self.epsilon_nu_diss_hat_rate_independent_analytical_func(
                        lmbda_nu_hat_max, lmbda_nu_hat_val,
                        lmbda_nu_hat_steps[lmbda_nu_hat_indx-1],
                        epsilon_nu_diss_hat_crit_val_prior)
                )
            
            # update values
            epsilon_nu_diss_hat_crit_val_prior = epsilon_nu_diss_hat_crit_val
        
        return epsilon_nu_diss_hat_crit_val
    
    def expctd_val_epsilon_nu_sci_hat_intgrnd_rate_independent_analytical_func(
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
                return 0.
            else:
                epsilon_nu_sci_hat_val = (
                    self.epsilon_nu_sci_hat_analytical_func(lmbda_nu_hat_val)
                )
                p_nu_sci_hat_prime_val = (
                    self.p_nu_sci_hat_prime_rate_independent_analytical_func(lmbda_nu_hat_val)
                )
                epsilon_nu_sci_hat_prime_val = (
                    self.epsilon_nu_sci_hat_prime_rate_independent_analytical_func(lmbda_nu_hat_val)
                )
                return (epsilon_nu_sci_hat_val * p_nu_sci_hat_prime_val
                        / epsilon_nu_sci_hat_prime_val)
    
    def expctd_val_epsilon_nu_sci_hat_cum_intgrl_rate_independent_analytical_func(
            self, expctd_val_epsilon_nu_sci_hat_intgrnd_val,
            epsilon_nu_sci_hat_val,
            expctd_val_epsilon_nu_sci_hat_intgrnd_val_prior,
            epsilon_nu_sci_hat_val_prior,
            expctd_val_epsilon_nu_sci_hat_val_prior):
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
            expctd_val_epsilon_nu_sci_hat_val_prior + np.trapz(
                [expctd_val_epsilon_nu_sci_hat_intgrnd_val_prior,
                expctd_val_epsilon_nu_sci_hat_intgrnd_val],
                x=[epsilon_nu_sci_hat_val_prior, epsilon_nu_sci_hat_val])
        )

    def epsilon_cnu_diss_hat_prime_rate_independent_analytical_func(
            self, lmbda_nu_hat):
        """Rate-independent derivative of the nondimensional
        rate-independent dissipated chain scission energy according to
        the fundamental analytical scission model for segment stretches
        less than the critical segment stretch. The rate-independent
        derivative is taken with respect to the applied segment stretch.
        
        This function computes the rate-independent derivative of the
        nondimensional rate-independent dissipated chain scission
        energy according to the fundamental analytical scission model
        for segment stretches less than the critical segment stretch.
        The rate-independent derivative is taken with respect to the
        applied segment stretch.
        """
        return (
            self.p_c_sci_hat_prime_rate_independent_analytical_func(lmbda_nu_hat)
            * self.epsilon_cnu_sci_hat_func(lmbda_nu_hat)
        )
    
    def epsilon_cnu_diss_hat_rate_independent_analytical_func(
            self, lmbda_nu_hat_max, lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
            epsilon_cnu_diss_hat_val_prior):
        """Nondimensional rate-independent dissipated chain scission
        energy according to the fundamental analytical scission model
        
        This function computes the nondimensional rate-independent
        dissipated chain scission energy according to the fundamental
        analytical scission model as a function of its prior value and
        the current, prior, and maximum applied segment stretch values.
        """
        # dissipated energy cannot be destroyed
        if lmbda_nu_hat_val < lmbda_nu_hat_max:
            return epsilon_cnu_diss_hat_val_prior
        # dissipated energy from fully broken chains remains fixed
        elif lmbda_nu_hat_max > self.lmbda_nu_crit:
            return epsilon_cnu_diss_hat_val_prior
        else:
            if (lmbda_nu_hat_val-1.) <= self.lmbda_nu_hat_inc:
                epsilon_cnu_diss_hat_prime_val = 0.
            else:
                epsilon_cnu_diss_hat_prime_val = (
                    self.epsilon_cnu_diss_hat_prime_rate_independent_analytical_func(lmbda_nu_hat_val)
                )
            
            # the lmbda_nu_hat increment here is guaranteed to be 
            # non-negative
            return (epsilon_cnu_diss_hat_val_prior 
                    + epsilon_cnu_diss_hat_prime_val
                    * (lmbda_nu_hat_val-lmbda_nu_hat_val_prior))
    
    def epsilon_cnu_diss_hat_crit_func(self):
        """Nondimensional rate-independent dissipated chain scission
        energy for a chain at the critical state
        
        This function computes the nondimensional rate-independent
        dissipated chain scission energy for a chain at the critical
        state
        """
        # Define the values of the applied segment stretch to 
        # calculate over
        lmbda_nu_hat_num_steps = (int(
            np.around(
                (self.lmbda_nu_crit-self.lmbda_nu_ref)/self.lmbda_nu_hat_inc))
            + 1)
        lmbda_nu_hat_steps = np.linspace(
            self.lmbda_nu_ref, self.lmbda_nu_crit, lmbda_nu_hat_num_steps)

        # initialization
        lmbda_nu_hat_max                    = 0
        epsilon_cnu_diss_hat_crit_val_prior = 0
        epsilon_cnu_diss_hat_crit_val       = 0

        for lmbda_nu_hat_indx in range(lmbda_nu_hat_num_steps):
            lmbda_nu_hat_val = lmbda_nu_hat_steps[lmbda_nu_hat_indx]
            lmbda_nu_hat_max = max([lmbda_nu_hat_max, lmbda_nu_hat_val])
            
            # preserve initialization and continue on
            if lmbda_nu_hat_indx == 0:
                epsilon_cnu_diss_hat_crit_val = 0.
            else:
                epsilon_cnu_diss_hat_crit_val = (
                    self.epsilon_cnu_diss_hat_rate_independent_analytical_func(
                        lmbda_nu_hat_max, lmbda_nu_hat_val,
                        lmbda_nu_hat_steps[lmbda_nu_hat_indx-1],
                        epsilon_cnu_diss_hat_crit_val_prior)
                )
            
            # update values
            epsilon_cnu_diss_hat_crit_val_prior = epsilon_cnu_diss_hat_crit_val
        
        return epsilon_cnu_diss_hat_crit_val
    
    def expctd_val_epsilon_cnu_sci_hat_intgrnd_rate_independent_analytical_func(
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
        # scission energy from fully broken segments remains fixed
        elif lmbda_nu_hat_max > self.lmbda_nu_crit:
            return (
                expctd_val_epsilon_cnu_sci_hat_intgrnd_val_prior
            )
        else:
            if (lmbda_nu_hat_val-1.) <= self.lmbda_nu_hat_inc:
                return 0.
            else:
                epsilon_cnu_sci_hat_val = (
                    self.epsilon_cnu_sci_hat_analytical_func(lmbda_nu_hat_val)
                )
                p_c_sci_hat_prime_val = (
                    self.p_c_sci_hat_prime_rate_independent_analytical_func(lmbda_nu_hat_val)
                )
                epsilon_cnu_sci_hat_prime_val = (
                    self.epsilon_cnu_sci_hat_prime_rate_independent_analytical_func(lmbda_nu_hat_val)
                )
                return (epsilon_cnu_sci_hat_val * p_c_sci_hat_prime_val
                        / epsilon_cnu_sci_hat_prime_val)
    
    def expctd_val_epsilon_cnu_sci_hat_cum_intgrl_rate_independent_analytical_func(
            self, expctd_val_epsilon_cnu_sci_hat_intgrnd_val,
            epsilon_cnu_sci_hat_val,
            expctd_val_epsilon_cnu_sci_hat_intgrnd_val_prior,
            epsilon_cnu_sci_hat_val_prior,
            expctd_val_epsilon_cnu_sci_hat_val_prior):
        """History-dependent integral of the statistical expected value
        of the rate-independent nondimensional chain scission energy
        
        This function computes the history-dependent integral of the
        statistical expected value of the rate-independent 
        nondimensional chain scission energy as a function of its
        prior value and the current and prior values of both the
        nondimensional chain scission energy and the integrand of the
        statistical expected value of the rate-independent
        nondimensional chain scission energy
        """
        return (
            expctd_val_epsilon_cnu_sci_hat_val_prior + np.trapz(
                [expctd_val_epsilon_cnu_sci_hat_intgrnd_val_prior,
                expctd_val_epsilon_cnu_sci_hat_intgrnd_val],
                x=[epsilon_cnu_sci_hat_val_prior, epsilon_cnu_sci_hat_val])
        )
    
    def I_intgrnd_func(self, lmbda_c_eq, n):
        """Integrand involved in the intact equilibrium chain
        configuration partition function integration
        
        This function computes the integrand involved in the intact 
        equilibrium chain configuration partition function integration
        as a function of the equilibrium chain stretch, integer n, and
        segment number nu
        """
        lmbda_nu = self.lmbda_nu_func(lmbda_c_eq)
        psi_cnu  = self.psi_cnu_func(lmbda_nu, lmbda_c_eq)
        
        return np.exp(-self.nu*(psi_cnu+self.zeta_nu_char)) * lmbda_c_eq**(n+2)
    
    def I_func(self, n):
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
            args=(n,), epsabs=1.0e-12, epsrel=1.0e-12)[0]
    
    # Reference equilibrium chain stretch
    def A_nu_func(self):
        """Reference equilibrium chain stretch
        
        This function numerically computes the reference equilibrium
        chain stretch
        """
        # second moment of the intact chain configuration pdf
        I_2 = self.I_func(2)
        # zeroth moment of the intact chain configuration pdf
        I_0 = self.I_func(0)
        sqrt_arg = (1. / (1.+self.nu*np.exp(-self.epsilon_nu_diss_hat_crit))
                    * I_2 / I_0)
        
        return np.sqrt(sqrt_arg)
