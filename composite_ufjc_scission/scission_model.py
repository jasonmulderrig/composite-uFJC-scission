"""The module for the composite uFJC scission model specifying the
fundamental analytical scission model.
"""

# Import external modules
from __future__ import division
import sys
import numpy as np

# Import internal modules
from .core import CompositeuFJC


class AnalyticalScissionCompositeuFJC(CompositeuFJC):
    """The composite uFJC scission model class specifying the
    fundamental analytical scission model.

    This class contains methods specifying the fundamental scission 
    model, which involve defining both energetic and probabilistic
    quantities. It inherits all attributes and methods from the 
    ``CompositeuFJC`` class.
    """
    def __init__(self, **kwargs):
        """
        Initializes the ``AnalyticalScissionModelCompositeuFJC`` class.
        
        Initialize and inherit all attributes and methods from the
        ``CompositeuFJC`` class instance. Calculate and retain
        parameters that intrinsically depend on the fundamental scission
        model in the composite uFJC scission model.
        """
        CompositeuFJC.__init__(self, **kwargs)

        # Parameters needed for numerical calculations
        self.lmbda_nu_hat_inc = 0.0001
        self.num_quad_points = 1001
        
        p_nu_sci_hat_0 = 0.001
        p_nu_sci_hat_half = 0.5
        p_nu_sci_hat_1 = 0.999
        p_c_sci_hat_0 = 0.001
        p_c_sci_hat_half = 0.5
        p_c_sci_hat_1 = 0.999

        # Calculate and retain numerically calculated parameters
        (self.epsilon_nu_diss_hat_crit, self.epsilon_cnu_diss_hat_crit,
         self.u_nu_hat_p_nu_sur_hat_max, self.lmbda_nu_hat_u_nu_hat_p_nu_sur_hat_max,
         self.s_cnu_hat_p_nu_sur_hat_max, self.lmbda_nu_hat_s_cnu_hat_p_nu_sur_hat_max,
         self.psi_cnu_hat_p_nu_sur_hat_max, self.lmbda_nu_hat_psi_cnu_hat_p_nu_sur_hat_max,
         self.xi_c_hat_p_nu_sur_hat_max, self.lmbda_nu_hat_xi_c_hat_p_nu_sur_hat_max,
         self.u_nu_hat_p_c_sur_hat_max, self.lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max,
         self.s_cnu_hat_p_c_sur_hat_max, self.lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max,
         self.psi_cnu_hat_p_c_sur_hat_max, self.lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max,
         self.xi_c_hat_p_c_sur_hat_max, self.lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max,
         self.lmbda_nu_hat_p_nu_sci_hat_rms, self.lmbda_nu_hat_p_nu_sci_hat_mean,
         self.epsilon_nu_diss_hat_p_nu_sci_hat_rms, self.epsilon_nu_diss_hat_p_nu_sci_hat_mean,
         self.lmbda_nu_hat_p_c_sci_hat_rms, self.lmbda_nu_hat_p_c_sci_hat_mean,
         self.epsilon_cnu_diss_hat_p_c_sci_hat_rms, self.epsilon_cnu_diss_hat_p_c_sci_hat_mean
        ) = self.scission_parameters_func()

        self.A_nu = self.A_nu_func()
        self.Lambda_nu_ref = self.lmbda_nu_func(self.A_nu)

        self.g_c_crit = (
            0.5 * self.A_nu * self.nu * self.epsilon_cnu_diss_hat_crit
        )

        self.lmbda_nu_crit_p_nu_sci_hat_0 = (
            self.lmbda_nu_p_nu_sci_hat_analytical_func(p_nu_sci_hat_0)
        )
        self.lmbda_nu_crit_p_nu_sci_hat_half = (
            self.lmbda_nu_p_nu_sci_hat_analytical_func(p_nu_sci_hat_half)
        )
        self.lmbda_nu_crit_p_nu_sci_hat_1 = (
            self.lmbda_nu_p_nu_sci_hat_analytical_func(p_nu_sci_hat_1)
        )
        self.lmbda_nu_crit_p_c_sci_hat_0 = (
            self.lmbda_nu_p_c_sci_hat_analytical_func(p_c_sci_hat_0)
        )
        self.lmbda_nu_crit_p_c_sci_hat_half = (
            self.lmbda_nu_p_c_sci_hat_analytical_func(p_c_sci_hat_half)
        )
        self.lmbda_nu_crit_p_c_sci_hat_1 = (
            self.lmbda_nu_p_c_sci_hat_analytical_func(p_c_sci_hat_1)
        )

    def u_nu_tot_hat_func(self, lmbda_nu_hat, lmbda_nu):
        """Nondimensional total segment potential under an applied chain
        force.
        
        This function computes the nondimensional total segment
        potential under an applied chain force as a function of the
        applied segment stretch and the segment stretch specifying a
        particular state in the energy landscape.
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return (
            self.u_nu_func(lmbda_nu)
            - lmbda_nu * self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat)
        )
    
    def u_nu_hat_func(self, lmbda_nu_hat, lmbda_nu):
        """Nondimensional total distorted segment potential under an
        applied chain force.
        
        This function computes the nondimensional total distorted
        segment potential under an applied chain force as a function
        of the applied segment stretch and the segment stretch
        specifying a particular state in the energy landscape.
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return (
            self.u_nu_func(lmbda_nu)
            - (lmbda_nu-lmbda_nu_hat)
            * self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat)
        )
    
    def lmbda_nu_locmin_hat_func(self, lmbda_nu_hat):
        """Segment stretch corresponding to the local minimum of the
        nondimensional total (distorted) segment potential under an 
        applied chain force.
        
        This function computes the segment stretch corresponding to the
        local minimum of the nondimensional total (distorted) segment
        potential under an applied chain force as a function of the
        applied segment stretch.
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return (
            1. + self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat) / self.kappa_nu
        )
    
    def lmbda_nu_locmax_hat_func(self, lmbda_nu_hat):
        """Segment stretch corresponding to the local maximum of the
        nondimensional total (distorted) segment potential under an 
        applied chain force.
        
        This function computes the segment stretch corresponding to the
        local maximum of the nondimensional total (distorted) segment
        potential under an applied chain force as a function of the
        applied segment stretch.
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        if lmbda_nu_hat <= 1.:
            return np.inf
        else:
            cbrt_arg_nmrtr = self.zeta_nu_char**2
            cbrt_arg_dnmntr = (
                self.kappa_nu * self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat)
            )
            cbrt_arg = cbrt_arg_nmrtr / cbrt_arg_dnmntr
            return 1. + np.cbrt(cbrt_arg)
    
    def epsilon_nu_sci_hat_func(self, lmbda_nu_hat):
        """Nondimensional segment scission energy.
        
        This function computes the nondimensional segment scission
        energy as a function of the applied segment stretch.
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return (
            self.psi_cnu_func(lmbda_nu_hat, lmbda_c_eq_hat) + self.zeta_nu_char
        )
    
    def epsilon_cnu_sci_hat_func(self, lmbda_nu_hat):
        """Nondimensional chain scission energy per segment.
        
        This function computes the nondimensional chain scission energy
        per segment as a function of the applied segment stretch.
        """
        return self.epsilon_nu_sci_hat_func(lmbda_nu_hat)
    
    def e_nu_sci_hat_func(self, lmbda_nu_hat):
        """Nondimensional segment scission activation energy barrier.
        
        This function computes the nondimensional segment scission
        activation energy barrier as a function of the applied segment
        stretch.
        """
        lmbda_nu_locmin_hat = self.lmbda_nu_locmin_hat_func(lmbda_nu_hat)
        lmbda_nu_locmax_hat = self.lmbda_nu_locmax_hat_func(lmbda_nu_hat)
        
        if lmbda_nu_hat <= 1:
            return self.zeta_nu_char
        elif lmbda_nu_hat <= self.lmbda_nu_crit:
            return (
                self.u_nu_hat_func(lmbda_nu_hat, lmbda_nu_locmax_hat)
                - self.u_nu_hat_func(lmbda_nu_hat, lmbda_nu_locmin_hat)
            )
        else:
            return 0.
    
    def p_nu_sci_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of segment scission.
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch.
        """
        return np.exp(-self.e_nu_sci_hat_func(lmbda_nu_hat))

    def p_nu_sur_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of segment survival.
        
        This function computes the rate-independent probability of
        segment survival as a function of the applied segment stretch.
        """
        return 1. - self.p_nu_sci_hat_func(lmbda_nu_hat)
    
    def p_c_sur_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of chain survival.
        
        This function computes the rate-independent probability of chain
        survival as a function of the applied segment stretch.
        """
        return self.p_nu_sur_hat_func(lmbda_nu_hat)**self.nu
    
    def p_c_sci_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of chain scission.
        
        This function computes the rate-independent probability of chain
        scission as a function of the applied segment stretch.
        """
        return 1. - self.p_c_sur_hat_func(lmbda_nu_hat)
    
    def epsilon_nu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional segment scission
        energy function.
        
        This function computes the nondimensional segment scission
        energy as a function of the applied segment stretch.
        """
        return self.psi_cnu_analytical_func(lmbda_nu_hat) + self.zeta_nu_char
    
    def epsilon_cnu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional chain scission
        energy function per segment.
        
        This function computes the nondimensional chain scission
        energy per segment as a function of the applied segment stretch.
        """
        return self.epsilon_nu_sci_hat_analytical_func(lmbda_nu_hat)
    
    def e_nu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional segment scission
        activation energy barrier.
        
        This function computes the nondimensional segment scission
        activation energy barrier as a function of the applied segment
        stretch.
        """
        if lmbda_nu_hat <= 1. + self.cond_val:
            return self.zeta_nu_char
        elif lmbda_nu_hat <= self.lmbda_nu_crit:
            cbrt_arg = (
                self.zeta_nu_char**2 * self.kappa_nu * (lmbda_nu_hat-1.)**2
            )
            return (
                0.5 * self.kappa_nu * (lmbda_nu_hat-1.)**2
                - 1.5 * np.cbrt(cbrt_arg) + self.zeta_nu_char
            )
        else:
            return 0.
    
    def e_nu_sci_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the nondimensional
        segment scission activation energy barrier taken with respect to
        the applied segment stretch.
        
        This function computes the derivative of the nondimensional
        segment scission activation energy barrier taken with respect to
        the applied segment stretch as a function of applied segment
        stretch.
        """
        if lmbda_nu_hat <= 1. + self.cond_val:
            return -np.inf
        elif lmbda_nu_hat <= self.lmbda_nu_crit:
            cbrt_arg = self.zeta_nu_char**2 * self.kappa_nu / (lmbda_nu_hat-1.)
            return self.kappa_nu * (lmbda_nu_hat-1.) - np.cbrt(cbrt_arg)
        else:
            return 0.
    
    def p_nu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        segment scission.
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch.
        """
        return np.exp(-self.e_nu_sci_hat_analytical_func(lmbda_nu_hat))

    def p_nu_sur_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        segment survival.
        
        This function computes the rate-independent probability of
        segment survival as a function of the applied segment stretch.
        """
        return 1. - self.p_nu_sci_hat_analytical_func(lmbda_nu_hat)
    
    def p_c_sur_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        chain survival.
        
        This function computes the rate-independent probability of
        chain survival as a function of the applied segment stretch.
        """
        return self.p_nu_sur_hat_analytical_func(lmbda_nu_hat)**self.nu
    
    def p_c_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        chain scission.
        
        This function computes the rate-independent probability of
        chain scission as a function of the applied segment stretch.
        """
        return 1. - self.p_c_sur_hat_analytical_func(lmbda_nu_hat)
    
    def lmbda_nu_p_nu_sci_hat_analytical_func(self, p_nu_sci_hat_val):
        """Segment stretch as a function of the rate-independent
        probability of segment scission.
        
        This function calculates the segment stretch as a function of
        the rate-independent probability of segment scission. This
        calculation is based upon the analytical form of the
        nondimensional segment scission activation energy barrier.
        """
        pi_tilde = -3. * np.cbrt((self.zeta_nu_char/self.kappa_nu)**2)

        rho_tilde = (
            2 / self.kappa_nu * (self.zeta_nu_char+np.log(p_nu_sci_hat_val))
        )
        arccos_arg = 3. * rho_tilde / (2.*pi_tilde) * np.sqrt(-3./pi_tilde)
        cos_arg = 1. / 3. * np.arccos(arccos_arg) - 2. * np.pi / 3.

        phi_tilde = 2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)

        return 1. + np.sqrt(phi_tilde**3)
    
    def lmbda_nu_p_c_sci_hat_analytical_func(self, p_c_sci_hat_val):
        """Segment stretch as a function of the rate-independent
        probability of chain scission.
        
        This function calculates the segment stretch as a function of
        the rate-independent probability of chain scission. This
        calculation is based upon the analytical form of the
        nondimensional segment scission activation energy barrier.
        """
        p_nu_sci_hat_val = 1. - (1.-p_c_sci_hat_val)**(1./self.nu)

        return self.lmbda_nu_p_nu_sci_hat_analytical_func(p_nu_sci_hat_val)

    def p_nu_sci_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        if lmbda_nu_hat <= 1. + self.cond_val:
            return 0.
        else:
            return (
                -self.p_nu_sci_hat_analytical_func(lmbda_nu_hat)
                * self.e_nu_sci_hat_prime_analytical_func(lmbda_nu_hat)
            )
    
    def p_nu_sur_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        if lmbda_nu_hat <= 1. + self.cond_val:
            return 0.
        else:
            return -self.p_nu_sci_hat_prime_analytical_func(lmbda_nu_hat)
    
    def p_c_sur_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        if lmbda_nu_hat <= 1. + self.cond_val:
            return 0.
        else:
            return (
                -self.nu
                * (1.-self.p_nu_sci_hat_analytical_func(lmbda_nu_hat))**(self.nu-1)
                * self.p_nu_sci_hat_prime_analytical_func(lmbda_nu_hat)
            )

    def p_c_sci_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        if lmbda_nu_hat <= 1. + self.cond_val:
            return 0.
        else:
            return -self.p_c_sur_hat_prime_analytical_func(lmbda_nu_hat)

    def epsilon_nu_diss_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the nondimensional
        rate-independent dissipated segment scission energy taken with 
        respect to the applied segment stretch.
        
        This function computes the derivative of the nondimensional
        rate-independent dissipated segment scission energy taken with
        respect to the applied segment stretch as a function of applied
        segment stretch.
        """
        return (
            self.p_nu_sci_hat_prime_analytical_func(lmbda_nu_hat)
            * self.epsilon_nu_sci_hat_analytical_func(lmbda_nu_hat)
        )
    
    def epsilon_nu_diss_hat_analytical_func(
            self, lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
            lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
            epsilon_nu_diss_hat_val_prior):
        """Analytical form of the nondimensional rate-independent
        dissipated segment scission energy.
        
        This function computes the nondimensional rate-independent
        dissipated segment scission energy as a function of its prior
        value, the current and prior values of the applied segment
        stretch, and the current and prior values of the maximum applied
        segment stretch.
        """
        # dissipated energy cannot be destroyed
        if lmbda_nu_hat_val <= lmbda_nu_hat_val_prior:
            return epsilon_nu_diss_hat_val_prior
        elif lmbda_nu_hat_val < lmbda_nu_hat_max_val:
            return epsilon_nu_diss_hat_val_prior
        # dissipated energy from fully broken segments remains fixed
        elif lmbda_nu_hat_max_val_prior > self.lmbda_nu_crit:
            return epsilon_nu_diss_hat_val_prior
        else:
            # no dissipated energy at equilibrium
            if (lmbda_nu_hat_val-1.) <= self.lmbda_nu_hat_inc:
                epsilon_nu_diss_hat_prime_val = 0.
            else:
                # dissipated energy is created with respect to the prior
                # value of maximum applied segment stretch
                if lmbda_nu_hat_val_prior < lmbda_nu_hat_max_val_prior:
                    lmbda_nu_hat_val_prior = lmbda_nu_hat_max_val_prior
                # dissipated energy plateaus at the critical segment
                # stretch
                if (lmbda_nu_hat_max_val_prior < self.lmbda_nu_crit and 
                    lmbda_nu_hat_val > self.lmbda_nu_crit):
                    lmbda_nu_hat_val = self.lmbda_nu_crit
                
                epsilon_nu_diss_hat_prime_val = (
                    self.epsilon_nu_diss_hat_prime_analytical_func(
                        lmbda_nu_hat_val)
                )
            
            return (
                epsilon_nu_diss_hat_val_prior + epsilon_nu_diss_hat_prime_val
                * (lmbda_nu_hat_val-lmbda_nu_hat_val_prior)
            )
    
    def epsilon_nu_diss_hat_rate_independent_scission_func(
            self, lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
            lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
            epsilon_nu_diss_hat_val_prior):
        """Nondimensional rate-independent dissipated segment scission
        energy.
        
        This function computes the nondimensional rate-independent
        dissipated segment scission energy as a function of its prior
        value, the current and prior values of the applied segment
        stretch, and the current and prior values of the maximum applied
        segment stretch.
        """
        # dissipated energy cannot be destroyed
        if lmbda_nu_hat_val <= lmbda_nu_hat_val_prior:
            return epsilon_nu_diss_hat_val_prior
        elif lmbda_nu_hat_val < lmbda_nu_hat_max_val:
            return epsilon_nu_diss_hat_val_prior
        # dissipated energy from fully broken segments remains fixed
        elif lmbda_nu_hat_max_val_prior > self.lmbda_nu_crit:
            return epsilon_nu_diss_hat_val_prior
        else:
            # no dissipated energy at equilibrium
            if (lmbda_nu_hat_val-1.) <= self.lmbda_nu_hat_inc:
                epsilon_nu_diss_hat_prime_val = 0.
            else:
                # dissipated energy is created with respect to the prior
                # value of maximum applied segment stretch
                if lmbda_nu_hat_val_prior < lmbda_nu_hat_max_val_prior:
                    lmbda_nu_hat_val_prior = lmbda_nu_hat_max_val_prior
                # dissipated energy plateaus at the critical segment
                # stretch
                if (lmbda_nu_hat_max_val_prior < self.lmbda_nu_crit and 
                    lmbda_nu_hat_val > self.lmbda_nu_crit):
                    lmbda_nu_hat_val = self.lmbda_nu_crit
                
                p_nu_sci_hat_val_prior = (
                    self.p_nu_sci_hat_func(lmbda_nu_hat_val_prior)
                )
                p_nu_sci_hat_val = (
                    self.p_nu_sci_hat_func(lmbda_nu_hat_val)
                )
                p_nu_sci_hat_prime_val = (
                    (p_nu_sci_hat_val-p_nu_sci_hat_val_prior)
                    / (lmbda_nu_hat_val-lmbda_nu_hat_val_prior)
                )
                epsilon_nu_sci_hat_val = (
                    self.epsilon_nu_sci_hat_func(lmbda_nu_hat_val)
                )
                epsilon_nu_diss_hat_prime_val = (
                    p_nu_sci_hat_prime_val * epsilon_nu_sci_hat_val 
                )
            
            return (
                epsilon_nu_diss_hat_val_prior + epsilon_nu_diss_hat_prime_val
                * (lmbda_nu_hat_val-lmbda_nu_hat_val_prior)
            )
    
    def epsilon_cnu_diss_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the nondimensional
        rate-independent dissipated chain scission energy per segment
        taken with respect to the applied segment stretch.
        
        This function computes the derivative of the nondimensional
        rate-independent dissipated chain scission energy per segment
        taken with respect to the applied segment stretch as a function
        of applied segment stretch.
        """
        return (
            self.p_c_sci_hat_prime_analytical_func(lmbda_nu_hat)
            * self.epsilon_cnu_sci_hat_analytical_func(lmbda_nu_hat)
        )
    
    def epsilon_cnu_diss_hat_analytical_func(
            self, lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
            lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
            epsilon_cnu_diss_hat_val_prior):
        """Analytical form of the nondimensional rate-independent
        dissipated chain scission energy per segment.
        
        This function computes the nondimensional rate-independent
        dissipated chain scission energy per segment as a function of
        its prior value, the current and prior values of the applied
        segment stretch, and the current and prior values of the maximum
        applied segment stretch.
        """
        # dissipated energy cannot be destroyed
        if lmbda_nu_hat_val <= lmbda_nu_hat_val_prior:
            return epsilon_cnu_diss_hat_val_prior
        elif lmbda_nu_hat_val < lmbda_nu_hat_max_val:
            return epsilon_cnu_diss_hat_val_prior
        # dissipated energy from fully broken chains remains fixed
        elif lmbda_nu_hat_max_val_prior > self.lmbda_nu_crit:
            return epsilon_cnu_diss_hat_val_prior
        else:
            # no dissipated energy at equilibrium
            if (lmbda_nu_hat_val-1.) <= self.lmbda_nu_hat_inc:
                epsilon_cnu_diss_hat_prime_val = 0.
            else:
                # dissipated energy is created with respect to the prior
                # value of maximum applied segment stretch
                if lmbda_nu_hat_val_prior < lmbda_nu_hat_max_val_prior:
                    lmbda_nu_hat_val_prior = lmbda_nu_hat_max_val_prior
                # dissipated energy plateaus at the critical segment
                # stretch
                if (lmbda_nu_hat_max_val_prior < self.lmbda_nu_crit and 
                    lmbda_nu_hat_val > self.lmbda_nu_crit):
                    lmbda_nu_hat_val = self.lmbda_nu_crit
                
                epsilon_cnu_diss_hat_prime_val = (
                    self.epsilon_cnu_diss_hat_prime_analytical_func(
                        lmbda_nu_hat_val)
                )
            
            return (
                epsilon_cnu_diss_hat_val_prior + epsilon_cnu_diss_hat_prime_val
                * (lmbda_nu_hat_val-lmbda_nu_hat_val_prior)
            )
    
    def epsilon_cnu_diss_hat_rate_independent_scission_func(
            self, lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
            lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
            epsilon_cnu_diss_hat_val_prior):
        """Nondimensional rate-independent dissipated chain scission
        energy per segment.
        
        This function computes the nondimensional rate-independent
        dissipated chain scission energy per segment as a function of
        its prior value, the current and prior values of the applied
        segment stretch, and the current and prior values of the maximum
        applied segment stretch.
        """
        # dissipated energy cannot be destroyed
        if lmbda_nu_hat_val <= lmbda_nu_hat_val_prior:
            return epsilon_cnu_diss_hat_val_prior
        elif lmbda_nu_hat_val < lmbda_nu_hat_max_val:
            return epsilon_cnu_diss_hat_val_prior
        # dissipated energy from fully broken chains remains fixed
        elif lmbda_nu_hat_max_val_prior > self.lmbda_nu_crit:
            return epsilon_cnu_diss_hat_val_prior
        else:
            # no dissipated energy at equilibrium
            if (lmbda_nu_hat_val-1.) <= self.lmbda_nu_hat_inc:
                epsilon_cnu_diss_hat_prime_val = 0.
            else:
                # dissipated energy is created with respect to the prior
                # value of maximum applied segment stretch
                if lmbda_nu_hat_val_prior < lmbda_nu_hat_max_val_prior:
                    lmbda_nu_hat_val_prior = lmbda_nu_hat_max_val_prior
                # dissipated energy plateaus at the critical segment
                # stretch
                if (lmbda_nu_hat_max_val_prior < self.lmbda_nu_crit and 
                    lmbda_nu_hat_val > self.lmbda_nu_crit):
                    lmbda_nu_hat_val = self.lmbda_nu_crit
                
                p_c_sci_hat_val_prior = (
                    self.p_c_sci_hat_func(lmbda_nu_hat_val_prior)
                )
                p_c_sci_hat_val = (
                    self.p_c_sci_hat_func(lmbda_nu_hat_val)
                )
                p_c_sci_hat_prime_val = (
                    (p_c_sci_hat_val-p_c_sci_hat_val_prior)
                    / (lmbda_nu_hat_val-lmbda_nu_hat_val_prior)
                )
                epsilon_cnu_sci_hat_val = (
                    self.epsilon_cnu_sci_hat_func(lmbda_nu_hat_val)
                )
                epsilon_cnu_diss_hat_prime_val = (
                    p_c_sci_hat_prime_val * epsilon_cnu_sci_hat_val 
                )
            
            return (
                epsilon_cnu_diss_hat_val_prior + epsilon_cnu_diss_hat_prime_val
                * (lmbda_nu_hat_val-lmbda_nu_hat_val_prior)
            )
    
    def scission_parameters_func(self):
        """Segment-level and chain-level scission parameters.
        
        This function computes the following segment-level scission
        parameters:
        -> Nondimensional rate-independent dissipated segment scission
        energy for a segment at the critical state
        -> Maximum value of the product of the nondimensional segment
        potential energy with the probability of segment survival, and
        the corresponding segment stretch
        -> Maximum value of the product of the nondimensional
        segment-level entropic free energy with the probability of
        segment survival, and the corresponding segment stretch
        -> Maximum value of the product of the nondimensional
        segment-level Helmholtz free energy with the probability of
        segment survival, and the corresponding segment stretch
        -> Maximum value of the product of the nondimensional chain
        force with the probability of segment survival, and the
        corresponding segment stretch
        -> Root-mean-square segment stretch of the probability density
        function corresponding to the probability of segment scission,
        along with the corresponding value of the nondimensional
        rate-independent dissipated segment scission energy
        -> Mean segment stretch of the probability density function
        corresponding to the probability of segment scission, along with
        the corresponding value of the nondimensional rate-independent
        dissipated segment scission energy
        
        This function also computes the following chain-level scission
        parameters:
        -> Nondimensional per segment rate-independent dissipated chain
        scission energy for a chain at the critical state
        -> Maximum value of the product of the nondimensional segment
        potential energy with the probability of chain survival, and
        the corresponding segment stretch
        -> Maximum value of the product of the nondimensional
        segment-level entropic free energy with the probability of
        chain survival, and the corresponding segment stretch
        -> Maximum value of the product of the nondimensional per
        segment chain-level Helmholtz free energy with the probability
        of chain survival, and the corresponding segment stretch
        -> Maximum value of the product of the nondimensional chain
        force with the probability of chain survival, and the
        corresponding segment stretch
        -> Root-mean-square segment stretch of the probability density
        function corresponding to the probability of chain scission,
        along with the corresponding value of the nondimensional per
        segment rate-independent dissipated chain scission energy
        -> Mean segment stretch of the probability density function
        corresponding to the probability of chain scission, along with
        the corresponding value of the nondimensional per segment
        rate-independent dissipated chain scission energy
        """
        # Define the values of the applied segment stretch to 
        # calculate over
        lmbda_nu_hat_num_steps = (
            int(np.around(
                (self.lmbda_nu_crit-self.lmbda_nu_ref)/self.lmbda_nu_hat_inc))
            + 1
        )
        lmbda_nu_hat_steps = (
            np.linspace(
                self.lmbda_nu_ref, self.lmbda_nu_crit, lmbda_nu_hat_num_steps)
        )
        
        # Make arrays to allocate results
        lmbda_nu_hat = []
        lmbda_nu_hat_max = []
        p_nu_sci_hat = []
        epsilon_nu_diss_hat = []
        u_nu_hat_p_nu_sur_hat = []
        s_cnu_hat_p_nu_sur_hat = []
        psi_cnu_hat_p_nu_sur_hat = []
        xi_c_hat_p_nu_sur_hat = []
        p_c_sci_hat = []
        epsilon_cnu_diss_hat = []
        u_nu_hat_p_c_sur_hat = []
        s_cnu_hat_p_c_sur_hat = []
        psi_cnu_hat_p_c_sur_hat = []
        xi_c_hat_p_c_sur_hat = []
        
        # Initialization
        lmbda_nu_hat_max_val = 0.
        epsilon_nu_diss_hat_val = 0.
        u_nu_hat_p_nu_sur_hat_val = 0.
        s_cnu_hat_p_nu_sur_hat_val = 0.
        psi_cnu_hat_p_nu_sur_hat_val = 0.
        xi_c_hat_p_nu_sur_hat_val = 0.
        epsilon_cnu_diss_hat_val = 0.
        u_nu_hat_p_c_sur_hat_val = 0.
        s_cnu_hat_p_c_sur_hat_val = 0.
        psi_cnu_hat_p_c_sur_hat_val = 0.
        xi_c_hat_p_c_sur_hat_val = 0.
        
        # Calculate results through specified applied segment
        # stretch values
        for lmbda_nu_hat_indx in range(lmbda_nu_hat_num_steps):
            lmbda_nu_hat_val = lmbda_nu_hat_steps[lmbda_nu_hat_indx]
            lmbda_nu_hat_max_val = max([lmbda_nu_hat_max_val, lmbda_nu_hat_val])
            lmbda_c_eq_hat_val = self.lmbda_c_eq_func(lmbda_nu_hat_val)
            lmbda_comp_nu_hat_val = lmbda_c_eq_hat_val - lmbda_nu_hat_val + 1.
            u_nu_hat_val = self.u_nu_func(lmbda_nu_hat_val)
            s_cnu_hat_val = self.s_cnu_func(lmbda_comp_nu_hat_val)
            psi_cnu_hat_val = (
                self.psi_cnu_func(lmbda_nu_hat_val, lmbda_c_eq_hat_val)
            )
            xi_c_hat_val = (
                self.xi_c_func(lmbda_nu_hat_val, lmbda_c_eq_hat_val)
            )
            p_nu_sci_hat_val = self.p_nu_sci_hat_func(lmbda_nu_hat_val)
            p_nu_sur_hat_val = self.p_nu_sur_hat_func(lmbda_nu_hat_val)
            p_c_sur_hat_val = self.p_c_sur_hat_func(lmbda_nu_hat_val)
            p_c_sci_hat_val = self.p_c_sci_hat_func(lmbda_nu_hat_val)
            
            # Initialization
            if lmbda_nu_hat_indx == 0:
                u_nu_hat_init_val = u_nu_hat_val
                s_cnu_hat_init_val = s_cnu_hat_val
                psi_cnu_hat_init_val = psi_cnu_hat_val
                xi_c_hat_init_val = xi_c_hat_val
            else:
                epsilon_nu_diss_hat_val = (
                    self.epsilon_nu_diss_hat_rate_independent_scission_func(
                        lmbda_nu_hat_max_val,
                        lmbda_nu_hat_max[lmbda_nu_hat_indx-1],
                        lmbda_nu_hat_val,
                        lmbda_nu_hat[lmbda_nu_hat_indx-1],
                        epsilon_nu_diss_hat[lmbda_nu_hat_indx-1])
                )
                u_nu_hat_p_nu_sur_hat_val = (
                    p_nu_sur_hat_val * (u_nu_hat_val-u_nu_hat_init_val)
                )
                s_cnu_hat_p_nu_sur_hat_val = (
                    p_nu_sur_hat_val * (s_cnu_hat_val-s_cnu_hat_init_val)
                )
                psi_cnu_hat_p_nu_sur_hat_val = (
                    p_nu_sur_hat_val * (psi_cnu_hat_val-psi_cnu_hat_init_val)
                )
                xi_c_hat_p_nu_sur_hat_val = (
                    p_nu_sur_hat_val * (xi_c_hat_val-xi_c_hat_init_val)
                )
                epsilon_cnu_diss_hat_val = (
                    self.epsilon_cnu_diss_hat_rate_independent_scission_func(
                        lmbda_nu_hat_max_val,
                        lmbda_nu_hat_max[lmbda_nu_hat_indx-1],
                        lmbda_nu_hat_val,
                        lmbda_nu_hat[lmbda_nu_hat_indx-1],
                        epsilon_cnu_diss_hat[lmbda_nu_hat_indx-1])
                )
                u_nu_hat_p_c_sur_hat_val = (
                    p_c_sur_hat_val * (u_nu_hat_val-u_nu_hat_init_val)
                )
                s_cnu_hat_p_c_sur_hat_val = (
                    p_c_sur_hat_val * (s_cnu_hat_val-s_cnu_hat_init_val)
                )
                psi_cnu_hat_p_c_sur_hat_val = (
                    p_c_sur_hat_val * (psi_cnu_hat_val-psi_cnu_hat_init_val)
                )
                xi_c_hat_p_c_sur_hat_val = (
                    p_c_sur_hat_val * (xi_c_hat_val-xi_c_hat_init_val)
                )
            
            # Append values to lists
            lmbda_nu_hat.append(lmbda_nu_hat_val)
            lmbda_nu_hat_max.append(lmbda_nu_hat_max_val)
            p_nu_sci_hat.append(p_nu_sci_hat_val)
            epsilon_nu_diss_hat.append(epsilon_nu_diss_hat_val)
            u_nu_hat_p_nu_sur_hat.append(u_nu_hat_p_nu_sur_hat_val)
            s_cnu_hat_p_nu_sur_hat.append(s_cnu_hat_p_nu_sur_hat_val)
            psi_cnu_hat_p_nu_sur_hat.append(psi_cnu_hat_p_nu_sur_hat_val)
            xi_c_hat_p_nu_sur_hat.append(xi_c_hat_p_nu_sur_hat_val)
            p_c_sci_hat.append(p_c_sci_hat_val)
            epsilon_cnu_diss_hat.append(epsilon_cnu_diss_hat_val)
            u_nu_hat_p_c_sur_hat.append(u_nu_hat_p_c_sur_hat_val)
            s_cnu_hat_p_c_sur_hat.append(s_cnu_hat_p_c_sur_hat_val)
            psi_cnu_hat_p_c_sur_hat.append(psi_cnu_hat_p_c_sur_hat_val)
            xi_c_hat_p_c_sur_hat.append(xi_c_hat_p_c_sur_hat_val)
        
        # Critical dissipated scission energy
        epsilon_nu_diss_hat_crit = epsilon_nu_diss_hat[-1]
        epsilon_cnu_diss_hat_crit = epsilon_cnu_diss_hat[-1]

        # Maximum strength/survival values and associated segment
        # stretches
        indx_u_nu_hat_p_nu_sur_hat_max = (
            np.argmax(np.asarray(u_nu_hat_p_nu_sur_hat))
        )
        u_nu_hat_p_nu_sur_hat_max = (
            u_nu_hat_p_nu_sur_hat[indx_u_nu_hat_p_nu_sur_hat_max]
        )
        lmbda_nu_hat_u_nu_hat_p_nu_sur_hat_max = (
            lmbda_nu_hat[indx_u_nu_hat_p_nu_sur_hat_max]
        )
        indx_s_cnu_hat_p_nu_sur_hat_max = (
            np.argmax(np.asarray(s_cnu_hat_p_nu_sur_hat))
        )
        s_cnu_hat_p_nu_sur_hat_max = (
            s_cnu_hat_p_nu_sur_hat[indx_s_cnu_hat_p_nu_sur_hat_max]
        )
        lmbda_nu_hat_s_cnu_hat_p_nu_sur_hat_max = (
            lmbda_nu_hat[indx_s_cnu_hat_p_nu_sur_hat_max]
        )
        indx_psi_cnu_hat_p_nu_sur_hat_max = (
            np.argmax(np.asarray(psi_cnu_hat_p_nu_sur_hat))
        )
        psi_cnu_hat_p_nu_sur_hat_max = (
            psi_cnu_hat_p_nu_sur_hat[indx_psi_cnu_hat_p_nu_sur_hat_max]
        )
        lmbda_nu_hat_psi_cnu_hat_p_nu_sur_hat_max = (
            lmbda_nu_hat[indx_psi_cnu_hat_p_nu_sur_hat_max]
        )
        indx_xi_c_hat_p_nu_sur_hat_max = (
            np.argmax(np.asarray(xi_c_hat_p_nu_sur_hat))
        )
        xi_c_hat_p_nu_sur_hat_max = (
            xi_c_hat_p_nu_sur_hat[indx_xi_c_hat_p_nu_sur_hat_max]
        )
        lmbda_nu_hat_xi_c_hat_p_nu_sur_hat_max = (
            lmbda_nu_hat[indx_xi_c_hat_p_nu_sur_hat_max]
        )
        indx_u_nu_hat_p_c_sur_hat_max = (
            np.argmax(np.asarray(u_nu_hat_p_c_sur_hat))
        )
        u_nu_hat_p_c_sur_hat_max = (
            u_nu_hat_p_c_sur_hat[indx_u_nu_hat_p_c_sur_hat_max]
        )
        lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max = (
            lmbda_nu_hat[indx_u_nu_hat_p_c_sur_hat_max]
        )
        indx_s_cnu_hat_p_c_sur_hat_max = (
            np.argmax(np.asarray(s_cnu_hat_p_c_sur_hat))
        )
        s_cnu_hat_p_c_sur_hat_max = (
            s_cnu_hat_p_c_sur_hat[indx_s_cnu_hat_p_c_sur_hat_max]
        )
        lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max = (
            lmbda_nu_hat[indx_s_cnu_hat_p_c_sur_hat_max]
        )
        indx_psi_cnu_hat_p_c_sur_hat_max = (
            np.argmax(np.asarray(psi_cnu_hat_p_c_sur_hat))
        )
        psi_cnu_hat_p_c_sur_hat_max = (
            psi_cnu_hat_p_c_sur_hat[indx_psi_cnu_hat_p_c_sur_hat_max]
        )
        lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max = (
            lmbda_nu_hat[indx_psi_cnu_hat_p_c_sur_hat_max]
        )
        indx_xi_c_hat_p_c_sur_hat_max = (
            np.argmax(np.asarray(xi_c_hat_p_c_sur_hat))
        )
        xi_c_hat_p_c_sur_hat_max = (
            xi_c_hat_p_c_sur_hat[indx_xi_c_hat_p_c_sur_hat_max]
        )
        lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max = (
            lmbda_nu_hat[indx_xi_c_hat_p_c_sur_hat_max]
        )
        
        # Probability density distribution of segment scission
        # root-mean-square and mean calculations
        lmbda_nu_hat_arr = np.asarray(lmbda_nu_hat)
        p_nu_sci_hat_arr = np.asarray(p_nu_sci_hat)
        partial_p_nu_sci_hat__partial_lmbda_nu_hat_arr = (
            np.gradient(p_nu_sci_hat_arr, lmbda_nu_hat_arr, edge_order=2)
        )

        Z_nu_sci = (
            np.trapz(partial_p_nu_sci_hat__partial_lmbda_nu_hat_arr, lmbda_nu_hat_arr)
        )

        lmbda_nu_hat_p_nu_sci_hat_rms_intrgrnd_arr = (
            partial_p_nu_sci_hat__partial_lmbda_nu_hat_arr * lmbda_nu_hat_arr**2
        )
        lmbda_nu_hat_p_nu_sci_hat_mean_intrgrnd_arr = (
            partial_p_nu_sci_hat__partial_lmbda_nu_hat_arr * lmbda_nu_hat_arr
        )

        lmbda_nu_hat_p_nu_sci_hat_rms = (
            np.sqrt(np.trapz(lmbda_nu_hat_p_nu_sci_hat_rms_intrgrnd_arr, lmbda_nu_hat_arr)/Z_nu_sci)
        )
        lmbda_nu_hat_p_nu_sci_hat_mean = (
            np.trapz(lmbda_nu_hat_p_nu_sci_hat_mean_intrgrnd_arr, lmbda_nu_hat_arr)/Z_nu_sci
        )

        epsilon_nu_diss_hat_p_nu_sci_hat_rms = (
            np.interp(
                lmbda_nu_hat_p_nu_sci_hat_rms,
                lmbda_nu_hat_arr, epsilon_nu_diss_hat)
        )

        epsilon_nu_diss_hat_p_nu_sci_hat_mean = (
            np.interp(
                lmbda_nu_hat_p_nu_sci_hat_mean,
                lmbda_nu_hat_arr, epsilon_nu_diss_hat)
        )
        
        p_c_sci_hat_arr = np.asarray(p_c_sci_hat)
        partial_p_c_sci_hat__partial_lmbda_nu_hat_arr = (
                np.gradient(p_c_sci_hat_arr, lmbda_nu_hat_arr, edge_order=2)
            )
        
        Z_c_sci = (
            np.trapz(partial_p_c_sci_hat__partial_lmbda_nu_hat_arr, lmbda_nu_hat_arr)
        )

        lmbda_nu_hat_p_c_sci_hat_rms_intrgrnd_arr = (
            partial_p_c_sci_hat__partial_lmbda_nu_hat_arr * lmbda_nu_hat_arr**2
        )
        lmbda_nu_hat_p_c_sci_hat_mean_intrgrnd_arr = (
            partial_p_c_sci_hat__partial_lmbda_nu_hat_arr * lmbda_nu_hat_arr
        )

        lmbda_nu_hat_p_c_sci_hat_rms = (
            np.sqrt(np.trapz(lmbda_nu_hat_p_c_sci_hat_rms_intrgrnd_arr, lmbda_nu_hat_arr)/Z_c_sci)
        )
        lmbda_nu_hat_p_c_sci_hat_mean = (
            np.trapz(lmbda_nu_hat_p_c_sci_hat_mean_intrgrnd_arr, lmbda_nu_hat_arr)/Z_c_sci
        )

        epsilon_cnu_diss_hat_p_c_sci_hat_rms = (
            np.interp(
                lmbda_nu_hat_p_c_sci_hat_rms,
                lmbda_nu_hat_arr, epsilon_cnu_diss_hat)
        )

        epsilon_cnu_diss_hat_p_c_sci_hat_mean = (
            np.interp(
                lmbda_nu_hat_p_c_sci_hat_mean,
                lmbda_nu_hat_arr, epsilon_cnu_diss_hat)
        )
        
        del (
            lmbda_nu_hat_steps, lmbda_nu_hat, lmbda_nu_hat_max, p_nu_sci_hat,
            epsilon_nu_diss_hat, u_nu_hat_p_nu_sur_hat, s_cnu_hat_p_nu_sur_hat,
            psi_cnu_hat_p_nu_sur_hat, xi_c_hat_p_nu_sur_hat, p_c_sci_hat,
            epsilon_cnu_diss_hat, u_nu_hat_p_c_sur_hat, s_cnu_hat_p_c_sur_hat,
            psi_cnu_hat_p_c_sur_hat, xi_c_hat_p_c_sur_hat, lmbda_nu_hat_arr,
            p_nu_sci_hat_arr, partial_p_nu_sci_hat__partial_lmbda_nu_hat_arr,
            lmbda_nu_hat_p_nu_sci_hat_rms_intrgrnd_arr,
            lmbda_nu_hat_p_nu_sci_hat_mean_intrgrnd_arr,
            p_c_sci_hat_arr, partial_p_c_sci_hat__partial_lmbda_nu_hat_arr,
            lmbda_nu_hat_p_c_sci_hat_rms_intrgrnd_arr,
            lmbda_nu_hat_p_c_sci_hat_mean_intrgrnd_arr
        )
        
        return (
            epsilon_nu_diss_hat_crit, epsilon_cnu_diss_hat_crit,
            u_nu_hat_p_nu_sur_hat_max, lmbda_nu_hat_u_nu_hat_p_nu_sur_hat_max,
            s_cnu_hat_p_nu_sur_hat_max, lmbda_nu_hat_s_cnu_hat_p_nu_sur_hat_max,
            psi_cnu_hat_p_nu_sur_hat_max, lmbda_nu_hat_psi_cnu_hat_p_nu_sur_hat_max,
            xi_c_hat_p_nu_sur_hat_max, lmbda_nu_hat_xi_c_hat_p_nu_sur_hat_max,
            u_nu_hat_p_c_sur_hat_max, lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max,
            s_cnu_hat_p_c_sur_hat_max, lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max,
            psi_cnu_hat_p_c_sur_hat_max, lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max,
            xi_c_hat_p_c_sur_hat_max, lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max,
            lmbda_nu_hat_p_nu_sci_hat_rms, lmbda_nu_hat_p_nu_sci_hat_mean,
            epsilon_nu_diss_hat_p_nu_sci_hat_rms, epsilon_nu_diss_hat_p_nu_sci_hat_mean,
            lmbda_nu_hat_p_c_sci_hat_rms, lmbda_nu_hat_p_c_sci_hat_mean,
            epsilon_cnu_diss_hat_p_c_sci_hat_rms, epsilon_cnu_diss_hat_p_c_sci_hat_mean
        )
    
    def Z_intact_func(self, lmbda_c_eq):
        """Integrand involved in the intact equilibrium chain
        configuration partition function integration
        
        This function computes the integrand involved in the intact 
        equilibrium chain configuration partition function integration
        as a function of the equilibrium chain stretch, integer n, and
        segment number nu
        """
        lmbda_nu = self.lmbda_nu_func(lmbda_c_eq)
        psi_cnu = self.psi_cnu_func(lmbda_nu, lmbda_c_eq)
        
        return np.exp(-self.nu*(psi_cnu+self.zeta_nu_char))
    
    def A_nu_func(self):
        """Reference equilibrium chain stretch.
        
        This function computes the reference equilibrium chain stretch
        via numerical quadrature.
        """
        def J_func(lmbda_c_eq_ref, lmbda_c_eq_crit):
            """Jacobian for the master space-equilibrium chain
            configuration space transformation.
            
            This function computes the Jacobian for the master space
            equilibrium chain configuration space transformation.
            """
            return (lmbda_c_eq_crit-lmbda_c_eq_ref)/2.
        def lmbda_c_eq_point_func(point):
            """Equilibrium chain stretch as a function of master space
            coordinate point.

            This function computes the equilibrium chain stretch as a
            function of master space coordinate point.
            """
            J = J_func(self.lmbda_c_eq_ref, self.lmbda_c_eq_crit)
            return J*(1+point) + self.lmbda_c_eq_ref
        
        # Numerical quadrature scheme for integration in the master
        # space, which corresponds to the initial intact equilibrium
        # chain configuration
        points, weights = np.polynomial.legendre.leggauss(self.num_quad_points)
        
        # sort points in ascending order
        indx_ascd_order = np.argsort(points)
        points = points[indx_ascd_order]
        weights = weights[indx_ascd_order]
        
        # Jacobian for the master space-equilibrium chain configuration
        # space transformation
        J = J_func(self.lmbda_c_eq_ref, self.lmbda_c_eq_crit)
        
        # Equilibrium chain stretches corresponding to the master space
        # points for the initial intact chain configuration
        lmbda_c_eq_0_points = lmbda_c_eq_point_func(points)
        
        # Integrand of the zeroth moment of the initial intact chain
        # configuration equilibrium probability density distribution without
        # without normalization
        I_0_intrgrnd = np.asarray(
            [self.Z_intact_func(lmbda_c_eq_0_point) * lmbda_c_eq_0_point**2
            for lmbda_c_eq_0_point in lmbda_c_eq_0_points]
        )
        
        # Zeroth moment of the initial intact chain configuration
        # equilibrium probability density distribution without
        # normalization
        I_0 = np.sum(np.multiply(weights, I_0_intrgrnd))*J
        
        # Total configuration equilibrium partition function
        Z_eq_tot = (1.+self.nu*np.exp(-self.epsilon_nu_diss_hat_crit)) * I_0
        
        # Integrand of the second moment of the initial intact chain
        # configuration equilibrium probability density distribution without
        # without normalization
        I_2_intrgrnd = np.asarray(
            [self.Z_intact_func(lmbda_c_eq_0_point) * lmbda_c_eq_0_point**4
            for lmbda_c_eq_0_point in lmbda_c_eq_0_points]
        )
        
        # Second moment of the initial intact chain configuration
        # equilibrium probability density distribution without
        # normalization
        I_2 = np.sum(np.multiply(weights, I_2_intrgrnd))*J
        
        del points, weights, lmbda_c_eq_0_points, I_0_intrgrnd, I_2_intrgrnd
        
        # Reference equilibrium chain stretch
        return np.sqrt(I_2/Z_eq_tot)