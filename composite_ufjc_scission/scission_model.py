"""The module for the composite uFJC scission model specifying the
fundamental analytical scission model.
"""

# Import external modules
from __future__ import division
import sys
import numpy as np
import numquad as nq

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
        self.lmbda_nu_hat_inc  = 0.0005
        self.num_quad_points   = 1001
        
        p_nu_sci_hat_0    = 0.001
        p_nu_sci_hat_half = 0.5
        p_nu_sci_hat_1    = 0.999
        p_c_sci_hat_0     = 0.001
        p_c_sci_hat_half  = 0.5
        p_c_sci_hat_1     = 0.999

        # Calculate and retain numerically calculated parameters
        self.epsilon_nu_diss_hat_crit  = self.epsilon_nu_diss_hat_crit_func()
        self.epsilon_cnu_diss_hat_crit = self.epsilon_cnu_diss_hat_crit_func()
        self.A_nu          = self.A_nu_func()
        self.Lambda_nu_ref = self.lmbda_nu_func(self.A_nu)

        self.g_c_crit = (
            0.5 * self.A_nu * self.nu**2 * self.epsilon_cnu_diss_hat_crit
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

        phi_tilde =  2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)

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
    
    def epsilon_nu_diss_hat_crit_func(self):
        """Nondimensional rate-independent dissipated segment scission
        energy for a chain at the critical state.
        
        This function computes the nondimensional rate-independent
        dissipated segment scission energy for a chain at the critical
        state.
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

        # initialization
        lmbda_nu_hat_max_val_prior = 0.
        lmbda_nu_hat_max_val       = 0.
        epsilon_nu_diss_hat_crit_val_prior = 0.
        epsilon_nu_diss_hat_crit_val       = 0.

        for lmbda_nu_hat_indx in range(lmbda_nu_hat_num_steps):
            lmbda_nu_hat_val = lmbda_nu_hat_steps[lmbda_nu_hat_indx]
            lmbda_nu_hat_max_val = max([lmbda_nu_hat_max_val, lmbda_nu_hat_val])
            
            if lmbda_nu_hat_indx == 0:
                pass
            else:
                epsilon_nu_diss_hat_crit_val = (
                    self.epsilon_nu_diss_hat_rate_independent_scission_func(
                        lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
                        lmbda_nu_hat_val,
                        lmbda_nu_hat_steps[lmbda_nu_hat_indx-1],
                        epsilon_nu_diss_hat_crit_val_prior)
                )
            
            # update values
            lmbda_nu_hat_max_val_prior = lmbda_nu_hat_max_val
            epsilon_nu_diss_hat_crit_val_prior = epsilon_nu_diss_hat_crit_val
        
        return epsilon_nu_diss_hat_crit_val
    
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
    
    def epsilon_cnu_diss_hat_crit_func(self):
        """Nondimensional rate-independent dissipated chain scission
        energy per segment for a chain at the critical state.
        
        This function computes the nondimensional rate-independent
        dissipated chain scission energy per segment for a chain at the
        critical state.
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

        # initialization
        lmbda_nu_hat_max_val_prior = 0.
        lmbda_nu_hat_max_val       = 0.
        epsilon_cnu_diss_hat_crit_val_prior = 0.
        epsilon_cnu_diss_hat_crit_val       = 0.

        for lmbda_nu_hat_indx in range(lmbda_nu_hat_num_steps):
            lmbda_nu_hat_val = lmbda_nu_hat_steps[lmbda_nu_hat_indx]
            lmbda_nu_hat_max_val = max([lmbda_nu_hat_max_val, lmbda_nu_hat_val])
            
            if lmbda_nu_hat_indx == 0:
                pass
            else:
                epsilon_cnu_diss_hat_crit_val = (
                    self.epsilon_cnu_diss_hat_rate_independent_scission_func(
                        lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
                        lmbda_nu_hat_val,
                        lmbda_nu_hat_steps[lmbda_nu_hat_indx-1],
                        epsilon_cnu_diss_hat_crit_val_prior)
                )
            
            # update values
            lmbda_nu_hat_max_val_prior = lmbda_nu_hat_max_val
            epsilon_cnu_diss_hat_crit_val_prior = epsilon_cnu_diss_hat_crit_val
        
        return epsilon_cnu_diss_hat_crit_val
    
    def Z_intact_func(self, lmbda_c_eq):
        """Integrand involved in the intact equilibrium chain
        configuration partition function integration
        
        This function computes the integrand involved in the intact 
        equilibrium chain configuration partition function integration
        as a function of the equilibrium chain stretch, integer n, and
        segment number nu
        """
        lmbda_nu = self.lmbda_nu_func(lmbda_c_eq)
        psi_cnu  = self.psi_cnu_func(lmbda_nu, lmbda_c_eq)
        
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
        scheme = nq.c1.gauss_legendre(self.num_quad_points)
        
        # sort points in ascending order
        indx_ascd_order = np.argsort(scheme.points)
        points = scheme.points[indx_ascd_order]
        weights = scheme.weights[indx_ascd_order]
        
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
        
        # Reference equilibrium chain stretch
        return np.sqrt(I_2/Z_eq_tot)


class SmoothstepScissionCompositeuFJC(CompositeuFJC):
    """The composite uFJC scission model class specifying the smoothstep
    scission model.

    This class contains methods specifying the smoothstep scission
    model, which involve defining both energetic and probabilistic
    quantities. It inherits all attributes and methods from the 
    ``CompositeuFJC`` class.
    """
    def __init__(self, **kwargs):
        """
        Initializes the ``SmoothstepScissionModelCompositeuFJC`` class.
        
        Initialize and inherit all attributes and methods from the
        ``CompositeuFJC`` class instance. Calculate and retain
        parameters that intrinsically depend on the fundamental scission
        model in the composite uFJC scission model.
        """
        CompositeuFJC.__init__(self, **kwargs)

        lmbda_nu_crit_min = kwargs.get("lmbda_nu_crit_min", None)
        lmbda_nu_crit_max = kwargs.get("lmbda_nu_crit_max", None)

        if lmbda_nu_crit_min is None:
            error_message = """\
                Error: Need to specify the minimum critical segment stretch \
                where segment scission initiates. \
                """
            sys.exit(error_message)
        if lmbda_nu_crit_max is None:
            error_message = """\
                Error: Need to specify the maximum critical segment stretch \
                where segments have undergone complete scission. \
                """
            sys.exit(error_message)
        if lmbda_nu_crit_min < self.lmbda_nu_ref:
            error_message = """\
                Error: The minimum critical segment stretch is less than the \
                minimum reference segment stretch of 1. The minimum critical \
                segment stretch needs to be greater than the minimum reference \
                segment stretch of 1. \
                """
            sys.exit(error_message)
        if lmbda_nu_crit_max > self.lmbda_nu_crit:
            error_message = """\
                Error: The maximum critical segment stretch is greater than \
                the critical segment stretch. The maximum critical segment \
                stretch needs to be less than the critical segment stretch. \
                """
            sys.exit(error_message)
        
        # Retain specified parameters
        self.lmbda_nu_crit_min = lmbda_nu_crit_min
        self.lmbda_nu_crit_max = lmbda_nu_crit_max

        # Parameters needed for numerical calculations
        self.lmbda_nu_hat_inc  = 0.0005
        self.num_quad_points   = 1001

        # Calculate and retain numerically calculated parameters
        self.epsilon_nu_diss_hat_crit  = self.epsilon_nu_diss_hat_crit_func()
        self.epsilon_cnu_diss_hat_crit = self.epsilon_cnu_diss_hat_crit_func()
        self.A_nu          = self.A_nu_func()
        self.Lambda_nu_ref = self.lmbda_nu_func(self.A_nu)

        self.g_c_crit = (
            0.5 * self.A_nu * self.nu**2 * self.epsilon_cnu_diss_hat_crit
        )

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
    
    def p_nu_sci_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of segment scission.
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch.
        """
        if lmbda_nu_hat < self.lmbda_nu_crit_min:
            return 0.
        elif lmbda_nu_hat > self.lmbda_nu_crit_max:
            return 1.
        else:
            x = (
                (lmbda_nu_hat-self.lmbda_nu_crit_min)
                / (self.lmbda_nu_crit_max-self.lmbda_nu_crit_min)
            )
            return (
                70 * x**9 - 315 * x**8 + 540 * x**7 - 420 * x**6 + 126 * x**5
            )

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
    
    def p_nu_sci_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of segment
        scission taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        if (lmbda_nu_hat < self.lmbda_nu_crit_min or 
            lmbda_nu_hat > self.lmbda_nu_crit_max):
            return 0.
        else:
            x = (
                (lmbda_nu_hat-self.lmbda_nu_crit_min)
                / (self.lmbda_nu_crit_max-self.lmbda_nu_crit_min)
            )
            return (
                (1./(self.lmbda_nu_crit_max-self.lmbda_nu_crit_min))
                * (630*x**8-2520*x**7+3780*x**6-2520*x**5+630*x**4)
            )
    
    def p_nu_sur_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of segment
        survival taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        return -self.p_nu_sci_hat_prime_func(lmbda_nu_hat)
    
    def p_c_sur_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of chain
        survival taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return (
                -self.nu
                * (1.-self.p_nu_sci_hat_func(lmbda_nu_hat))**(self.nu-1)
                * self.p_nu_sci_hat_prime_func(lmbda_nu_hat)
            )

    def p_c_sci_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of chain
        scission taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return -self.p_c_sur_hat_prime_func(lmbda_nu_hat)
    
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
    
    def p_nu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        segment scission.
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch.
        """
        return self.p_nu_sci_hat_func(lmbda_nu_hat)

    def p_nu_sur_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        segment survival.
        
        This function computes the rate-independent probability of
        segment survival as a function of the applied segment stretch.
        """
        return self.p_nu_sur_hat_func(lmbda_nu_hat)
    
    def p_c_sur_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        chain survival.
        
        This function computes the rate-independent probability of
        chain survival as a function of the applied segment stretch.
        """
        return self.p_c_sur_hat_func(lmbda_nu_hat)
    
    def p_c_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        chain scission.
        
        This function computes the rate-independent probability of
        chain scission as a function of the applied segment stretch.
        """
        return self.p_c_sci_hat_func( lmbda_nu_hat)
    
    def p_nu_sci_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        return self.p_nu_sci_hat_prime_func(lmbda_nu_hat)
    
    def p_nu_sur_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        return self.p_nu_sur_hat_prime_func(lmbda_nu_hat)
    
    def p_c_sur_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return self.p_c_sur_hat_prime_func(lmbda_nu_hat)

    def p_c_sci_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return self.p_c_sci_hat_prime_func(lmbda_nu_hat)

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
    
    def epsilon_nu_diss_hat_crit_func(self):
        """Nondimensional rate-independent dissipated segment scission
        energy for a chain at the critical state.
        
        This function computes the nondimensional rate-independent
        dissipated segment scission energy for a chain at the critical
        state.
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

        # initialization
        lmbda_nu_hat_max_val_prior = 0.
        lmbda_nu_hat_max_val       = 0.
        epsilon_nu_diss_hat_crit_val_prior = 0.
        epsilon_nu_diss_hat_crit_val       = 0.

        for lmbda_nu_hat_indx in range(lmbda_nu_hat_num_steps):
            lmbda_nu_hat_val = lmbda_nu_hat_steps[lmbda_nu_hat_indx]
            lmbda_nu_hat_max_val = max([lmbda_nu_hat_max_val, lmbda_nu_hat_val])
            
            if lmbda_nu_hat_indx == 0:
                pass
            else:
                epsilon_nu_diss_hat_crit_val = (
                    self.epsilon_nu_diss_hat_rate_independent_scission_func(
                        lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
                        lmbda_nu_hat_val,
                        lmbda_nu_hat_steps[lmbda_nu_hat_indx-1],
                        epsilon_nu_diss_hat_crit_val_prior)
                )
            
            # update values
            lmbda_nu_hat_max_val_prior = lmbda_nu_hat_max_val
            epsilon_nu_diss_hat_crit_val_prior = epsilon_nu_diss_hat_crit_val
        
        return epsilon_nu_diss_hat_crit_val
    
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
    
    def epsilon_cnu_diss_hat_crit_func(self):
        """Nondimensional rate-independent dissipated chain scission
        energy per segment for a chain at the critical state.
        
        This function computes the nondimensional rate-independent
        dissipated chain scission energy per segment for a chain at the
        critical state.
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

        # initialization
        lmbda_nu_hat_max_val_prior = 0.
        lmbda_nu_hat_max_val       = 0.
        epsilon_cnu_diss_hat_crit_val_prior = 0.
        epsilon_cnu_diss_hat_crit_val       = 0.

        for lmbda_nu_hat_indx in range(lmbda_nu_hat_num_steps):
            lmbda_nu_hat_val = lmbda_nu_hat_steps[lmbda_nu_hat_indx]
            lmbda_nu_hat_max_val = max([lmbda_nu_hat_max_val, lmbda_nu_hat_val])
            
            if lmbda_nu_hat_indx == 0:
                pass
            else:
                epsilon_cnu_diss_hat_crit_val = (
                    self.epsilon_cnu_diss_hat_rate_independent_scission_func(
                        lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
                        lmbda_nu_hat_val,
                        lmbda_nu_hat_steps[lmbda_nu_hat_indx-1],
                        epsilon_cnu_diss_hat_crit_val_prior)
                )
            
            # update values
            lmbda_nu_hat_max_val_prior = lmbda_nu_hat_max_val
            epsilon_cnu_diss_hat_crit_val_prior = epsilon_cnu_diss_hat_crit_val
        
        return epsilon_cnu_diss_hat_crit_val
    
    def Z_intact_func(self, lmbda_c_eq):
        """Integrand involved in the intact equilibrium chain
        configuration partition function integration
        
        This function computes the integrand involved in the intact 
        equilibrium chain configuration partition function integration
        as a function of the equilibrium chain stretch, integer n, and
        segment number nu
        """
        lmbda_nu = self.lmbda_nu_func(lmbda_c_eq)
        psi_cnu  = self.psi_cnu_func(lmbda_nu, lmbda_c_eq)
        
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
        scheme = nq.c1.gauss_legendre(self.num_quad_points)
        
        # sort points in ascending order
        indx_ascd_order = np.argsort(scheme.points)
        points = scheme.points[indx_ascd_order]
        weights = scheme.weights[indx_ascd_order]
        
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
        
        # Reference equilibrium chain stretch
        return np.sqrt(I_2/Z_eq_tot)


class SigmoidScissionCompositeuFJC(CompositeuFJC):
    """The composite uFJC scission model class specifying the sigmoid
    scission model.

    This class contains methods specifying the sigmoid scission
    model, which involve defining both energetic and probabilistic
    quantities. It inherits all attributes and methods from the
    ``CompositeuFJC`` class.
    """
    def __init__(self, **kwargs):
        """
        Initializes the ``SigmoidScissionModelCompositeuFJC`` class.
        
        Initialize and inherit all attributes and methods from the
        ``CompositeuFJC`` class instance. Calculate and retain
        parameters that intrinsically depend on the fundamental scission
        model in the composite uFJC scission model.
        """
        CompositeuFJC.__init__(self, **kwargs)

        tau = kwargs.get("tau", None)
        lmbda_nu_check = kwargs.get("lmbda_nu_check", None)

        if tau is None:
            error_message = """\
                Error: Need to specify the logistic growth rate of the segment \
                scission. \
                """
            sys.exit(error_message)
        if lmbda_nu_check is None:
            error_message = """\
                Error: Need to specify the segment stretch value at which \
                point the rate-independent probability of segment scission \
                equals 0.5. \
                """
            sys.exit(error_message)
        
        # Retain specified parameters
        self.tau = tau
        self.lmbda_nu_check = lmbda_nu_check

        # Parameters needed for numerical calculations
        self.lmbda_nu_hat_inc  = 0.0005
        self.num_quad_points   = 1001

        # Calculate and retain numerically calculated parameters
        self.epsilon_nu_diss_hat_crit  = self.epsilon_nu_diss_hat_crit_func()
        self.epsilon_cnu_diss_hat_crit = self.epsilon_cnu_diss_hat_crit_func()
        self.A_nu          = self.A_nu_func()
        self.Lambda_nu_ref = self.lmbda_nu_func(self.A_nu)

        self.g_c_crit = (
            0.5 * self.A_nu * self.nu**2 * self.epsilon_cnu_diss_hat_crit
        )

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
    
    def p_nu_sci_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of segment scission.
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch.
        """
        exp_arg = -self.tau * (lmbda_nu_hat-self.lmbda_nu_check)
        sqrt_arg = 1. - 1. / (1.+np.exp(exp_arg))
        return 1. - np.sqrt(sqrt_arg)

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
    
    def p_nu_sci_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of segment
        scission taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        exp_arg = -self.tau * (lmbda_nu_hat-self.lmbda_nu_check)
        sqrt_arg = 1. - 1. / (1.+np.exp(exp_arg))
        trm_i = self.tau / 2.
        trm_ii = 1. / np.sqrt(sqrt_arg)
        trm_iii = 1. / (1.+np.exp(exp_arg))**2
        trm_iv = np.exp(exp_arg)
        return trm_i * trm_ii * trm_iii * trm_iv
    
    def p_nu_sur_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of segment
        survival taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        return -self.p_nu_sci_hat_prime_func(lmbda_nu_hat)
    
    def p_c_sur_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of chain
        survival taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return (
                -self.nu
                * (1.-self.p_nu_sci_hat_func(lmbda_nu_hat))**(self.nu-1)
                * self.p_nu_sci_hat_prime_func(lmbda_nu_hat)
            )

    def p_c_sci_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of chain
        scission taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return -self.p_c_sur_hat_prime_func(lmbda_nu_hat)
    
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
    
    def p_nu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        segment scission.
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch.
        """
        return self.p_nu_sci_hat_func(lmbda_nu_hat)

    def p_nu_sur_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        segment survival.
        
        This function computes the rate-independent probability of
        segment survival as a function of the applied segment stretch.
        """
        return self.p_nu_sur_hat_func(lmbda_nu_hat)
    
    def p_c_sur_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        chain survival.
        
        This function computes the rate-independent probability of
        chain survival as a function of the applied segment stretch.
        """
        return self.p_c_sur_hat_func(lmbda_nu_hat)
    
    def p_c_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        chain scission.
        
        This function computes the rate-independent probability of
        chain scission as a function of the applied segment stretch.
        """
        return self.p_c_sci_hat_func( lmbda_nu_hat)
    
    def p_nu_sci_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        return self.p_nu_sci_hat_prime_func(lmbda_nu_hat)
    
    def p_nu_sur_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        return self.p_nu_sur_hat_prime_func(lmbda_nu_hat)
    
    def p_c_sur_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return self.p_c_sur_hat_prime_func(lmbda_nu_hat)

    def p_c_sci_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return self.p_c_sci_hat_prime_func(lmbda_nu_hat)

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
    
    def epsilon_nu_diss_hat_crit_func(self):
        """Nondimensional rate-independent dissipated segment scission
        energy for a chain at the critical state.
        
        This function computes the nondimensional rate-independent
        dissipated segment scission energy for a chain at the critical
        state.
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

        # initialization
        lmbda_nu_hat_max_val_prior = 0.
        lmbda_nu_hat_max_val       = 0.
        epsilon_nu_diss_hat_crit_val_prior = 0.
        epsilon_nu_diss_hat_crit_val       = 0.

        for lmbda_nu_hat_indx in range(lmbda_nu_hat_num_steps):
            lmbda_nu_hat_val = lmbda_nu_hat_steps[lmbda_nu_hat_indx]
            lmbda_nu_hat_max_val = max([lmbda_nu_hat_max_val, lmbda_nu_hat_val])
            
            if lmbda_nu_hat_indx == 0:
                pass
            else:
                epsilon_nu_diss_hat_crit_val = (
                    self.epsilon_nu_diss_hat_rate_independent_scission_func(
                        lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
                        lmbda_nu_hat_val,
                        lmbda_nu_hat_steps[lmbda_nu_hat_indx-1],
                        epsilon_nu_diss_hat_crit_val_prior)
                )
            
            # update values
            lmbda_nu_hat_max_val_prior = lmbda_nu_hat_max_val
            epsilon_nu_diss_hat_crit_val_prior = epsilon_nu_diss_hat_crit_val
        
        return epsilon_nu_diss_hat_crit_val
    
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
    
    def epsilon_cnu_diss_hat_crit_func(self):
        """Nondimensional rate-independent dissipated chain scission
        energy per segment for a chain at the critical state.
        
        This function computes the nondimensional rate-independent
        dissipated chain scission energy per segment for a chain at the
        critical state.
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

        # initialization
        lmbda_nu_hat_max_val_prior = 0.
        lmbda_nu_hat_max_val       = 0.
        epsilon_cnu_diss_hat_crit_val_prior = 0.
        epsilon_cnu_diss_hat_crit_val       = 0.

        for lmbda_nu_hat_indx in range(lmbda_nu_hat_num_steps):
            lmbda_nu_hat_val = lmbda_nu_hat_steps[lmbda_nu_hat_indx]
            lmbda_nu_hat_max_val = max([lmbda_nu_hat_max_val, lmbda_nu_hat_val])
            
            if lmbda_nu_hat_indx == 0:
                pass
            else:
                epsilon_cnu_diss_hat_crit_val = (
                    self.epsilon_cnu_diss_hat_rate_independent_scission_func(
                        lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
                        lmbda_nu_hat_val,
                        lmbda_nu_hat_steps[lmbda_nu_hat_indx-1],
                        epsilon_cnu_diss_hat_crit_val_prior)
                )
            
            # update values
            lmbda_nu_hat_max_val_prior = lmbda_nu_hat_max_val
            epsilon_cnu_diss_hat_crit_val_prior = epsilon_cnu_diss_hat_crit_val
        
        return epsilon_cnu_diss_hat_crit_val
    
    def Z_intact_func(self, lmbda_c_eq):
        """Integrand involved in the intact equilibrium chain
        configuration partition function integration
        
        This function computes the integrand involved in the intact 
        equilibrium chain configuration partition function integration
        as a function of the equilibrium chain stretch, integer n, and
        segment number nu
        """
        lmbda_nu = self.lmbda_nu_func(lmbda_c_eq)
        psi_cnu  = self.psi_cnu_func(lmbda_nu, lmbda_c_eq)
        
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
        scheme = nq.c1.gauss_legendre(self.num_quad_points)
        
        # sort points in ascending order
        indx_ascd_order = np.argsort(scheme.points)
        points = scheme.points[indx_ascd_order]
        weights = scheme.weights[indx_ascd_order]
        
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
        
        # Reference equilibrium chain stretch
        return np.sqrt(I_2/Z_eq_tot)