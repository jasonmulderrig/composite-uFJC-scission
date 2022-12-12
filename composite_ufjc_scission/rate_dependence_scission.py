"""The module for the composite uFJC scission model specifying the
rate-dependent or rate-independent nature of scission.
"""

# Import external modules
from __future__ import division
import numpy as np
import sys


class RateIndependentScission(object):
    """The composite uFJC scission model class specifying
    rate-independent scission.

    This class contains methods specifying rate-independent scission for
    the composite uFJC chain model, which involve defining both
    energetic and probabilistic quantities. Via class inheritance in the
    ``RateIndependentScissionCompositeuFJC`` class, this class inherits
    all attributes and methods from the
    ``AnalyticalScissionCompositeuFJC`` class, which inherits all
    attributes and methods from the ``CompositeuFJC`` class.
    """
    def __init__(self):
        pass

    def epsilon_nu_diss_hat_func(
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
        return (
            self.epsilon_nu_diss_hat_rate_independent_scission_func(
                lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
                lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
                epsilon_nu_diss_hat_val_prior)
        )
    
    def epsilon_cnu_diss_hat_func(
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
        return (
            self.epsilon_cnu_diss_hat_rate_independent_scission_func(
                lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
                lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
                epsilon_cnu_diss_hat_val_prior)
        )
    

class RateDependentScission(object):
    """The composite uFJC scission model class specifying rate-dependent
    scission.

    This class contains methods specifying rate-dependent scission for
    the composite uFJC chain model, which involve defining both
    energetic and probabilistic quantities. Via class inheritance in the
    ``RateDependentScissionCompositeuFJC`` class, this class inherits
    all attributes and methods from the
    ``AnalyticalScissionCompositeuFJC`` class, which inherits all
    attributes and methods from the ``CompositeuFJC`` class.
    """
    def __init__(self, **kwargs):
        omega_0 = kwargs.get("omega_0", None)

        if omega_0 is None:
            error_message = """\
                Error: Need to specify the microscopic frequency of segments \
                in the chains for rate-dependent deformation. \
                """
            sys.exit(error_message)
        
        # Retain specified parameters
        self.omega_0 = omega_0
    
    def p_nu_sci_hat_cmltv_intgrl_func(
            self, p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior, t_prior,
            p_nu_sci_hat_cmltv_intgrl_val_prior):
        """History-dependent time integral of the rate-independent
        probability of segment scission.
        
        This function computes the history-dependent time integral of
        the rate-independent probability of segment scission as a
        function of its prior value and the current and prior values of
        both the rate-independent probability of segment scission and
        time.
        """
        return (
            p_nu_sci_hat_cmltv_intgrl_val_prior + np.trapz(
                [p_nu_sci_hat_val_prior, p_nu_sci_hat_val], x=[t_prior, t_val])
        )
    
    def rho_nu_func(self, p_nu_sci_hat_cmltv_intgrl_val):
        """Rate-dependent probability of segment survival.
        
        This function computes the rate-dependent probability of segment
        survival as a function of the history-dependent time integral of
        the rate-independent probability of segment scission.
        """
        return np.exp(-self.omega_0*p_nu_sci_hat_cmltv_intgrl_val)
    
    def gamma_nu_func(self, p_nu_sci_hat_cmltv_intgrl_val):
        """Rate-dependent probability of segment scission.
        
        This function computes the rate-dependent probability of segment
        scission as a function of the history-dependent time integral of
        the rate-independent probability of segment scission.
        """
        return 1. - self.rho_nu_func(p_nu_sci_hat_cmltv_intgrl_val)
    
    def rho_nu_dot_func(self, p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val):
        """Time rate-of-change of the rate-dependent probability of
        segment survival.
        
        This function computes the time rate-of-change of the
        rate-dependent probability of segment survival as a function of
        the rate-independent probability of segment scission and the
        history-dependent time integral of the rate-independent
        probability of segment scission.
        """
        return (
            -self.omega_0 * p_nu_sci_hat_val
            * self.rho_nu_func(p_nu_sci_hat_cmltv_intgrl_val)
        )
    
    def gamma_nu_dot_func(
            self, p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val):
        """Time rate-of-change of the rate-dependent probability of
        segment scission.
        
        This function computes the time rate-of-change of the
        rate-dependent probability of segment scission as a function of
        the rate-independent probability of segment scission and the
        history-dependent time integral of the rate-independent
        probability of segment scission.
        """
        return (
            -self.rho_nu_dot_func(
                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val)
        )
    
    def epsilon_nu_diss_hat_func(
            self, p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
            epsilon_nu_sci_hat_val, t_val, t_prior,
            epsilon_nu_diss_hat_val_prior):
        """Nondimensional rate-dependent dissipated segment scission
        energy.
        
        This function computes the nondimensional rate-dependent
        dissipated segment scission energy as a function of its prior
        value, the current nondimensional segment scission energy, the
        current and prior values of time, the current rate-independent
        probability of segment scission, and the history-dependent time
        integral of the rate-independent probability of segment
        scission.
        """
        gamma_nu_dot_val = (
            self.gamma_nu_dot_func(
                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val)
        )
        epsilon_nu_diss_hat_dot_val = gamma_nu_dot_val * epsilon_nu_sci_hat_val
        
        return (
            epsilon_nu_diss_hat_val_prior + epsilon_nu_diss_hat_dot_val
            * (t_val-t_prior)
        )
    
    def rho_c_func(self, p_nu_sci_hat_cmltv_intgrl_val):
        """Rate-dependent probability of chain survival.
        
        This function computes the rate-dependent probability of chain
        survival as a function of the history-dependent time integral of
        the rate-independent probability of segment scission.
        """
        return np.exp(-self.nu*self.omega_0*p_nu_sci_hat_cmltv_intgrl_val)
    
    def gamma_c_func(self, p_nu_sci_hat_cmltv_intgrl_val):
        """Rate-dependent probability of chain scission.
        
        This function computes the rate-dependent probability of chain
        scission as a function of the history-dependent time integral of
        the rate-independent probability of segment scission.
        """
        return 1. - self.rho_c_func(p_nu_sci_hat_cmltv_intgrl_val)
    
    def rho_c_dot_func(self, p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val):
        """Time rate-of-change of the rate-dependent probability of
        chain survival.
        
        This function computes the time rate-of-change of the
        rate-dependent probability of chain survival as a function of
        the rate-independent probability of segment scission and the
        history-dependent time integral of the rate-independent
        probability of segment scission.
        """
        return (
            -self.nu * self.omega_0 * p_nu_sci_hat_val
            * self.rho_c_func(p_nu_sci_hat_cmltv_intgrl_val)
        )
    
    def gamma_c_dot_func(
            self, p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val):
        """Time rate-of-change of the rate-dependent probability of
        chain scission.
        
        This function computes the time rate-of-change of the
        rate-dependent probability of chain scission as a function of
        the rate-independent probability of segment scission and the
        history-dependent time integral of the rate-independent
        probability of segment scission.
        """
        return (
            -self.rho_c_dot_func(
                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val)
        )
    
    def epsilon_cnu_diss_hat_func(
            self, p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
            epsilon_cnu_sci_hat_val, t_val, t_prior,
            epsilon_cnu_diss_hat_val_prior):
        """Nondimensional rate-dependent dissipated chain scission
        energy per segment.
        
        This function computes the nondimensional rate-dependent
        dissipated chain scission energy per segment as a function of
        its prior value, the current nondimensional chain scission
        energy per segment, the current and prior values of time, the
        current rate-independent probability of segment scission, and
        the history-dependent time integral of the rate-independent
        probability of segment scission.
        """
        gamma_c_dot_val = (
            self.gamma_c_dot_func(
                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val)
        )
        epsilon_cnu_diss_hat_dot_val = gamma_c_dot_val * epsilon_cnu_sci_hat_val
        
        return (
            epsilon_cnu_diss_hat_val_prior + epsilon_cnu_diss_hat_dot_val
            * (t_val-t_prior)
        )