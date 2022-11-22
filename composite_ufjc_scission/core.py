"""The core single-chain module for the composite uFJC model"""

# Import external modules
from __future__ import division
import sys
import numpy as np


class CoreCompositeuFJC(object):
    """The composite uFJC single-chain model class.
    
    This class contains methods specifying the core functions and
    parameters underpinning the composite uFJC single-chain model
    independent of scission.
    """
    def __init__(self, **kwargs):
        """
        Initializes the ``CoreCompositeuFJC`` class, producing a
        composite uFJC single chain model instance.
        """
        # Define and store numerical tolerance parameters
        min_exponent = np.log(sys.float_info.min) / np.log(10)
        max_exponent = np.log(sys.float_info.max) / np.log(10)
        eps_val      = np.finfo(float).eps
        cond_val     = eps_val * 5e12

        self.min_exponent = min_exponent
        self.max_exponent = max_exponent
        self.eps_val      = eps_val
        self.cond_val     = cond_val

        # Get default parameter values
        rate_dependence = kwargs.get("rate_dependence", None)
        scission_model  = kwargs.get("scission_model", None)
        omega_0         = kwargs.get("omega_0", None)
        nu              = kwargs.get("nu", None)
        nu_b            = kwargs.get("nu_b", None)
        zeta_b_char     = kwargs.get("zeta_b_char", None)
        kappa_b         = kwargs.get("kappa_b", None)
        zeta_nu_char    = kwargs.get("zeta_nu_char", None)
        kappa_nu        = kwargs.get("kappa_nu", None)
        
        # Check the correctness of the specified parameters
        # Calculate segment-level parameters from provided bond-level
        # parameters if necessary
        if (rate_dependence != 'rate_dependent' and 
            rate_dependence != 'rate_independent'):
            error_message = """\
                Error: Need to specify the chain dependence on the rate of \
                applied deformation. Either rate-dependent or rate-independent \
                deformation can be used. \
                """
            sys.exit(error_message)
        if (scission_model != 'exact' and 
            scission_model != 'shifted_weibull_approximation'):
            error_message = """\
                Error: Need to specify the chain scission model. Either the \
                exact stochastic model or the shifted Weibull approximation of \
                the exact stochastic model can be used. \
                """
            sys.exit(error_message)
        if rate_dependence == 'rate_dependent' and omega_0 is None:
            error_message = """\
                Error: Need to specify the microscopic frequency of segments \
                in the chains for rate-dependent deformation. \
                """
            sys.exit(error_message)
        if nu is None:
            sys.exit('Error: Need to specify nu in the composite uFJC.')
        elif nu_b is None:
            if zeta_nu_char is None:
                error_message = """\
                    Error: Need to specify zeta_nu_char in the composite uFJC \
                    when nu_b is not specified.
                    """
                sys.exit(error_message)
            elif kappa_nu is None:
                error_message = """\
                    Error: Need to specify kappa_nu in the composite uFJC \
                    when nu_b is not specified. \
                    """
                sys.exit(error_message)
        elif nu_b is not None:
            if zeta_b_char is None:
                error_message = """\
                    Error: Need to specify zeta_b_char in the composite uFJC \
                    when nu_b is specified. \
                    """
                sys.exit(error_message)
            elif kappa_b is None:
                error_message = """\
                    Error: Need to specify kappa_b in the composite uFJC when \
                    nu_b is specified
                    """
                sys.exit(error_message)
            else:
                zeta_nu_char = nu_b * zeta_b_char
                kappa_nu     = nu_b * kappa_b
        
        # Retain specified parameters
        self.rate_dependence = rate_dependence
        self.scission_model  = scission_model
        self.omega_0         = omega_0
        self.nu              = nu
        self.nu_b            = nu_b
        self.zeta_b_char     = zeta_b_char
        self.kappa_b         = kappa_b
        self.zeta_nu_char    = zeta_nu_char
        self.kappa_nu        = kappa_nu
        
        # Calculate and retain analytically derived parameters
        self.lmbda_nu_ref    = 1.
        self.lmbda_c_eq_ref  = 0.
        self.lmbda_nu_crit   = 1. + np.sqrt(self.zeta_nu_char/self.kappa_nu)
        self.lmbda_c_eq_crit = (1. + np.sqrt(self.zeta_nu_char/self.kappa_nu)
                                - np.sqrt(1./(self.kappa_nu*self.zeta_nu_char)))
        self.xi_c_crit = np.sqrt(zeta_nu_char*kappa_nu)
        
        # Calculate and retain numerically calculated parameters
        self.lmbda_c_eq_pade2berg_crit = self.lmbda_c_eq_pade2berg_crit_func()
        self.lmbda_nu_pade2berg_crit   = self.lmbda_nu_func(
            self.lmbda_c_eq_pade2berg_crit)
    
    def u_nu_har_func(self, lmbda_nu):
        """Nondimensional harmonic segment potential energy
        
        This function computes the nondimensional harmonic segment 
        potential energy as a function of the segment stretch.
        """
        return 0.5 * self.kappa_nu * (lmbda_nu-1.)**2 - self.zeta_nu_char
    
    def u_nu_subcrit_func(self, lmbda_nu):
        """Nondimensional sub-critical chain state segment potential
        energy
        
        This function computes the nondimensional sub-critical chain
        state segment potential energy as a function of the segment
        stretch.
        """
        return self.u_nu_har_func(lmbda_nu)
    
    def u_nu_supercrit_func(self, lmbda_nu):
        """Nondimensional super-critical chain state segment potential
        energy
        
        This function computes the nondimensional super-critical chain
        state segment potential energy as a function of the segment
        stretch.
        """
        return -self.zeta_nu_char**2 / (2.*self.kappa_nu*(lmbda_nu-1.)**2)
    
    def u_nu_func(self, lmbda_nu):
        """Nondimensional composite uFJC segment potential energy
        
        This function computes the nondimensional composite uFJC 
        segment potential energy as a function of the segment stretch.
        """
        if lmbda_nu <= self.lmbda_nu_crit:
            return self.u_nu_subcrit_func(lmbda_nu)
        
        else:
            return self.u_nu_supercrit_func(lmbda_nu)
    
    def u_nu_har_comp_func(self, lmbda_nu):
        """Nondimensional harmonic segment potential energy contribution
        to the alternate nondimensional composite uFJC segment potential
        energy representation using Macaulay brackets
        
        This function computes the nondimensional harmonic segment
        potential energy contribution to the alternate nondimensional
        composite uFJC segment potential energy representation using 
        Macaulay brackets as a function of the segment stretch.
        """
        sgn = np.sign(lmbda_nu-1.)
        M_arg = sgn * self.kappa_nu * (lmbda_nu-1.)**2 - self.zeta_nu_char
        M_val = self.M_func(M_arg)
        M_val_frac = M_val / (M_val+self.zeta_nu_char)
        nonneg_u_nu_har_val = self.u_nu_har_func(lmbda_nu) + self.zeta_nu_char
        
        return (1.-M_val_frac)**2 * nonneg_u_nu_har_val - self.zeta_nu_char
    
    def u_nu_sci_comp_func(self, lmbda_nu):
        """Nondimensional segment scission potential energy contribution
        to the alternate nondimensional composite uFJC segment potential
        energy representation using Macaulay brackets
        
        This function computes the nondimensional segment scission
        potential energy contribution to the alternate nondimensional
        composite uFJC segment potential energy representation using
        Macaulay brackets as a function of the segment stretch.
        """
        sgn = np.sign(lmbda_nu-1.)
        M_arg = sgn * self.kappa_nu * (lmbda_nu-1.)**2 - self.zeta_nu_char
        M_val = self.M_func(M_arg)
        M_val_frac = M_val / (M_val+self.zeta_nu_char)

        return self.zeta_nu_char * (M_val_frac-1.)
    
    def u_nu_M_func(self, lmbda_nu):
        """Nondimensional composite uFJC segment potential energy
        representation using Macaulay brackets
        
        This function computes the nondimensional composite uFJC segment
        potential energy representation using Macaulay brackets as a
        function of the segment stretch.
        """
        return (self.u_nu_har_comp_func(lmbda_nu) 
                + self.u_nu_sci_comp_func(lmbda_nu))
    
    def M_func(self, x):
        """Macaulay brackets
        
        This function computes the value of the Macaulay brackets as a
        function of a number x.
        """
        if x < 0:
            return 0.
        else:
            return x
    
    def subcrit_lmbda_nu_berg_approx_func(self, lmbda_c_eq):
        """Sub-critical chain state segment stretch as derived via the
        Bergstrom approximant for the inverse Langevin function
        
        This function computes the sub-critical chain state segment
        stretch (as derived via the Bergstrom approximant for the
        inverse Langevin function) as a function of the equilibrium
        chain stretch and nondimensional segment stiffness.
        """
        sqrt_arg = lmbda_c_eq**2 - 2. * lmbda_c_eq + 1. + 4. / self.kappa_nu
        return (lmbda_c_eq+1.+np.sqrt(sqrt_arg)) / 2.
    
    def subcrit_lmbda_nu_pade_approx_func(self, lmbda_c_eq):
        """Sub-critical chain state segment stretch as derived via the
        Pade approximant for the inverse Langevin function
        
        This function computes the sub-critical chain state segment
        stretch (as derived via the Pade approximant for the inverse
        Langevin function) as a function of the equilibrium chain
        stretch and nondimensional segment stiffness.
        """
        # analytical solution
        if lmbda_c_eq == 0.:
            return 1.
        
        else:
            alpha_tilde = 1.
            
            trm_i  = -3. * (self.kappa_nu+1.)
            trm_ii = -(2.*self.kappa_nu+3.)
            beta_tilde_nmrtr = trm_i + lmbda_c_eq * trm_ii
            beta_tilde_dnmntr = self.kappa_nu + 1.
            beta_tilde  = beta_tilde_nmrtr / beta_tilde_dnmntr
            
            trm_i   = 2. * self.kappa_nu
            trm_ii  = 4. * self.kappa_nu + 6.
            trm_iii = self.kappa_nu + 3.
            gamma_tilde_nmrtr = trm_i + lmbda_c_eq * (trm_ii+lmbda_c_eq*trm_iii)
            gamma_tilde_dnmntr = self.kappa_nu + 1.
            gamma_tilde  = gamma_tilde_nmrtr / gamma_tilde_dnmntr

            trm_i   = 2.
            trm_ii  = 2. * self.kappa_nu
            trm_iii = self.kappa_nu + 3.
            delta_tilde_nmrtr = (trm_i - lmbda_c_eq
                                    * (trm_ii+lmbda_c_eq*(trm_iii+lmbda_c_eq)))
            delta_tilde_dnmntr = self.kappa_nu + 1.
            delta_tilde  = delta_tilde_nmrtr / delta_tilde_dnmntr

            pi_tilde_nmrtr  = 3. * alpha_tilde * gamma_tilde - beta_tilde**2
            pi_tilde_dnmntr = 3. * alpha_tilde**2
            pi_tilde = pi_tilde_nmrtr / pi_tilde_dnmntr

            rho_tilde_nmrtr = (2. * beta_tilde**3 
                                - 9. * alpha_tilde * beta_tilde * gamma_tilde 
                                + 27. * alpha_tilde**2 * delta_tilde)
            rho_tilde_dnmntr = 27. * alpha_tilde**3
            rho_tilde = rho_tilde_nmrtr / rho_tilde_dnmntr
            
            arccos_arg = 3. * rho_tilde / (2.*pi_tilde) * np.sqrt(-3./pi_tilde)
            cos_arg = 1. /3. * np.arccos(arccos_arg) - 2. * np.pi / 3.
            return (2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)
                    - beta_tilde / (3.*alpha_tilde))
    
    def lmbda_c_eq_pade2berg_crit_func(self):
        """Pade-to-Bergstrom (P2B) critical equilibrium chain stretch
        value
        
        This function returns Pade-to-Bergstrom (P2B) critical 
        equilibrium chain stretch value as determined via a scipy 
        optimize curve_fit analysis.
        """
        n = 0.818706900266885
        b = 0.61757545643322586
        return 1. / self.kappa_nu**n + b
    
    def pade2berg_crit_func(self):
        """Pade-to-Bergstrom (P2B) critical segment stretch and critical
        equilibrium chain stretch values
        
        This function numerically calculates the Pade-to-Bergstrom (P2B)
        critical segment stretch and critical equilibrium chain stretch
        values
        """
        lmbda_c_eq_min   = 0.
        lmbda_c_eq_max   = 1.
        lmbda_c_eq_steps = np.linspace(
            lmbda_c_eq_min, lmbda_c_eq_max, int(1e4)+1)
        
        # Make arrays to allocate results
        lmbda_c_eq         = []
        lmbda_nu_bergapprx = []
        lmbda_nu_padeapprx = []
        
        for lmbda_c_eq_indx in range(len(lmbda_c_eq_steps)):
            lmbda_c_eq_val = lmbda_c_eq_steps[lmbda_c_eq_indx]
            lmbda_nu_bergapprx_val = self.subcrit_lmbda_nu_berg_approx_func(
                lmbda_c_eq_val)
            lmbda_nu_padeapprx_val = self.subcrit_lmbda_nu_pade_approx_func(
                lmbda_c_eq_val)
            
            lmbda_c_eq = np.append(lmbda_c_eq, lmbda_c_eq_val)
            lmbda_nu_bergapprx = np.append(
                lmbda_nu_bergapprx, lmbda_nu_bergapprx_val)
            lmbda_nu_padeapprx = np.append(
                lmbda_nu_padeapprx, lmbda_nu_padeapprx_val)
        
        pade2berg_crit_indx = np.argmin(
            np.abs(lmbda_nu_padeapprx-lmbda_nu_bergapprx))
        lmbda_nu_pade2berg_crit = min(
            [lmbda_nu_bergapprx[pade2berg_crit_indx],
            lmbda_nu_padeapprx[pade2berg_crit_indx]])
        lmbda_c_eq_pade2berg_crit = lmbda_c_eq[pade2berg_crit_indx]
        return lmbda_nu_pade2berg_crit, lmbda_c_eq_pade2berg_crit
    
    def lmbda_c_eq_func(self, lmbda_nu):
        """Equilibrium chain stretch
        
        This function computes the equilibrium chain stretch as a 
        function of the segment stretch.
        """
        # analytical solution (Pade approximant)
        if lmbda_nu == 1.:
            return 0.
        
        # Pade approximant
        elif lmbda_nu < self.lmbda_nu_pade2berg_crit:
            alpha_tilde = 1.
            
            trm_i  = self.kappa_nu + 3.
            trm_ii = 1.
            beta_tilde = trm_i * (trm_ii-lmbda_nu)

            trm_i   = 2. * self.kappa_nu + 3.
            trm_ii  = 2.
            trm_iii = 2. * self.kappa_nu
            gamma_tilde = trm_i * (lmbda_nu**2-trm_ii*lmbda_nu) + trm_iii
            
            trm_i   = self.kappa_nu + 1.
            trm_ii  = 3.
            trm_iii = 2.
            trm_iv  = self.kappa_nu
            trm_v   = 1.
            delta_tilde = (trm_i * (trm_ii*lmbda_nu**2-lmbda_nu**3)
                            - trm_iii * (trm_iv*lmbda_nu+trm_v))
            
            pi_tilde_nmrtr  = 3. * alpha_tilde * gamma_tilde - beta_tilde**2
            pi_tilde_dnmntr = 3. * alpha_tilde**2
            pi_tilde = pi_tilde_nmrtr / pi_tilde_dnmntr

            rho_tilde_nmrtr = (2. * beta_tilde**3 
                                - 9. * alpha_tilde * beta_tilde * gamma_tilde 
                                + 27. * alpha_tilde**2 * delta_tilde)
            rho_tilde_dnmntr = 27. * alpha_tilde**3
            rho_tilde = rho_tilde_nmrtr / rho_tilde_dnmntr

            arccos_arg = 3. * rho_tilde / (2.*pi_tilde) * np.sqrt(-3./pi_tilde)
            cos_arg = 1. / 3. * np.arccos(arccos_arg) - 2. * np.pi / 3.
            return (2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)
                    - beta_tilde / (3.*alpha_tilde))
        
        # Bergstrom approximant
        elif lmbda_nu <= self.lmbda_nu_crit:
            return lmbda_nu - 1. / (self.kappa_nu*(lmbda_nu-1.))
        
        # Bergstrom approximant
        else:
            return (lmbda_nu 
                    - self.kappa_nu / self.zeta_nu_char**2 * (lmbda_nu-1.)**3)
    
    def lmbda_nu_func(self, lmbda_c_eq):
        """Segment stretch
        
        This function computes the segment stretch as a function of 
        the equilibrium chain stretch.
        """
        # analytical solution (Pade approximant)
        if lmbda_c_eq == 0.:
            return 1.
        
        # Pade approximant
        elif lmbda_c_eq < self.lmbda_c_eq_pade2berg_crit:
            alpha_tilde = 1.
            
            trm_i  = -3. * (self.kappa_nu+1.)
            trm_ii = -(2.*self.kappa_nu+3.)
            beta_tilde_nmrtr  = trm_i + lmbda_c_eq * trm_ii
            beta_tilde_dnmntr = self.kappa_nu + 1.
            beta_tilde  = beta_tilde_nmrtr / beta_tilde_dnmntr
            
            trm_i   = 2. * self.kappa_nu
            trm_ii  = 4. * self.kappa_nu + 6.
            trm_iii = self.kappa_nu + 3.
            gamma_tilde_nmrtr = trm_i + lmbda_c_eq * (trm_ii+lmbda_c_eq*trm_iii)
            gamma_tilde_dnmntr = self.kappa_nu + 1.
            gamma_tilde  = gamma_tilde_nmrtr / gamma_tilde_dnmntr

            trm_i   = 2.
            trm_ii  = 2. * self.kappa_nu
            trm_iii = self.kappa_nu + 3.
            delta_tilde_nmrtr = (trm_i - lmbda_c_eq 
                                    * (trm_ii+lmbda_c_eq*(trm_iii+lmbda_c_eq)))
            delta_tilde_dnmntr = self.kappa_nu + 1.
            delta_tilde  = delta_tilde_nmrtr / delta_tilde_dnmntr

            pi_tilde_nmrtr  = 3. * alpha_tilde * gamma_tilde - beta_tilde**2
            pi_tilde_dnmntr = 3. * alpha_tilde**2
            pi_tilde = pi_tilde_nmrtr / pi_tilde_dnmntr

            rho_tilde_nmrtr = (2. * beta_tilde**3 
                                - 9. * alpha_tilde * beta_tilde * gamma_tilde 
                                + 27. * alpha_tilde**2 * delta_tilde)
            rho_tilde_dnmntr = 27. * alpha_tilde**3
            rho_tilde = rho_tilde_nmrtr / rho_tilde_dnmntr
            
            arccos_arg = 3. * rho_tilde / (2.*pi_tilde) * np.sqrt(-3./pi_tilde)
            cos_arg = 1. / 3. * np.arccos(arccos_arg) - 2. * np.pi / 3.
            return (2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)
                    - beta_tilde / (3.*alpha_tilde))
        
        # Bergstrom approximant
        elif lmbda_c_eq <= self.lmbda_c_eq_crit:
            sqrt_arg = lmbda_c_eq**2 - 2. * lmbda_c_eq + 1. + 4. / self.kappa_nu
            return (lmbda_c_eq+1.+np.sqrt(sqrt_arg)) / 2.
        
        # Bergstrom approximant
        else:
            alpha_tilde = 1.
            beta_tilde  = -3.
            gamma_tilde = 3. - self.zeta_nu_char**2 / self.kappa_nu
            delta_tilde = self.zeta_nu_char**2 / self.kappa_nu * lmbda_c_eq - 1.

            pi_tilde_nmrtr = 3. * alpha_tilde * gamma_tilde - beta_tilde**2
            pi_tilde_dnmntr = 3. * alpha_tilde**2
            pi_tilde  = pi_tilde_nmrtr / pi_tilde_dnmntr

            rho_tilde_nmrtr = (2. * beta_tilde**3
                                - 9. * alpha_tilde * beta_tilde * gamma_tilde
                                + 27. * alpha_tilde**2 * delta_tilde)
            rho_tilde_dnmntr = 27. * alpha_tilde**3
            rho_tilde = rho_tilde_nmrtr / rho_tilde_dnmntr
            
            arccos_arg = 3. * rho_tilde / (2.*pi_tilde) * np.sqrt(-3./pi_tilde)
            cos_arg = 1. / 3. * np.arccos(arccos_arg) - 2. * np.pi / 3.
            return (2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg) 
                    - beta_tilde / (3.*alpha_tilde))

    def inv_L_func(self, lmbda_comp_nu):
        """Jedynak R[9,2] inverse Langevin approximant
        
        This function computes the Jedynak R[9,2] inverse Langevin
        approximant as a function of the result of the equilibrium
        chain stretch minus the segment stretch plus one
        """
        return (lmbda_comp_nu * (3.-1.00651*lmbda_comp_nu**2
                -0.962251*lmbda_comp_nu**4+1.47353*lmbda_comp_nu**6
                -0.48953*lmbda_comp_nu**8) 
                / ((1.-lmbda_comp_nu)*(1.+1.01524*lmbda_comp_nu)))
    
    def s_cnu_func(self, lmbda_comp_nu):
        """Nondimensional chain-level entropic free energy contribution
        per segment as calculated by the Jedynak R[9,2] inverse Langevin
        approximate
        
        This function computes the nondimensional chain-level entropic
        free energy contribution per segment as calculated by the
        Jedynak R[9,2] inverse Langevin approximate as a function of the
        result of the equilibrium chain stretch minus the segment
        stretch plus one
        """
        return (0.0602726941412868 * lmbda_comp_nu**8
                + 0.00103401966455583 * lmbda_comp_nu**7
                - 0.162726405850159 * lmbda_comp_nu**6
                - 0.00150537112388157 * lmbda_comp_nu**5
                - 0.00350216312906114 * lmbda_comp_nu**4
                - 0.00254138511870934 * lmbda_comp_nu**3
                + 0.488744117329956 * lmbda_comp_nu**2
                + 0.0071635921950366 * lmbda_comp_nu
                - 0.999999503781195 * np.log(1.00000000002049-lmbda_comp_nu)
                - 0.992044340231098 * np.log(lmbda_comp_nu+0.98498877114821)
                - 0.0150047080499398)
    
    def psi_cnu_func(self, lmbda_nu, lmbda_c_eq):
        """Nondimensional chain-level Helmholtz free energy per segment
        
        This function computes the nondimensional chain-level Helmholtz
        free energy per segment as a function of the segment stretch and
        the equilibrium chain stretch
        """
        lmbda_comp_nu = lmbda_c_eq - lmbda_nu + 1.
        
        return self.u_nu_func(lmbda_nu) + self.s_cnu_func(lmbda_comp_nu)
    
    def xi_c_func(self, lmbda_nu, lmbda_c_eq):
        """Nondimensional chain force
        
        This function computes the nondimensional chain force as a
        function of the segment stretch and the equilibrium chain
        stretch
        """
        lmbda_comp_nu = lmbda_c_eq - lmbda_nu + 1.
        
        return self.inv_L_func(lmbda_comp_nu)
    
    def u_nu_tot_hat_func(self, lmbda_nu_hat, lmbda_nu):
        """Nondimensional total segment potential under an applied chain
        force
        
        This function computes the nondimensional total segment
        potential under an applied chain force as a function of the
        applied segment stretch and the segment stretch specifying a
        particular state in the energy landscape
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return (self.u_nu_func(lmbda_nu)
                - lmbda_nu * self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat))
    
    def u_nu_hat_func(self, lmbda_nu_hat, lmbda_nu):
        """Nondimensional total distorted segment potential under an
        applied chain force
        
        This function computes the nondimensional total distorted
        segment potential under an applied chain force as a function
        of the applied segment stretch and the segment stretch
        specifying a particular state in the energy landscape
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return (self.u_nu_func(lmbda_nu)
                - (lmbda_nu-lmbda_nu_hat)
                * self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat))
    
    def lmbda_nu_xi_c_hat_func(self, xi_c_hat):
        """Segment stretch under an applied chain force
        
        This function computes the segment stretch under an applied 
        chain force as a function of the applied nondimensional chain
        force xi_c_hat
        """
        if xi_c_hat > self.xi_c_crit + self.cond_val:
            error_message = """\
                Applied nondimensional chain force value is greater than the \
                analytically calculated critical maximum nondimensional chain \
                force value xi_c_crit. \
                """
            sys.exit(error_message)
        else:
            return 1. + 1. / self.kappa_nu * xi_c_hat
    
    def lmbda_nu_locmin_hat_func(self, lmbda_nu_hat):
        """Segment stretch corresponding to the local minimum of the
        nondimensional total distorted segment potential under an 
        applied chain force
        
        This function computes the segment stretch corresponding to the
        local minimum of the nondimensional total distorted segment
        potential under an applied chain force as a function of the
        applied segment stretch
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return (1. + 1. / self.kappa_nu
                * self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat))
    
    def lmbda_nu_locmax_hat_func(self, lmbda_nu_hat):
        """Segment stretch corresponding to the local maximum of the
        nondimensional total distorted segment potential under an 
        applied chain force
        
        This function computes the segment stretch corresponding to the
        local maximum of the nondimensional total distorted segment
        potential under an applied chain force as a function of the
        applied segment stretch
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        if lmbda_nu_hat == 1.:
            return np.inf
        
        else:
            cbrt_arg_nmrtr = self.zeta_nu_char**2
            cbrt_arg_dnmntr = (self.kappa_nu 
                                * self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat))
            cbrt_arg = cbrt_arg_nmrtr / cbrt_arg_dnmntr
            return 1. + np.cbrt(cbrt_arg)
    
    def e_nu_sci_hat_func(self, lmbda_nu_hat):
        """Nondimensional segment scission activation energy barrier
        
        This function computes the nondimensional segment scission
        activation energy barrier as a function of the applied segment
        stretch
        """
        lmbda_nu_locmin_hat = self.lmbda_nu_locmin_hat_func(lmbda_nu_hat)
        lmbda_nu_locmax_hat = self.lmbda_nu_locmax_hat_func(lmbda_nu_hat)
        
        # if lmbda_nu_hat == 1
        if lmbda_nu_locmax_hat == np.inf:
            return self.zeta_nu_char
        
        elif lmbda_nu_hat > self.lmbda_nu_crit:
            return 0.
        
        else:
            return (self.u_nu_hat_func(lmbda_nu_hat, lmbda_nu_locmax_hat) 
                    - self.u_nu_hat_func(lmbda_nu_hat, lmbda_nu_locmin_hat))
    
    def epsilon_nu_sci_hat_func(self, lmbda_nu_hat):
        """Nondimensional segment scission energy
        
        This function computes the nondimensional segment scission
        energy as a function of the applied segment stretch
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return (self.psi_cnu_func(lmbda_nu_hat, lmbda_c_eq_hat) 
                + self.zeta_nu_char)
    
    def epsilon_cnu_sci_hat_func(self, lmbda_nu_hat):
        """Nondimensional chain scission energy
        
        This function computes the nondimensional chain scission energy
        as a function of the applied segment stretch
        """
        return self.epsilon_nu_sci_hat_func(lmbda_nu_hat)