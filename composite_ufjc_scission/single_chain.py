"""The core single-chain module for the composite uFJC scission model"""

# Import necessary libraries
from __future__ import division
import sys
import numpy as np
from scipy import integrate

# Define numerical tolerance parameters
min_exponent = np.log(sys.float_info.min) / np.log(10)
max_exponent = np.log(sys.float_info.max) / np.log(10)
eps_val      = np.finfo(float).eps
cond_val     = eps_val * 5e12


class CompositeuFJC:
    """The composite uFJC scission single-chain model class."""
    
    def __init__(self, **kwargs):
        """
        Initializes the ``CompositeuFJC`` class, producing a composite
        uFJC scission single chain model instance.
        """
        
        # Get default parameter values
        rate_dependence = kwargs.get("rate_dependence", None)
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
        
        # Parameters needed for numerical calculations
        self.lmbda_nu_hat_inc = 0.0005
        
        # Calculate and retain numerically calculated parameters
        self.lmbda_c_eq_pade2berg_crit = self.lmbda_c_eq_pade2berg_crit_func()
        self.lmbda_nu_pade2berg_crit   = self.lmbda_nu_func(
            self.lmbda_c_eq_pade2berg_crit)
        self.epsilon_nu_diss_hat_crit  = self.epsilon_nu_diss_hat_crit_func()
        self.A_nu                      = self.A_nu_func()
        self.Lambda_nu_ref             = self.lmbda_nu_func(self.A_nu)
    
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

        return ((1.-M_val/(M_val+self.zeta_nu_char))**2 
                * self.u_nu_har_func(lmbda_nu))
    
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

        return self.zeta_nu_char * (M_val/(M_val+self.zeta_nu_char)-1.)
    
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
        chain stretch.
        """
        sqrt_arg = lmbda_c_eq**2 - 2. * lmbda_c_eq + 1. + 4. / self.kappa_nu
        return (lmbda_c_eq+1.+np.sqrt(sqrt_arg)) / 2.
    
    def subcrit_lmbda_nu_pade_approx_func(self, lmbda_c_eq):
        """Sub-critical chain state segment stretch as derived via the
        Pade approximant for the inverse Langevin function
        
        This function computes the sub-critical chain state segment
        stretch (as derived via the Pade approximant for the inverse
        Langevin function) as a function of the equilibrium chain
        stretch.
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
        if xi_c_hat > self.xi_c_crit + cond_val:
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
    
    def epsilon_cnu_sci_hat_func(self, lmbda_nu_hat):
        """Nondimensional chain scission energy
        
        This function computes the nondimensional chain scission energy
        as a function of the applied segment stretch
        """
        return self.epsilon_nu_sci_hat_func(lmbda_nu_hat)
    
    def epsilon_nu_diss_hat_func(
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
    
    def expctd_val_epsilon_nu_sci_hat_intgrnd_func(
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
            return expctd_val_epsilon_nu_sci_hat_intgrnd_val_prior
            # statistical expected value of the nondimensional segment
            # scission energy from fully broken segments remains fixed
        elif lmbda_nu_hat_max > self.lmbda_nu_crit:
            return expctd_val_epsilon_nu_sci_hat_intgrnd_val_prior
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
    
    def expctd_val_epsilon_nu_sci_hat_cum_intgrl_func(
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
            expctd_val_epsilon_nu_sci_hat_val_prior
            + integrate.trapezoid(
                [expctd_val_epsilon_nu_sci_hat_intgrnd_val_prior,
                expctd_val_epsilon_nu_sci_hat_intgrnd_val],
                x=[epsilon_nu_sci_hat_prior, epsilon_nu_sci_hat_val])
        )

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
                epsilon_nu_diss_hat_crit_val = 0
            else:
                epsilon_nu_diss_hat_crit_val = self.epsilon_nu_diss_hat_func(
                    lmbda_nu_hat_max, lmbda_nu_hat_val,
                    lmbda_nu_hat_steps[lmbda_nu_hat_indx-1],
                    epsilon_nu_diss_hat_crit_val_prior)
            
            # update values
            epsilon_nu_diss_hat_crit_val_prior = epsilon_nu_diss_hat_crit_val
        
        return epsilon_nu_diss_hat_crit_val
    
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

    def epsilon_cnu_diss_hat_func(self, **kwargs):
        """Nondimensional dissipated chain scission energy
        
        This function computes the nondimensional dissipated chain
        scission energy for both rate-independent and rate-dependent
        cases. For the rate-independent case, this is a function of its
        prior value, as well as the current, prior, and maximum applied
        segment stretch values. For the rate-dependent case, this is a
        function of its prior value, the current applied segment
        stretch, the current and prior values of time, the current
        probability of segment scission, and the history-dependent time
        integral of the probability of segment scission
        """
        if self.rate_dependence == 'rate_independent':
            # Initialize parameter values
            lmbda_nu_hat_max       = kwargs.get("lmbda_nu_hat_max", None)
            lmbda_nu_hat_val       = kwargs.get("lmbda_nu_hat_val", None)
            lmbda_nu_hat_val_prior = kwargs.get("lmbda_nu_hat_val_prior", None)
            epsilon_cnu_diss_hat_val_prior = kwargs.get(
                "epsilon_cnu_diss_hat_val_prior", None)
            
            # Check the correctness of the specified parameters
            if lmbda_nu_hat_max is None:
                error_message = """\
                    Error: Need to specify lmbda_nu_hat_max for \
                    epsilon_cnu_diss_hat_func. \
                    """
                sys.exit(error_message)
            elif lmbda_nu_hat_val is None:
                error_message = """\
                    Error: Need to specify lmbda_nu_hat_val for \
                    epsilon_cnu_diss_hat_func. \
                    """
                sys.exit(error_message)
            elif lmbda_nu_hat_val_prior is None:
                error_message = """\
                    Error: Need to specify lmbda_nu_hat_val_prior for \
                    epsilon_cnu_diss_hat_func. \
                    """
                sys.exit(error_message)
            elif epsilon_cnu_diss_hat_val_prior is None:
                error_message = """\
                    Error: Need to specify epsilon_cnu_diss_hat_val_prior for \
                    epsilon_cnu_diss_hat_func. \
                    """
                sys.exit(error_message)
            # dissipated energy cannot be destroyed
            if lmbda_nu_hat_val < lmbda_nu_hat_max:
                return epsilon_cnu_diss_hat_val_prior
            # dissipated energy from fully broken chains remains
            # fixed
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
        
        elif self.rate_dependence == 'rate_dependent':
            # Initialize parameter values
            p_nu_sci_hat_val = kwargs.get("p_nu_sci_hat_val", None)
            p_nu_sci_hat_cum_intgrl_val = kwargs.get(
                "p_nu_sci_hat_cum_intgrl_val", None)
            t_val = kwargs.get("t_val", None)
            lmbda_nu_hat_val = kwargs.get("lmbda_nu_hat_val", None)
            t_prior = kwargs.get("t_prior", None)
            epsilon_cnu_diss_hat_val_prior = kwargs.get(
                "epsilon_cnu_diss_hat_val_prior", None)

            # Check the correctness of the specified parameters
            if p_nu_sci_hat_val is None:
                error_message = """\
                    Error: Need to specify p_nu_sci_hat_val for \
                    epsilon_cnu_diss_hat_func. \
                    """
                sys.exit(error_message)
            elif p_nu_sci_hat_cum_intgrl_val is None:
                error_message = """\
                    Error: Need to specify p_nu_sci_hat_cum_intgrl_val for \
                    epsilon_cnu_diss_hat_func. \
                    """
                sys.exit(error_message)
            elif t_val is None:
                error_message = """\
                    Error: Need to specify t_val for \
                    epsilon_cnu_diss_hat_func. \
                    """
                sys.exit(error_message)
            elif lmbda_nu_hat_val is None:
                error_message = """\
                    Error: Need to specify lmbda_nu_hat_val for \
                    epsilon_cnu_diss_hat_func. \
                    """
                sys.exit(error_message)
            elif t_prior is None:
                error_message = """\
                    Error: Need to specify t_prior for \
                    epsilon_cnu_diss_hat_func. \
                    """
                sys.exit(error_message)
            elif epsilon_cnu_diss_hat_val_prior is None:
                error_message = """\
                    Error: Need to specify epsilon_cnu_diss_hat_val_prior for \
                    epsilon_cnu_diss_hat_func. \
                    """
                sys.exit(error_message)
            
            epsilon_nu_sci_hat_val = (
                self.epsilon_nu_sci_hat_func(lmbda_nu_hat_val)
            )
            epsilon_cnu_diss_hat_dot_val = (
                np.exp(-self.nu*self.omega_0*p_nu_sci_hat_cum_intgrl_val)
                * self.nu * self.omega_0 * p_nu_sci_hat_val 
                * epsilon_nu_sci_hat_val
            )
            
            return (epsilon_cnu_diss_hat_val_prior 
                    + epsilon_cnu_diss_hat_dot_val * (t_val-t_prior))
    
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