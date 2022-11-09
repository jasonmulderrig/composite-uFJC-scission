################################################################################################################################
# General setup
################################################################################################################################

# Import necessary libraries
from __future__ import division
import sys
import numpy as np
from scipy import integrate

# Numerical tolerance parameters
min_exponent = np.log(sys.float_info.min)/np.log(10)
max_exponent = np.log(sys.float_info.max)/np.log(10)
eps_val      = np.finfo(float).eps
cond_val     = eps_val*5e12

################################################################################################################################
# Composite extensible freely jointed chain
################################################################################################################################

class CompositeuFJC:
    
    ############################################################################################################################
    # Constructor
    ############################################################################################################################
    
    def __init__(self, **kwargs):
        
        # Initialize default parameter values
        rate_dependence          = kwargs.get("rate_dependence", None)
        omega_0                  = kwargs.get("omega_0", None)
        nu                       = kwargs.get("nu", None)
        nu_b                     = kwargs.get("nu_b", None)
        zeta_b_char              = kwargs.get("zeta_b_char", None)
        kappa_b                  = kwargs.get("kappa_b", None)
        zeta_nu_char             = kwargs.get("zeta_nu_char", None)
        kappa_nu                 = kwargs.get("kappa_nu", None)
        
        # Check the correctness of the specified parameters, calculate segment-level parameters from provided bond-level parameters if necessary, and retain specified parameters
        if rate_dependence != 'rate_dependent' and rate_dependence != 'rate_independent':
            sys.exit('Error: Need to specify the chain dependence on the rate of applied deformation. Either rate-dependent or rate-independent deformation can be used.')
        if rate_dependence == 'rate_dependent' and omega_0 is None:
            sys.exit('Error: Need to specify the microscopic frequency of segments in the chains for rate-dependent deformation.')
        if nu is None: # required to specify nu
            sys.exit('Error: Need to specify nu in the composite uFJC')
        elif nu_b is None: # take given zeta_nu_char and kappa_nu if nu_b is not specified
            if zeta_nu_char is None:
                sys.exit('Error: Need to specify zeta_nu_char in the composite uFJC when nu_b is not specified')
            elif kappa_nu is None:
                sys.exit('Error: Need to specify kappa_nu in the composite uFJC when nu_b is not specified')
        elif nu_b is not None: # use given zeta_b_char and kappa_b to calculate zeta_nu_char and kappa_nu if nu_b is specified
            if zeta_b_char is None:
                sys.exit('Error: Need to specify zeta_b_char in the composite uFJC when nu_b is specified')
            elif kappa_b is None:
                sys.exit('Error: Need to specify kappa_b in the composite uFJC when nu_b is specified')
            else:
                zeta_nu_char = nu_b*zeta_b_char
                kappa_nu     = nu_b*kappa_b
        
        self.rate_dependence = rate_dependence
        self.omega_0         = omega_0
        self.nu              = nu
        self.nu_b            = nu_b
        self.zeta_b_char     = zeta_b_char
        self.kappa_b         = kappa_b
        self.zeta_nu_char    = zeta_nu_char
        self.kappa_nu        = kappa_nu
        
        # Analytically derived physical parameters; retain specified parameters
        self.lmbda_nu_ref    = 1.
        self.lmbda_c_eq_ref  = 0.
        self.lmbda_nu_crit   = 1. + np.sqrt(self.zeta_nu_char/self.kappa_nu)
        self.lmbda_c_eq_crit = 1. + np.sqrt(self.zeta_nu_char/self.kappa_nu) - np.sqrt(1./(self.kappa_nu*self.zeta_nu_char))
        self.xi_c_crit       = np.sqrt(zeta_nu_char*kappa_nu)
        
        # Parameters needed for numerical calculations
        self.lmbda_nu_hat_inc = 0.0005
        
        # Numerically calculated parameters; retain specified parameters
        self.lmbda_c_eq_pade2berg_crit = self.lmbda_c_eq_pade2berg_crit_func()
        self.lmbda_nu_pade2berg_crit   = self.lmbda_nu_func(self.lmbda_c_eq_pade2berg_crit)
        self.epsilon_nu_diss_hat_crit  = self.epsilon_nu_diss_hat_crit_func()
        self.A_nu                      = self.A_nu_func()
        self.Lambda_nu_ref             = self.lmbda_nu_func(self.A_nu)
    
    ############################################################################################################################
    # Methods
    ############################################################################################################################
    
    # Nondimensional harmonic segment potential energy
    def u_nu_har_func(self, lmbda_nu):
        return 0.5*self.kappa_nu*( lmbda_nu - 1. )**2 - self.zeta_nu_char
    
    # Nondimensional sub-critical chain state segment potential energy
    def u_nu_subcrit_func(self, lmbda_nu):
        return self.u_nu_har_func(lmbda_nu)
    
    # Nondimensional super-critical chain state segment potential energy
    def u_nu_supercrit_func(self, lmbda_nu):
        return -self.zeta_nu_char**2 / ( 2.*self.kappa_nu*( lmbda_nu - 1. )**2 )
    
    # Nondimensional uFJC segment potential energy conditional function
    def u_nu_cond_func(self, lmbda_nu):
        
        if lmbda_nu < self.lmbda_nu_crit:
            return self.u_nu_subcrit_func(lmbda_nu)
        
        else:
            return self.u_nu_supercrit_func(lmbda_nu)
    
    # Nondimensional harmonic segment potential energy contribution
    def u_nu_har_comp_func(self, lmbda_nu):
        return ( 1. - self.M_func( self.kappa_nu*np.sign( lmbda_nu - 1. )*( lmbda_nu - 1. )**2 - self.zeta_nu_char ) / ( self.M_func( self.kappa_nu*np.sign( lmbda_nu - 1. )*( lmbda_nu - 1. )**2 - self.zeta_nu_char ) + self.zeta_nu_char ))**2 * 0.5*self.kappa_nu*( lmbda_nu - 1. )**2 - self.zeta_nu_char
    
    # Nondimensional segment scission energy contribution
    def u_nu_sci_comp_func(self, lmbda_nu):
        return self.zeta_nu_char*self.M_func( self.kappa_nu*np.sign( lmbda_nu - 1. )*( lmbda_nu - 1. )**2 - self.zeta_nu_char ) / ( self.M_func( self.kappa_nu*np.sign( lmbda_nu - 1. )*( lmbda_nu - 1. )**2 - self.zeta_nu_char ) + self.zeta_nu_char ) - self.zeta_nu_char
    
    # Nondimensional uFJC segment potential energy
    def u_nu_func(self, lmbda_nu):
        return self.u_nu_har_comp_func(lmbda_nu) + self.u_nu_sci_comp_func(lmbda_nu) + self.zeta_nu_char
    
    # Macaulay brackets
    def M_func(self, x):
        if x < 0:
            return 0.
        else:
            return x
    
    # Segment stretch as a function of equilibrium chain stretch via the Bergstrom approximation
    def subcrit_lmbda_nu_berg_approx_func(self, lmbda_c_eq):
        return ( lmbda_c_eq + 1. + np.sqrt( lmbda_c_eq**2 - 2.*lmbda_c_eq + 1. + 4./self.kappa_nu ))/2.
    
    # Segment stretch as a function of equilibrium chain stretch via the Pade approximation
    def subcrit_lmbda_nu_pade_approx_func(self, lmbda_c_eq):
        if lmbda_c_eq == 0.: # analytical solution, achieved with the use of the Pade approximant
            return 1.
        
        else: # Pade approximant
            alpha_tilde = 1.
            beta_tilde  = -(( 3.*( self.kappa_nu + 1. ) + lmbda_c_eq*( 2.*self.kappa_nu + 3. ))/( self.kappa_nu + 1. ))
            gamma_tilde = (( 2.*self.kappa_nu + lmbda_c_eq*( 4.*self.kappa_nu + 6. + lmbda_c_eq*( self.kappa_nu + 3. )))/( self.kappa_nu + 1. ))
            delta_tilde = (( 2. - lmbda_c_eq*( 2.*self.kappa_nu + lmbda_c_eq*( self.kappa_nu + 3. + lmbda_c_eq )))/( self.kappa_nu + 1. ))
            pi_tilde    = ( 3.*alpha_tilde*gamma_tilde - beta_tilde**2 )/( 3.*alpha_tilde**2 )
            rho_tilde   = ( 2.*beta_tilde**3 - 9.*alpha_tilde*beta_tilde*gamma_tilde + 27.*alpha_tilde**2*delta_tilde )/( 27.*alpha_tilde**3 )
            
            arccos_arg = 3.*rho_tilde/(2.*pi_tilde)*np.sqrt(-3./pi_tilde)
            return 2.*np.sqrt(-pi_tilde/3.)*np.cos( 1./3.*np.arccos(arccos_arg) - 2.*np.pi/3. ) - beta_tilde/(3.*alpha_tilde)
    
    # Function to calculate the critical equilibrium chain stretch value, as a function of nondimensional segment stiffness, below and above which the Pade and Bergstrom approximates are to be respectively used
    def lmbda_c_eq_pade2berg_crit_func(self):
        n = 0.818706900266885 # Calculated from scipy optimize curve_fit analysis
        b = 0.61757545643322586 # Calculated from scipy optimize curve_fit analysis
        return 1./(self.kappa_nu**n) + b
    
    # Function to calculate the segment stretch and equilibrium stretch values below and above which the Pade and Bergstrom approximates are to be respectively used
    def pade2berg_crit_func(self):
        lmbda_c_eq_min   = 0.
        lmbda_c_eq_max   = 1.
        lmbda_c_eq_steps = np.linspace(lmbda_c_eq_min, lmbda_c_eq_max, int(1e4)+1)
        
        # Make arrays to allocate results
        lmbda_c_eq         = []
        lmbda_nu_bergapprx = []
        lmbda_nu_padeapprx = []
        
        for lmbda_c_eq_indx in range(len(lmbda_c_eq_steps)):
            lmbda_c_eq_val         = lmbda_c_eq_steps[lmbda_c_eq_indx]
            lmbda_nu_bergapprx_val = self.subcrit_lmbda_nu_berg_approx_func(lmbda_c_eq_val)
            lmbda_nu_padeapprx_val = self.subcrit_lmbda_nu_pade_approx_func(lmbda_c_eq_val)
            
            lmbda_c_eq         = np.append(lmbda_c_eq, lmbda_c_eq_val)
            lmbda_nu_bergapprx = np.append(lmbda_nu_bergapprx, lmbda_nu_bergapprx_val)
            lmbda_nu_padeapprx = np.append(lmbda_nu_padeapprx, lmbda_nu_padeapprx_val)
        
        pade2berg_crit_indx       = np.argmin(np.abs(lmbda_nu_padeapprx-lmbda_nu_bergapprx))
        lmbda_nu_pade2berg_crit   = min([lmbda_nu_bergapprx[pade2berg_crit_indx], lmbda_nu_padeapprx[pade2berg_crit_indx]])
        lmbda_c_eq_pade2berg_crit = lmbda_c_eq[pade2berg_crit_indx]
        return lmbda_nu_pade2berg_crit, lmbda_c_eq_pade2berg_crit
    
    # Equilibrium chain stretch as a function of segment stretch
    def lmbda_c_eq_func(self, lmbda_nu):
        
        if lmbda_nu == 1.: # analytical solution, achieved with the use of the Pade approximant
            return 0.
        
        elif lmbda_nu < self.lmbda_nu_pade2berg_crit: # Pade approximant
            alpha_tilde = 1.
            beta_tilde  = ( self.kappa_nu + 3. )*( 1. - lmbda_nu )
            gamma_tilde = ( 2.*self.kappa_nu + 3. )*( lmbda_nu**2 - 2.*lmbda_nu ) + 2.*self.kappa_nu
            delta_tilde = ( self.kappa_nu + 1. )*( 3.*lmbda_nu**2 - lmbda_nu**3 ) - 2.*( self.kappa_nu*lmbda_nu + 1. )
            pi_tilde    = ( 3.*alpha_tilde*gamma_tilde - beta_tilde**2 )/( 3.*alpha_tilde**2 )
            rho_tilde   = ( 2.*beta_tilde**3 - 9.*alpha_tilde*beta_tilde*gamma_tilde + 27.*alpha_tilde**2*delta_tilde )/( 27.*alpha_tilde**3 )
            
            arccos_arg = 3.*rho_tilde/(2.*pi_tilde)*np.sqrt(-3./pi_tilde)
            return 2.*np.sqrt(-pi_tilde/3.)*np.cos( 1./3.*np.arccos(arccos_arg) - 2.*np.pi/3. ) - beta_tilde/(3.*alpha_tilde)
        
        elif lmbda_nu < self.lmbda_nu_crit: # Bergstrom approximant
            return lmbda_nu - 1./self.kappa_nu*1./( lmbda_nu - 1. )
        
        else: # Bergstrom approximant
            return lmbda_nu - self.kappa_nu/self.zeta_nu_char**2*( lmbda_nu - 1. )**3
    
    # Segment stretch as a function of equilibrium chain stretch
    def lmbda_nu_func(self, lmbda_c_eq):
        
        if lmbda_c_eq == 0.: # analytical solution, achieved with the use of the Pade approximant
            return 1.
        
        elif lmbda_c_eq < self.lmbda_c_eq_pade2berg_crit: # Pade approximant
            alpha_tilde = 1.
            beta_tilde  = -(( 3.*( self.kappa_nu + 1. ) + lmbda_c_eq*( 2.*self.kappa_nu + 3. ))/( self.kappa_nu + 1. ))
            gamma_tilde = (( 2.*self.kappa_nu + lmbda_c_eq*( 4.*self.kappa_nu + 6. + lmbda_c_eq*( self.kappa_nu + 3. )))/( self.kappa_nu + 1. ))
            delta_tilde = (( 2. - lmbda_c_eq*( 2.*self.kappa_nu + lmbda_c_eq*( self.kappa_nu + 3. + lmbda_c_eq )))/( self.kappa_nu + 1. ))
            pi_tilde    = ( 3.*alpha_tilde*gamma_tilde - beta_tilde**2 )/( 3.*alpha_tilde**2 )
            rho_tilde   = ( 2.*beta_tilde**3 - 9.*alpha_tilde*beta_tilde*gamma_tilde + 27.*alpha_tilde**2*delta_tilde )/( 27.*alpha_tilde**3 )
            
            arccos_arg = 3.*rho_tilde/(2.*pi_tilde)*np.sqrt(-3./pi_tilde)
            return 2.*np.sqrt(-pi_tilde/3.)*np.cos( 1./3.*np.arccos(arccos_arg) - 2.*np.pi/3. ) - beta_tilde/(3.*alpha_tilde)
        
        elif lmbda_c_eq < self.lmbda_c_eq_crit: # Bergstrom approximant
            return ( lmbda_c_eq + 1. + np.sqrt( lmbda_c_eq**2 - 2.*lmbda_c_eq + 1. + 4./self.kappa_nu ))/2.
        
        else: # Bergstrom approximant
            alpha_tilde = 1.
            beta_tilde  = -3.
            gamma_tilde = 3. - self.zeta_nu_char**2/self.kappa_nu
            delta_tilde = self.zeta_nu_char**2/self.kappa_nu*lmbda_c_eq - 1.
            pi_tilde    = ( 3.*alpha_tilde*gamma_tilde - beta_tilde**2 )/( 3.*alpha_tilde**2 )
            rho_tilde   = ( 2.*beta_tilde**3 - 9.*alpha_tilde*beta_tilde*gamma_tilde + 27.*alpha_tilde**2*delta_tilde )/( 27.*alpha_tilde**3 )
            
            arccos_arg = 3.*rho_tilde/(2.*pi_tilde)*np.sqrt(-3./pi_tilde)
            return 2.*np.sqrt(-pi_tilde/3.)*np.cos( 1./3.*np.arccos(arccos_arg) - 2.*np.pi/3. ) - beta_tilde/(3.*alpha_tilde)

    # Jedynak R[9,2] inverse Langevin approximate
    def inv_L_func(self, lmbda_comp_nu):
        return lmbda_comp_nu*( 3. - 1.00651*lmbda_comp_nu**2 - 0.962251*lmbda_comp_nu**4 + 1.47353*lmbda_comp_nu**6 - 0.48953*lmbda_comp_nu**8 )/( ( 1. - lmbda_comp_nu )*( 1. + 1.01524*lmbda_comp_nu ) )
    
    # Nondimensional entropic free energy contribution per segment as calculated by the Jedynak R[9,2] inverse Langevin approximate
    def s_cnu_func(self, lmbda_comp_nu):
        return 0.0602726941412868*lmbda_comp_nu**8 + 0.00103401966455583*lmbda_comp_nu**7 - 0.162726405850159*lmbda_comp_nu**6 - 0.00150537112388157*lmbda_comp_nu**5 \
            - 0.00350216312906114*lmbda_comp_nu**4 - 0.00254138511870934*lmbda_comp_nu**3 + 0.488744117329956*lmbda_comp_nu**2 + 0.0071635921950366*lmbda_comp_nu \
                - 0.999999503781195*np.log(1.00000000002049 - lmbda_comp_nu) - 0.992044340231098*np.log(lmbda_comp_nu + 0.98498877114821) - 0.0150047080499398
    
    # Nondimensional Helmholtz free energy per segment
    def psi_cnu_func(self, lmbda_nu, lmbda_c_eq):
        lmbda_comp_nu = lmbda_c_eq - lmbda_nu + 1.
        
        return self.u_nu_func(lmbda_nu) + self.s_cnu_func(lmbda_comp_nu)
    
    # Nondimensional chain force
    def xi_c_func(self, lmbda_nu, lmbda_c_eq):
        lmbda_comp_nu = lmbda_c_eq - lmbda_nu + 1.
        
        return self.inv_L_func(lmbda_comp_nu)
    
    ################### Need to switch the placement of the lmbda_nu and lmbda_nu_hat arguments to match manuscript: u_nu_tot_hat_func(self, lmbda_nu_hat, lmbda_nu)
    # Nondimensional total segment potential under an applied chain force xi_c_hat
    def u_nu_tot_hat_func(self, lmbda_nu, lmbda_nu_hat):
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return self.u_nu_func(lmbda_nu) - lmbda_nu*self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat)
    
    # Nondimensional total distorted segment potential under an applied chain force xi_c_hat
    def u_nu_hat_func(self, lmbda_nu, lmbda_nu_hat):
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return self.u_nu_func(lmbda_nu) - ( lmbda_nu - lmbda_nu_hat )*self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat)
    
    # Segment stretch value under an applied chain force xi_c_hat
    def lmbda_nu_xi_c_hat_func(self, xi_c_hat):
        if xi_c_hat > self.xi_c_crit + cond_val:
            sys.exit('Error: Applied chain force value is greater than the analytically calculated critical maximum chain force value xi_c_crit')
        else:
            return 1. + 1./self.kappa_nu*xi_c_hat
    
    # Segment stretch value of the local minimum of the nondimensional total distorted segment potential under an applied chain force xi_c_hat
    def lmbda_nu_locmin_hat_func(self, lmbda_nu_hat):
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return 1. + 1./self.kappa_nu*self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat)
    
    # Segment stretch value of the local maximum of the nondimensional total distorted segment potential under an applied chain force xi_c_hat
    def lmbda_nu_locmax_hat_func(self, lmbda_nu_hat):
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        if lmbda_nu_hat == 1.:
            return np.inf
        
        else:
            return 1. + np.cbrt(( self.zeta_nu_char**2/self.kappa_nu )*( 1./self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat )))
    
    # Nondimensional segment scission activation energy barrier
    def e_nu_sci_hat_func(self, lmbda_nu_hat):
        lmbda_nu_locmin_hat = self.lmbda_nu_locmin_hat_func(lmbda_nu_hat)
        lmbda_nu_locmax_hat = self.lmbda_nu_locmax_hat_func(lmbda_nu_hat)
        
        if lmbda_nu_locmax_hat == np.inf: # i.e., if lmbda_nu_hat = 1
            return self.zeta_nu_char
        
        elif lmbda_nu_hat > self.lmbda_nu_crit:
            return 0.
        
        else:
            return self.u_nu_hat_func(lmbda_nu_locmax_hat, lmbda_nu_hat) - self.u_nu_hat_func(lmbda_nu_locmin_hat, lmbda_nu_hat)
    
    # Nondimensional segment scission energy
    def epsilon_nu_sci_hat_func(self, lmbda_nu_hat):
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return self.psi_cnu_func(lmbda_nu_hat, lmbda_c_eq_hat) + self.zeta_nu_char
    
    # Probability of segment scission
    def p_nu_sci_hat_func(self, lmbda_nu_hat):
        return np.exp(-self.e_nu_sci_hat_func(lmbda_nu_hat))

    # Probability of segment survival
    def p_nu_sur_hat_func(self, lmbda_nu_hat):
        return 1. - self.p_nu_sci_hat_func(lmbda_nu_hat)
    
    # Probability of chain survival
    def p_c_sur_hat_func(self, lmbda_nu_hat):
        return self.p_nu_sur_hat_func(lmbda_nu_hat)**self.nu
    
    # Probability of chain scission
    def p_c_sci_hat_func(self, lmbda_nu_hat):
        return 1. - self.p_c_sur_hat_func(lmbda_nu_hat)
    
    def upsilon_c_func(self, k_cond_val, lmbda_nu_hat):
        return (1.-k_cond_val)*self.p_c_sur_hat_func(lmbda_nu_hat) + k_cond_val

    def d_c_func(self, k_cond_val, lmbda_nu_hat):
        return 1. - self.upsilon_c_func(k_cond_val, lmbda_nu_hat)
    
    # Nondimensional chain scission energy
    def epsilon_cnu_sci_hat_func(self, lmbda_nu_hat):
        return self.epsilon_nu_sci_hat_func(lmbda_nu_hat)
    
    # Nondimensional rate-independent dissipated segment scission energy
    def epsilon_nu_diss_hat_func(self, lmbda_nu_hat_max, lmbda_nu_hat_val, lmbda_nu_hat_val_prior, epsilon_nu_diss_hat_val_prior):
        if lmbda_nu_hat_val < lmbda_nu_hat_max: # dissipated energy cannot be destroyed
            return epsilon_nu_diss_hat_val_prior
        elif lmbda_nu_hat_max > self.lmbda_nu_crit: # completely broken segments previously dissipated energy
            return epsilon_nu_diss_hat_val_prior
        else:
            if ( lmbda_nu_hat_val - 1. ) <= self.lmbda_nu_hat_inc:
                epsilon_nu_diss_hat_prime_val = 0
            else:
                epsilon_nu_diss_hat_prime_val = self.p_nu_sci_hat_func(lmbda_nu_hat_val)*(np.cbrt(self.zeta_nu_char**2*self.kappa_nu/( lmbda_nu_hat_val - 1. )) - self.kappa_nu*( lmbda_nu_hat_val - 1. ))*self.epsilon_nu_sci_hat_func(lmbda_nu_hat_val)
            
            # the lmbda_nu_hat increment here is guaranteed to be non-negative
            return epsilon_nu_diss_hat_val_prior + epsilon_nu_diss_hat_prime_val*( lmbda_nu_hat_val - lmbda_nu_hat_val_prior )

    # Nondimensional rate-independent dissipated segment scission energy for a chain at the critical state
    def epsilon_nu_diss_hat_crit_func(self):
        # Define the values of the applied segment stretch to calculate over
        lmbda_nu_hat_num_steps = int(np.around((self.lmbda_nu_crit-self.lmbda_nu_ref)/self.lmbda_nu_hat_inc)) + 1
        lmbda_nu_hat_steps     = np.linspace(self.lmbda_nu_ref, self.lmbda_nu_crit, lmbda_nu_hat_num_steps)

        # initialization
        lmbda_nu_hat_max                   = 0
        epsilon_nu_diss_hat_crit_val_prior = 0
        epsilon_nu_diss_hat_crit_val       = 0

        for lmbda_nu_hat_indx in range(lmbda_nu_hat_num_steps):
            lmbda_nu_hat_val = lmbda_nu_hat_steps[lmbda_nu_hat_indx]
            lmbda_nu_hat_max = max([lmbda_nu_hat_max, lmbda_nu_hat_val])
            
            if lmbda_nu_hat_indx == 0: # preserve initialization and continue on
                epsilon_nu_diss_hat_crit_val = 0
            else:
                epsilon_nu_diss_hat_crit_val = self.epsilon_nu_diss_hat_func(lmbda_nu_hat_max, lmbda_nu_hat_val, lmbda_nu_hat_steps[lmbda_nu_hat_indx-1], epsilon_nu_diss_hat_crit_val_prior)
            
            # update values
            epsilon_nu_diss_hat_crit_val_prior = epsilon_nu_diss_hat_crit_val
        
        return epsilon_nu_diss_hat_crit_val
    
    # History-dependent time integral of the probability of segment scission
    def p_nu_sci_hat_cum_intgrl_func(self, p_nu_sci_hat_val, t_val, p_nu_sci_hat_prior, t_prior, p_nu_sci_hat_cum_intgrl_val_prior):
        return p_nu_sci_hat_cum_intgrl_val_prior + integrate.trapezoid([p_nu_sci_hat_prior, p_nu_sci_hat_val], x=[t_prior, t_val])
    
    # Rate-dependent probability of chain scission
    def gamma_c_func(self, p_nu_sci_hat_cum_intgrl_val):
        return 1. - np.exp(-self.nu*self.omega_0*p_nu_sci_hat_cum_intgrl_val)
    
    # Rate-dependent probability of chain survival
    def rho_c_func(self, p_nu_sci_hat_cum_intgrl_val):
        return 1. - self.gamma_c_func(p_nu_sci_hat_cum_intgrl_val)

    # Nondimensional dissipated chain scission energy
    def epsilon_cnu_diss_hat_func(self, **kwargs):
        if self.rate_dependence == 'rate_independent':
            # Initialize parameter values
            lmbda_nu_hat_max               = kwargs.get("lmbda_nu_hat_max", None)
            lmbda_nu_hat_val               = kwargs.get("lmbda_nu_hat_val", None)
            lmbda_nu_hat_val_prior         = kwargs.get("lmbda_nu_hat_val_prior", None)
            epsilon_cnu_diss_hat_val_prior = kwargs.get("epsilon_cnu_diss_hat_val_prior", None)
            
            # Check the correctness of the specified parameters
            if lmbda_nu_hat_max is None:
                sys.exit('Error: Need to specify lmbda_nu_hat_max for epsilon_cnu_diss_hat_func')
            elif lmbda_nu_hat_val is None:
                sys.exit('Error: Need to specify lmbda_nu_hat_val for epsilon_cnu_diss_hat_func')
            elif lmbda_nu_hat_val_prior is None:
                sys.exit('Error: Need to specify lmbda_nu_hat_val_prior for epsilon_cnu_diss_hat_func')
            elif epsilon_cnu_diss_hat_val_prior is None:
                sys.exit('Error: Need to specify epsilon_cnu_diss_hat_val_prior for epsilon_cnu_diss_hat_func')
            
            if lmbda_nu_hat_val < lmbda_nu_hat_max: # dissipated energy cannot be destroyed
                return epsilon_cnu_diss_hat_val_prior
            elif lmbda_nu_hat_max > self.lmbda_nu_crit: # completely broken chains previously dissipated energy
                return epsilon_cnu_diss_hat_val_prior
            else:
                if ( lmbda_nu_hat_val - 1. ) <= self.lmbda_nu_hat_inc:
                    epsilon_cnu_diss_hat_prime_val = 0
                else:
                    epsilon_cnu_diss_hat_prime_val = self.nu*( 1. - self.p_nu_sci_hat_func(lmbda_nu_hat_val) )**(self.nu-1)*self.p_nu_sci_hat_func(lmbda_nu_hat_val)*(np.cbrt(self.zeta_nu_char**2*self.kappa_nu/( lmbda_nu_hat_val - 1. )) - self.kappa_nu*( lmbda_nu_hat_val - 1. ))*self.epsilon_nu_sci_hat_func(lmbda_nu_hat_val)
                
                # the lmbda_nu_hat increment here is guaranteed to be positive in value
                return epsilon_cnu_diss_hat_val_prior + epsilon_cnu_diss_hat_prime_val*( lmbda_nu_hat_val - lmbda_nu_hat_val_prior )
        
        elif self.rate_dependence == 'rate_dependent':
            # Initialize parameter values
            p_nu_sci_hat_val               = kwargs.get("p_nu_sci_hat_val", None)
            p_nu_sci_hat_cum_intgrl_val    = kwargs.get("p_nu_sci_hat_cum_intgrl_val", None)
            t_val                          = kwargs.get("t_val", None)
            lmbda_nu_hat_val               = kwargs.get("lmbda_nu_hat_val", None)
            t_prior                        = kwargs.get("t_prior", None)
            epsilon_cnu_diss_hat_val_prior = kwargs.get("epsilon_cnu_diss_hat_val_prior", None)

            # Check the correctness of the specified parameters
            if p_nu_sci_hat_val is None:
                sys.exit('Error: Need to specify p_nu_sci_hat_val for epsilon_cnu_diss_hat_func')
            elif p_nu_sci_hat_cum_intgrl_val is None:
                sys.exit('Error: Need to specify p_nu_sci_hat_cum_intgrl_val for epsilon_cnu_diss_hat_func')
            elif t_val is None:
                sys.exit('Error: Need to specify t_val for epsilon_cnu_diss_hat_func')
            elif lmbda_nu_hat_val is None:
                sys.exit('Error: Need to specify lmbda_nu_hat_val for epsilon_cnu_diss_hat_func')
            elif t_prior is None:
                sys.exit('Error: Need to specify t_prior for epsilon_cnu_diss_hat_func')
            elif epsilon_cnu_diss_hat_val_prior is None:
                sys.exit('Error: Need to specify epsilon_cnu_diss_hat_val_prior for epsilon_cnu_diss_hat_func')
            
            epsilon_nu_sci_hat_val       = self.epsilon_nu_sci_hat_func(lmbda_nu_hat_val)
            epsilon_cnu_diss_hat_dot_val = np.exp(-self.nu*self.omega_0*p_nu_sci_hat_cum_intgrl_val)*self.nu*self.omega_0*p_nu_sci_hat_val*epsilon_nu_sci_hat_val
            
            return epsilon_cnu_diss_hat_val_prior + epsilon_cnu_diss_hat_dot_val*( t_val - t_prior )
    
    # Integrand involved in the intact equilibrium chain configuration partition function integration
    def I_integrand_func(self, lmbda_c_eq, n, nu):
        lmbda_nu = self.lmbda_nu_func(lmbda_c_eq)
        psi_cnu  = self.psi_cnu_func(lmbda_nu, lmbda_c_eq)
        
        return np.exp(-nu*(psi_cnu + self.zeta_nu_char))*lmbda_c_eq**(n-1)
    
    # Intact equilibrium chain configuration partition function integration through all admissible end-to-end chain distances up to the critical point
    def I_func(self, n, nu):
        return integrate.quad(self.I_integrand_func, self.lmbda_c_eq_ref, self.lmbda_c_eq_crit, args=(n,nu), epsabs=1.0e-12, epsrel=1.0e-12)[0]
    
    # Reference equilibrium chain stretch
    def A_nu_func(self):
        I_5 = self.I_func(5, self.nu)
        I_3 = self.I_func(3, self.nu)
        
        return np.sqrt(1./( 1. + self.nu*np.exp(-self.epsilon_nu_diss_hat_crit) )*I_5/I_3)