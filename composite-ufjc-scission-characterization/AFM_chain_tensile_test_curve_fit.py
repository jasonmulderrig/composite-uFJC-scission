"""The single-chain AFM tensile test curve fit validation 
characterization module for composite uFJCs that undergo scission
"""

# import external modules
from __future__ import division
from composite_ufjc_scission import (CompositeuFJCScissionCharacterizer,
RateIndependentScissionCompositeuFJC,
RateDependentScissionCompositeuFJC,
latex_formatting_figure,
save_current_figure,
save_current_figure_no_labels
)
import numpy as np
from math import floor, log10
from scipy import optimize
from scipy import constants
import matplotlib.pyplot as plt


class AFMChainTensileTestCurveFitCharacterizer(
        CompositeuFJCScissionCharacterizer):
    """The characterization class assessing the single-chain AFM tensile
    test curve fit validation for composite uFJCs that undergo scission.
    It inherits all attributes and methods from the
    ``CompositeuFJCScissionCharacterizer`` class.
    """
    def __init__(self, paper_authors, chain, T):
        """Initializes the ``AFMChainTensileTestCurveFitCharacterizer``
        class by initializing and inheriting all attributes and methods
        from the ``CompositeuFJCScissionCharacterizer`` class.
        """
        self.paper_authors = paper_authors
        self.chain = chain
        self.T = T

        CompositeuFJCScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        k_B  = constants.value(u'Boltzmann constant') # J/K
        N_A  = constants.value(u'Avogadro constant') # 1/mol
        beta = 1. / (k_B*self.T) # 1/J

        p = self.parameters

        p.characterizer.chain_data_directory = (
            "./AFM_chain_tensile_test_curve_fit_data/"
        )

        p.characterizer.paper_authors2polymer_type_dict = {
            "al-maawali-et-al": "pdms",
            "ortiz-and-hadziioannou": "pmaa",
            "hugel-et-al": "pva",
            "sakai-et-al": "ps",
            "yamamoto-et-al": "pmma",
            "zhang-et-al": "pnipam",
            "wang-et-al": "pdma"
        }
        p.characterizer.paper_authors2polymer_type_label_dict = {
            "al-maawali-et-al": r'$\textrm{PDMS single chain data}$',
            "ortiz-and-hadziioannou": r'$\textrm{PMAA single chain data}$',
            "hugel-et-al": r'$\textrm{PVA single chain data}$',
            "sakai-et-al": r'$\textrm{PS single chain data}$',
            "yamamoto-et-al": r'$\textrm{PMMA single chain data}$',
            "zhang-et-al": r'$\textrm{PNIPAM single chain data}$',
            "wang-et-al": r'$\textrm{PDMA single chain data}$'
        }
        p.characterizer.polymer_type_label2chain_backbone_bond_type_dict = {
            "pdms": "si-o",
            "pmaa": "c-c",
            "pva": "c-c",
            "ps": "c-c",
            "pmma": "c-c",
            "pnipam": "c-c",
            "pdma": "c-c"
        }
        p.characterizer.paper_authors_chain2xlim_chain_mechanical_response_plot = {
            "al-maawali-et-al": {
                "chain-a": [0, 70]
            },
            "ortiz-and-hadziioannou": {
                "chain-a": [0, 30],
                "chain-b": [0, 30],
                "chain-c": [0, 40],
                "chain-d": [0, 50],
                "chain-e": [0, 70],
                "chain-f": [0, 80],
                "chain-g": [0, 80],
                "chain-h": [0, 100]
            },
            "hugel-et-al": {
                "chain-a": [0, 1200],
                "chain-b": [0, 5000]
            },
            "sakai-et-al": {
                "chain-a": [0, 50]
            },
            "yamamoto-et-al": {
                "chain-a": [0, 100],
                "chain-b": [0, 125],
                "chain-c": [0, 175],
                "chain-d": [0, 200],
                "chain-e": [0, 250]
            },
            "zhang-et-al": {
                "chain-a": [0, 500]
            },
            "wang-et-al": {
                "chain-a": [0, 750],
                "chain-b": [0, 750],
                "chain-c": [0, 1250]
            }
        }
        p.characterizer.paper_authors_chain2ylim_chain_mechanical_response_plot = {
            "al-maawali-et-al": {
                "chain-a": [-0.1, 2]
            },
            "ortiz-and-hadziioannou": {
                "chain-a": [-0.1, 0.5],
                "chain-b": [-0.1, 0.5],
                "chain-c": [-0.1, 0.5],
                "chain-d": [-0.1, 0.5],
                "chain-e": [-0.1, 0.5],
                "chain-f": [-0.1, 0.5],
                "chain-g": [-0.1, 0.5],
                "chain-h": [-0.1, 0.5]
            },
            "hugel-et-al": {
                "chain-a": [-0.1, 1.5],
                "chain-b": [-0.1, 2.5]
            },
            "sakai-et-al": {
                "chain-a": [-0.02, 0.15]
            },
             "yamamoto-et-al": {
                "chain-a": [-0.1, 0.6],
                "chain-b": [-0.1, 0.6],
                "chain-c": [-0.1, 0.6],
                "chain-d": [-0.1, 0.6],
                "chain-e": [-0.1, 0.6]
            },
            "zhang-et-al": {
                "chain-a": [-0.1, 1.5]
            },
            "wang-et-al": {
                "chain-a": [-0.1, 1.5],
                "chain-b": [-0.1, 2.0],
                "chain-c": [-0.1, 1.0]
            }
        }
        # from DFT simulations on H_3C-CH_2-CH_3 (c-c) 
        # and H_3Si-O-CH_3 (si-o) by Beyer, J Chem. Phys., 2000
        p.characterizer.chain_backbone_bond_type2f_c_max_dict = {
            "c-c": 6.92,
            "si-o": 5.20
        } # nN
        # (c-c) from the CRC Handbook of Chemistry and Physics for
        # CH_3-C_2H_5 citing Luo, Y.R., Comprehensive Handbook of
        # Chemical Bond Energies, CRC Press, 2007
        # (si-o) from Schwaderer et al., Langmuir, 2008, citing Holleman
        # and Wilberg, Inorganic Chemistry, 2001
        chain_backbone_bond_type2epsilon_b_char_dict = {
            "c-c": 370.3,
            "si-o": 444
        } # kJ/mol
        p.characterizer.chain_backbone_bond_type2zeta_b_char_dict = {
            chain_backbone_bond_type_key: epsilon_b_char_val/N_A*1000*beta
            for chain_backbone_bond_type_key, epsilon_b_char_val
            in chain_backbone_bond_type2epsilon_b_char_dict.items()
        } # (kJ/mol -> kJ -> J)*1/J
        # (c-c) from the CRC Handbook of Chemistry and Physics for the
        # C-C bond in C#-H_2C-CH_2-C#
        # (si-o) from the CRC Handbook of Chemistry and Physics for the
        # Si-O bond in X_3-Si-O-C#
        chain_backbone_bond_type2l_b_eq_dict = {
            "c-c": 1.524,
            "si-o": 1.645
        } # Angstroms
        p.characterizer.chain_backbone_bond_type2l_b_eq_dict = {
            chain_backbone_bond_type_key: l_b_eq_val/10
            for chain_backbone_bond_type_key, l_b_eq_val
            in chain_backbone_bond_type2l_b_eq_dict.items()
        } # Angstroms -> nm

        p.characterizer.r_nu_fit_min = 0. # nm
        p.characterizer.r_nu_fit_inc = 1  # nm
        
        p.characterizer.l_nu_eq_max = 10 # nm
        
        p.characterizer.nu_min = 10
        p.characterizer.nu_max = 25000
        
        p.characterizer.kappa_nu_min = 100
        p.characterizer.kappa_nu_max = 5000

        p.characterizer.f_c_num_steps = 100001

        f_c_dot_list = [1e1, 1e5, 1e9] # nN/sec
        f_c_dot_exponent_list = [
            int(floor(log10(abs(f_c_dot_list[i]))))
            for i in range(len(f_c_dot_list))
        ]
        f_c_dot_label_list = [
            r'$\dot{f}_c='+'10^{0:d}'.format(f_c_dot_exponent_list[i])+'~nN/sec$'
            for i in range(len(f_c_dot_list))
        ]
        f_c_dot_color_list = ['orange', 'purple', 'green']

        p.characterizer.f_c_dot_list          = f_c_dot_list
        p.characterizer.f_c_dot_exponent_list = f_c_dot_exponent_list
        p.characterizer.f_c_dot_label_list    = f_c_dot_label_list
        p.characterizer.f_c_dot_color_list    = f_c_dot_color_list

    def prefix(self):
        """Set characterization prefix"""
        return "AFM_chain_tensile_test_curve_fit"
    
    def characterization(self):
        """Define characterization routine"""

        def chain_data_file_load_func(data_file_prefix, chain_data_directory):
            """Custom AFM single chain tensile test data file loading
            function

            This function loads in the force versus chain extension data
            from a directory containing a selection of such data from
            the AFM single chain tensile testing literature. Depending
            on the data, sensible numerical modificiations are made so
            that the data are physically meaningful. This is a function
            of the data filename and the data directory name.
            """
            if (self.paper_authors == "al-maawali-et-al"or
            self.paper_authors == "ortiz-and-hadziioannou" or
            self.paper_authors == "yamamoto-et-al"):
                data_file_name = (
                    data_file_prefix
                    + "-force-nN-vs-chain-end-to-end-distance-nm"
                )
            else:
                data_file_name = (
                    data_file_prefix
                    + "-force-pN-vs-chain-end-to-end-distance-nm"
                )
            # nm, nN or pN respectively
            # pre-arranged in ascending r_nu values
            r_nu, f_c = (
                np.loadtxt(
                    chain_data_directory+data_file_name+".csv", delimiter=',',
                    unpack=True)
            )
            if (self.paper_authors == "al-maawali-et-al" or
            self.paper_authors == "ortiz-and-hadziioannou" or
            self.paper_authors == "yamamoto-et-al"):
                f_c = -f_c
                if self.paper_authors == "al-maawali-et-al":
                    # chain force reading shifting correction
                    f_c = f_c - f_c[0]
            else:
                f_c = f_c / 1000 # pN -> nN
                if (self.paper_authors == "zhang-et-al" or
                self.paper_authors == "wang-et-al"):
                    # chain force reading shifting correction
                    f_c = f_c - f_c[0]
            
            return r_nu, f_c  # nm, nN respectively

        def l_nu_eq_nu_kappa_nu_fit_func(r_nu, f_c):
            """Comprehensive chain force versus extension curve fit
            
            This function returns the segment length, segment number,
            and nondimensional segment stiffness for the chain involved
            in a given AFM tensile test for the cases where the segment
            number and the number of bonds per segment are and are not
            restricted to integer values. These parameters are
            determined via a scheme utilizing scipy optimize curve_fit.
            It is a function of the chain force versus extension data
            from the AFM tensile test, and is implicitly a function of
            the nondimensional characteristic bond potential energy
            scale and bond length
            """
            def f_c_func_l_nu_eq_nu_kappa_nu_fit(r_nu, l_nu_eq, nu, kappa_nu):
                """Chain force calculated from chain extension, segment
                length, segment number, and nondimensional segment
                stiffness

                This function returns the chain force given chain
                extension, segment length, segment number, and
                nondimensional segment stiffness. It is also
                implicitly a function of the nondimensional
                characteristic bond potential energy scale and bond
                length. This function is also used to determine the
                segment length, segment number, and nondimensional
                segment stiffness for a chain via a scipy optimize
                curve_fit analysis
                """
                def lmbda_c_eq_pade2berg_crit_func(kappa_nu):
                    """Pade-to-Bergstrom (P2B) critical equilibrium
                    chain stretch value
                    
                    This function returns Pade-to-Bergstrom (P2B)
                    critical equilibrium chain stretch value as
                    determined via a scipy optimize curve_fit analysis.
                    It is a function of the nondimensional segment
                    stiffness
                    """
                    n = 0.818706900266885
                    b = 0.61757545643322586
                    return 1. / kappa_nu**n + b
                
                def lmbda_c_eq_crit_func(zeta_nu_char, kappa_nu):
                    """Critical equilibrium chain stretch
                    
                    This function computes critical equilibrium chain
                    stretch value as a function of nondimensional
                    characteristic segment potential energy scale and
                    nondimensional segment stiffness
                    """
                    return (
                        1. + np.sqrt(zeta_nu_char/kappa_nu)
                        - np.sqrt(1./(kappa_nu*zeta_nu_char))
                    )
                
                def lmbda_nu_func(lmbda_c_eq):
                    """Segment stretch

                    This function computes the segment stretch as a
                    function of the equilibrium chain stretch.
                    """
                    lmbda_nu = []
                    
                    for lmbda_c_eq_indx in range(len(lmbda_c_eq)):
                        lmbda_c_eq_val = lmbda_c_eq[lmbda_c_eq_indx]
                    
                        # analytical solution (Pade approximant)
                        if lmbda_c_eq_val == 0.:
                            lmbda_nu_val = 1.
                        
                        # Pade approximant
                        elif lmbda_c_eq_val < lmbda_c_eq_pade2berg_crit:
                            alpha_tilde = 1.

                            trm_i  = -3. * (kappa_nu+1.)
                            trm_ii = -(2.*kappa_nu+3.)
                            beta_tilde_nmrtr  = trm_i + lmbda_c_eq_val * trm_ii
                            beta_tilde_dnmntr = kappa_nu + 1.
                            beta_tilde  = beta_tilde_nmrtr / beta_tilde_dnmntr

                            trm_i   = 2. * kappa_nu
                            trm_ii  = 4. * kappa_nu + 6.
                            trm_iii = kappa_nu + 3.
                            gamma_tilde_nmrtr = (
                                trm_i + lmbda_c_eq_val 
                                * (trm_ii+lmbda_c_eq_val*trm_iii)
                            )
                            gamma_tilde_dnmntr = kappa_nu + 1.
                            gamma_tilde = gamma_tilde_nmrtr / gamma_tilde_dnmntr
                            
                            trm_i   = 2.
                            trm_ii  = 2. * kappa_nu
                            trm_iii = kappa_nu + 3.
                            delta_tilde_nmrtr = (
                                trm_i - lmbda_c_eq_val * 
                                (trm_ii+lmbda_c_eq_val*(trm_iii+lmbda_c_eq_val))
                            )
                            delta_tilde_dnmntr = kappa_nu + 1.
                            delta_tilde = delta_tilde_nmrtr / delta_tilde_dnmntr

                            pi_tilde_nmrtr  = (
                                3. * alpha_tilde * gamma_tilde - beta_tilde**2
                            )
                            pi_tilde_dnmntr = 3. * alpha_tilde**2
                            pi_tilde = pi_tilde_nmrtr / pi_tilde_dnmntr

                            rho_tilde_nmrtr = (
                                2. * beta_tilde**3
                                - 9. * alpha_tilde * beta_tilde * gamma_tilde
                                + 27. * alpha_tilde**2 * delta_tilde
                            )
                            rho_tilde_dnmntr = 27. * alpha_tilde**3
                            rho_tilde = rho_tilde_nmrtr / rho_tilde_dnmntr

                            arccos_arg = (
                                3. * rho_tilde / (2.*pi_tilde)
                                * np.sqrt(-3./pi_tilde)
                            )
                            
                            if arccos_arg >= 1. - 1.e-14:
                                arccos_arg =  1. - 1.e-14
                            elif arccos_arg < -1. + 1.e-14:
                                arccos_arg =  -1. + 1.e-14
                            
                            cos_arg = (
                                1. / 3. * np.arccos(arccos_arg)
                                - 2. * np.pi / 3.
                            )
                            lmbda_nu_val = (
                                2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)
                                - beta_tilde / (3.*alpha_tilde)
                            )
                        
                        # Bergstrom approximate
                        elif lmbda_c_eq_val <= lmbda_c_eq_crit:
                            sqrt_arg = (
                                lmbda_c_eq_val**2 - 2. * lmbda_c_eq_val + 1.
                                + 4. / kappa_nu
                            )
                            lmbda_nu_val = (
                                (lmbda_c_eq_val+1.+np.sqrt(sqrt_arg)) / 2.
                            )
                        
                        # Bergstrom approximate
                        else:
                            alpha_tilde = 1.
                            beta_tilde  = -3.
                            gamma_tilde = 3. - zeta_nu_char**2 / kappa_nu
                            delta_tilde = (
                                zeta_nu_char**2 / kappa_nu * lmbda_c_eq_val - 1.
                            )

                            pi_tilde_nmrtr = (
                                3. * alpha_tilde * gamma_tilde - beta_tilde**2
                            )
                            pi_tilde_dnmntr = 3. * alpha_tilde**2
                            pi_tilde = pi_tilde_nmrtr / pi_tilde_dnmntr

                            rho_tilde_nmrtr = (
                                2. * beta_tilde**3
                                - 9. * alpha_tilde * beta_tilde * gamma_tilde
                                + 27. * alpha_tilde**2 * delta_tilde
                            )
                            rho_tilde_dnmntr = 27. * alpha_tilde**3
                            rho_tilde = rho_tilde_nmrtr / rho_tilde_dnmntr

                            arccos_arg = (
                                3. * rho_tilde / (2.*pi_tilde)
                                * np.sqrt(-3./pi_tilde)
                            )
                            
                            if arccos_arg >= 1. - 1.e-14:
                                arccos_arg =  1. - 1.e-14
                            elif arccos_arg < -1. + 1.e-14:
                                arccos_arg =  -1. + 1.e-14
                            
                            cos_arg = (
                                1. / 3. * np.arccos(arccos_arg)
                                - 2. * np.pi / 3.
                            )
                            lmbda_nu_val = (
                                2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)
                                - beta_tilde / (3.*alpha_tilde)
                            )
                        
                        lmbda_nu.append(lmbda_nu_val)
                    
                    return np.asarray(lmbda_nu)
                
                def xi_c_func(lmbda_nu, lmbda_c_eq):
                    """Nondimensional chain force

                    This function computes the nondimensional chain 
                    force as a function of the segment stretch and the
                    equilibrium chain stretch
                    """
                    def inv_L_func(lmbda_comp_nu):
                        """Jedynak R[9,2] inverse Langevin approximant

                        This function computes the Jedynak R[9,2]
                        inverse Langevin approximant as a function of
                        the result of the equilibrium chain stretch
                        minus the segment stretch plus one
                        """
                        return (
                            lmbda_comp_nu * (3.-1.00651*lmbda_comp_nu**2
                            -0.962251*lmbda_comp_nu**4+1.47353*lmbda_comp_nu**6
                            -0.48953*lmbda_comp_nu**8)
                            / ((1.-lmbda_comp_nu)*(1.+1.01524*lmbda_comp_nu))
                        )
                    
                    lmbda_comp_nu = lmbda_c_eq - lmbda_nu + 1.
                    
                    return inv_L_func(lmbda_comp_nu)
                
                nu_b = l_nu_eq / l_b_eq
                zeta_nu_char = nu_b * zeta_b_char
                lmbda_c_eq_pade2berg_crit = (
                    lmbda_c_eq_pade2berg_crit_func(kappa_nu)
                )
                lmbda_c_eq_crit = lmbda_c_eq_crit_func(zeta_nu_char, kappa_nu)
                lmbda_c_eq = r_nu / (nu*l_nu_eq) # nm/nm
                lmbda_nu = lmbda_nu_func(lmbda_c_eq)
                xi_c = xi_c_func(lmbda_nu, lmbda_c_eq)
                f_c = xi_c / (beta*l_nu_eq) # 1/(1/(nN*nm))/nm = nN
                
                return f_c
            
            def f_c_func_nu_kappa_nu_fit(r_nu, nu, kappa_nu):
                """Chain force calculated from chain extension, segment
                number, and nondimensional segment stiffness

                This function returns the chain force given chain
                extension, segment number, and nondimensional segment
                stiffness. It is also implicitly a function of the
                segment length, nondimensional characteristic bond
                potential energy scale, and bond length. This function
                is also used to determine the segment length, segment
                number, and nondimensional segment stiffness for a chain
                via a scipy optimize curve_fit analysis
                """
                def lmbda_c_eq_pade2berg_crit_func(kappa_nu):
                    """Pade-to-Bergstrom (P2B) critical equilibrium
                    chain stretch value
                    
                    This function returns Pade-to-Bergstrom (P2B)
                    critical equilibrium chain stretch value as
                    determined via a scipy optimize curve_fit analysis.
                    It is a function of the nondimensional segment
                    stiffness
                    """
                    n = 0.818706900266885
                    b = 0.61757545643322586
                    return 1. / kappa_nu**n + b
                
                def lmbda_c_eq_crit_func(zeta_nu_char, kappa_nu):
                    """Critical equilibrium chain stretch
                    
                    This function computes critical equilibrium chain
                    stretch value as a function of nondimensional
                    characteristic segment potential energy scale and
                    nondimensional segment stiffness
                    """
                    return (
                        1. + np.sqrt(zeta_nu_char/kappa_nu)
                        - np.sqrt(1./(kappa_nu*zeta_nu_char))
                    )
                
                def lmbda_nu_func(lmbda_c_eq):
                    """Segment stretch

                    This function computes the segment stretch as a
                    function of the equilibrium chain stretch.
                    """
                    lmbda_nu = []
                    
                    for lmbda_c_eq_indx in range(len(lmbda_c_eq)):
                        lmbda_c_eq_val = lmbda_c_eq[lmbda_c_eq_indx]
                    
                        # analytical solution (Pade approximant)
                        if lmbda_c_eq_val == 0.:
                            lmbda_nu_val = 1.
                        
                        # Pade approximant
                        elif lmbda_c_eq_val < lmbda_c_eq_pade2berg_crit:
                            alpha_tilde = 1.

                            trm_i  = -3. * (kappa_nu+1.)
                            trm_ii = -(2.*kappa_nu+3.)
                            beta_tilde_nmrtr  = trm_i + lmbda_c_eq_val * trm_ii
                            beta_tilde_dnmntr = kappa_nu + 1.
                            beta_tilde  = beta_tilde_nmrtr / beta_tilde_dnmntr

                            trm_i   = 2. * kappa_nu
                            trm_ii  = 4. * kappa_nu + 6.
                            trm_iii = kappa_nu + 3.
                            gamma_tilde_nmrtr = (
                                trm_i + lmbda_c_eq_val 
                                * (trm_ii+lmbda_c_eq_val*trm_iii)
                            )
                            gamma_tilde_dnmntr = kappa_nu + 1.
                            gamma_tilde = gamma_tilde_nmrtr / gamma_tilde_dnmntr
                            
                            trm_i   = 2.
                            trm_ii  = 2. * kappa_nu
                            trm_iii = kappa_nu + 3.
                            delta_tilde_nmrtr = (
                                trm_i - lmbda_c_eq_val * 
                                (trm_ii+lmbda_c_eq_val*(trm_iii+lmbda_c_eq_val))
                            )
                            delta_tilde_dnmntr = kappa_nu + 1.
                            delta_tilde = delta_tilde_nmrtr / delta_tilde_dnmntr

                            pi_tilde_nmrtr  = (
                                3. * alpha_tilde * gamma_tilde - beta_tilde**2
                            )
                            pi_tilde_dnmntr = 3. * alpha_tilde**2
                            pi_tilde = pi_tilde_nmrtr / pi_tilde_dnmntr

                            rho_tilde_nmrtr = (
                                2. * beta_tilde**3
                                - 9. * alpha_tilde * beta_tilde * gamma_tilde
                                + 27. * alpha_tilde**2 * delta_tilde
                            )
                            rho_tilde_dnmntr = 27. * alpha_tilde**3
                            rho_tilde = rho_tilde_nmrtr / rho_tilde_dnmntr

                            arccos_arg = (
                                3. * rho_tilde / (2.*pi_tilde)
                                * np.sqrt(-3./pi_tilde)
                            )
                            
                            if arccos_arg >= 1. - 1.e-14:
                                arccos_arg =  1. - 1.e-14
                            elif arccos_arg < -1. + 1.e-14:
                                arccos_arg =  -1. + 1.e-14
                            
                            cos_arg = (
                                1. / 3. * np.arccos(arccos_arg)
                                - 2. * np.pi / 3.
                            )
                            lmbda_nu_val = (
                                2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)
                                - beta_tilde / (3.*alpha_tilde)
                            )
                        
                        # Bergstrom approximate
                        elif lmbda_c_eq_val <= lmbda_c_eq_crit:
                            sqrt_arg = (
                                lmbda_c_eq_val**2 - 2. * lmbda_c_eq_val + 1.
                                + 4. / kappa_nu
                            )
                            lmbda_nu_val = (
                                (lmbda_c_eq_val+1.+np.sqrt(sqrt_arg)) / 2.
                            )
                        
                        # Bergstrom approximate
                        else:
                            alpha_tilde = 1.
                            beta_tilde  = -3.
                            gamma_tilde = 3. - zeta_nu_char**2 / kappa_nu
                            delta_tilde = (
                                zeta_nu_char**2 / kappa_nu * lmbda_c_eq_val - 1.
                            )

                            pi_tilde_nmrtr = (
                                3. * alpha_tilde * gamma_tilde - beta_tilde**2
                            )
                            pi_tilde_dnmntr = 3. * alpha_tilde**2
                            pi_tilde = pi_tilde_nmrtr / pi_tilde_dnmntr

                            rho_tilde_nmrtr = (
                                2. * beta_tilde**3
                                - 9. * alpha_tilde * beta_tilde * gamma_tilde
                                + 27. * alpha_tilde**2 * delta_tilde
                            )
                            rho_tilde_dnmntr = 27. * alpha_tilde**3
                            rho_tilde = rho_tilde_nmrtr / rho_tilde_dnmntr

                            arccos_arg = (
                                3. * rho_tilde / (2.*pi_tilde)
                                * np.sqrt(-3./pi_tilde)
                            )
                            
                            if arccos_arg >= 1. - 1.e-14:
                                arccos_arg =  1. - 1.e-14
                            elif arccos_arg < -1. + 1.e-14:
                                arccos_arg =  -1. + 1.e-14
                            
                            cos_arg = (
                                1. / 3. * np.arccos(arccos_arg)
                                - 2. * np.pi / 3.
                            )
                            lmbda_nu_val = (
                                2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)
                                - beta_tilde / (3.*alpha_tilde)
                            )
                        
                        lmbda_nu.append(lmbda_nu_val)
                    
                    return np.asarray(lmbda_nu)
                
                def xi_c_func(lmbda_nu, lmbda_c_eq):
                    """Nondimensional chain force

                    This function computes the nondimensional chain 
                    force as a function of the segment stretch and the
                    equilibrium chain stretch
                    """
                    def inv_L_func(lmbda_comp_nu):
                        """Jedynak R[9,2] inverse Langevin approximant

                        This function computes the Jedynak R[9,2]
                        inverse Langevin approximant as a function of
                        the result of the equilibrium chain stretch
                        minus the segment stretch plus one
                        """
                        return (
                            lmbda_comp_nu * (3.-1.00651*lmbda_comp_nu**2
                            -0.962251*lmbda_comp_nu**4+1.47353*lmbda_comp_nu**6
                            -0.48953*lmbda_comp_nu**8)
                            / ((1.-lmbda_comp_nu)*(1.+1.01524*lmbda_comp_nu))
                        )
                    
                    lmbda_comp_nu = lmbda_c_eq - lmbda_nu + 1.
                    
                    return inv_L_func(lmbda_comp_nu)
                
                nu_b = l_nu_eq / l_b_eq
                zeta_nu_char = nu_b * zeta_b_char
                lmbda_c_eq_pade2berg_crit = (
                    lmbda_c_eq_pade2berg_crit_func(kappa_nu)
                )
                lmbda_c_eq_crit = lmbda_c_eq_crit_func(zeta_nu_char, kappa_nu)
                lmbda_c_eq = r_nu / (nu*l_nu_eq) # nm/nm
                lmbda_nu = lmbda_nu_func(lmbda_c_eq)
                xi_c = xi_c_func(lmbda_nu, lmbda_c_eq)
                f_c = xi_c / (beta*l_nu_eq) # 1/(1/(nN*nm))/nm = nN
                
                return f_c
            
            def f_c_func_kappa_nu_fit(r_nu, kappa_nu):
                """Chain force calculated from chain extension and
                nondimensional segment stiffness

                This function returns the chain force given chain 
                extension and nondimensional segment stiffness. It is
                also implicitly a function of the segment length,
                segment number, nondimensional characteristic bond
                potential energy scale, and bond length. This function
                is also used to determine the segment length, segment
                number, and nondimensional segment stiffness for a chain
                via a scipy optimize curve_fit analysis
                """
                def lmbda_c_eq_pade2berg_crit_func(kappa_nu):
                    """Pade-to-Bergstrom (P2B) critical equilibrium
                    chain stretch value
                    
                    This function returns Pade-to-Bergstrom (P2B)
                    critical equilibrium chain stretch value as
                    determined via a scipy optimize curve_fit analysis.
                    It is a function of the nondimensional segment
                    stiffness
                    """
                    n = 0.818706900266885
                    b = 0.61757545643322586
                    return 1. / kappa_nu**n + b
                
                def lmbda_c_eq_crit_func(zeta_nu_char, kappa_nu):
                    """Critical equilibrium chain stretch
                    
                    This function computes critical equilibrium chain
                    stretch value as a function of nondimensional
                    characteristic segment potential energy scale and
                    nondimensional segment stiffness
                    """
                    return (
                        1. + np.sqrt(zeta_nu_char/kappa_nu)
                        - np.sqrt(1./(kappa_nu*zeta_nu_char))
                    )
                
                def lmbda_nu_func(lmbda_c_eq):
                    """Segment stretch

                    This function computes the segment stretch as a
                    function of the equilibrium chain stretch.
                    """
                    lmbda_nu = []
                    
                    for lmbda_c_eq_indx in range(len(lmbda_c_eq)):
                        lmbda_c_eq_val = lmbda_c_eq[lmbda_c_eq_indx]
                    
                        # analytical solution (Pade approximant)
                        if lmbda_c_eq_val == 0.:
                            lmbda_nu_val = 1.
                        
                        # Pade approximant
                        elif lmbda_c_eq_val < lmbda_c_eq_pade2berg_crit:
                            alpha_tilde = 1.

                            trm_i  = -3. * (kappa_nu+1.)
                            trm_ii = -(2.*kappa_nu+3.)
                            beta_tilde_nmrtr  = trm_i + lmbda_c_eq_val * trm_ii
                            beta_tilde_dnmntr = kappa_nu + 1.
                            beta_tilde  = beta_tilde_nmrtr / beta_tilde_dnmntr

                            trm_i   = 2. * kappa_nu
                            trm_ii  = 4. * kappa_nu + 6.
                            trm_iii = kappa_nu + 3.
                            gamma_tilde_nmrtr = (
                                trm_i + lmbda_c_eq_val 
                                * (trm_ii+lmbda_c_eq_val*trm_iii)
                            )
                            gamma_tilde_dnmntr = kappa_nu + 1.
                            gamma_tilde = gamma_tilde_nmrtr / gamma_tilde_dnmntr
                            
                            trm_i   = 2.
                            trm_ii  = 2. * kappa_nu
                            trm_iii = kappa_nu + 3.
                            delta_tilde_nmrtr = (
                                trm_i - lmbda_c_eq_val * 
                                (trm_ii+lmbda_c_eq_val*(trm_iii+lmbda_c_eq_val))
                            )
                            delta_tilde_dnmntr = kappa_nu + 1.
                            delta_tilde = delta_tilde_nmrtr / delta_tilde_dnmntr

                            pi_tilde_nmrtr  = (
                                3. * alpha_tilde * gamma_tilde - beta_tilde**2
                            )
                            pi_tilde_dnmntr = 3. * alpha_tilde**2
                            pi_tilde = pi_tilde_nmrtr / pi_tilde_dnmntr

                            rho_tilde_nmrtr = (
                                2. * beta_tilde**3
                                - 9. * alpha_tilde * beta_tilde * gamma_tilde
                                + 27. * alpha_tilde**2 * delta_tilde
                            )
                            rho_tilde_dnmntr = 27. * alpha_tilde**3
                            rho_tilde = rho_tilde_nmrtr / rho_tilde_dnmntr

                            arccos_arg = (
                                3. * rho_tilde / (2.*pi_tilde)
                                * np.sqrt(-3./pi_tilde)
                            )
                            
                            if arccos_arg >= 1. - 1.e-14:
                                arccos_arg =  1. - 1.e-14
                            elif arccos_arg < -1. + 1.e-14:
                                arccos_arg =  -1. + 1.e-14
                            
                            cos_arg = (
                                1. / 3. * np.arccos(arccos_arg)
                                - 2. * np.pi / 3.
                            )
                            lmbda_nu_val = (
                                2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)
                                - beta_tilde / (3.*alpha_tilde)
                            )
                        
                        # Bergstrom approximate
                        elif lmbda_c_eq_val <= lmbda_c_eq_crit:
                            sqrt_arg = (
                                lmbda_c_eq_val**2 - 2. * lmbda_c_eq_val + 1.
                                + 4. / kappa_nu
                            )
                            lmbda_nu_val = (
                                (lmbda_c_eq_val+1.+np.sqrt(sqrt_arg)) / 2.
                            )
                        
                        # Bergstrom approximate
                        else:
                            alpha_tilde = 1.
                            beta_tilde  = -3.
                            gamma_tilde = 3. - zeta_nu_char**2 / kappa_nu
                            delta_tilde = (
                                zeta_nu_char**2 / kappa_nu * lmbda_c_eq_val - 1.
                            )

                            pi_tilde_nmrtr = (
                                3. * alpha_tilde * gamma_tilde - beta_tilde**2
                            )
                            pi_tilde_dnmntr = 3. * alpha_tilde**2
                            pi_tilde = pi_tilde_nmrtr / pi_tilde_dnmntr

                            rho_tilde_nmrtr = (
                                2. * beta_tilde**3
                                - 9. * alpha_tilde * beta_tilde * gamma_tilde
                                + 27. * alpha_tilde**2 * delta_tilde
                            )
                            rho_tilde_dnmntr = 27. * alpha_tilde**3
                            rho_tilde = rho_tilde_nmrtr / rho_tilde_dnmntr

                            arccos_arg = (
                                3. * rho_tilde / (2.*pi_tilde)
                                * np.sqrt(-3./pi_tilde)
                            )
                            
                            if arccos_arg >= 1. - 1.e-14:
                                arccos_arg =  1. - 1.e-14
                            elif arccos_arg < -1. + 1.e-14:
                                arccos_arg =  -1. + 1.e-14
                            
                            cos_arg = (
                                1. / 3. * np.arccos(arccos_arg)
                                - 2. * np.pi / 3.
                            )
                            lmbda_nu_val = (
                                2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)
                                - beta_tilde / (3.*alpha_tilde)
                            )
                        
                        lmbda_nu.append(lmbda_nu_val)
                    
                    return np.asarray(lmbda_nu)
                
                def xi_c_func(lmbda_nu, lmbda_c_eq):
                    """Nondimensional chain force

                    This function computes the nondimensional chain 
                    force as a function of the segment stretch and the
                    equilibrium chain stretch
                    """
                    def inv_L_func(lmbda_comp_nu):
                        """Jedynak R[9,2] inverse Langevin approximant

                        This function computes the Jedynak R[9,2]
                        inverse Langevin approximant as a function of
                        the result of the equilibrium chain stretch
                        minus the segment stretch plus one
                        """
                        return (
                            lmbda_comp_nu * (3.-1.00651*lmbda_comp_nu**2
                            -0.962251*lmbda_comp_nu**4+1.47353*lmbda_comp_nu**6
                            -0.48953*lmbda_comp_nu**8)
                            / ((1.-lmbda_comp_nu)*(1.+1.01524*lmbda_comp_nu))
                        )
                    
                    lmbda_comp_nu = lmbda_c_eq - lmbda_nu + 1.
                    
                    return inv_L_func(lmbda_comp_nu)
                
                nu_b = l_nu_eq / l_b_eq
                zeta_nu_char = nu_b * zeta_b_char
                lmbda_c_eq_pade2berg_crit = (
                    lmbda_c_eq_pade2berg_crit_func(kappa_nu)
                )
                lmbda_c_eq_crit = lmbda_c_eq_crit_func(zeta_nu_char, kappa_nu)
                lmbda_c_eq = r_nu / (nu*l_nu_eq) # nm/nm
                lmbda_nu = lmbda_nu_func(lmbda_c_eq)
                xi_c = xi_c_func(lmbda_nu, lmbda_c_eq)
                f_c = xi_c / (beta*l_nu_eq) # 1/(1/(nN*nm))/nm = nN
                
                return f_c
            
            r_nu_fit_max = r_nu[-1]
            
            r_nu_fit_num_steps = (
                int(np.around(
                    (r_nu_fit_max-cp.r_nu_fit_min)/cp.r_nu_fit_inc))
                + 1
            )
            r_nu_fit = (
                np.linspace(cp.r_nu_fit_min, r_nu_fit_max, r_nu_fit_num_steps)
             ) # nm
            
            l_nu_eq_min = l_b_eq # nm
            
            popt, pcov = (
                optimize.curve_fit(
                    f_c_func_l_nu_eq_nu_kappa_nu_fit, r_nu, f_c,
                    bounds=((l_nu_eq_min, cp.nu_min, cp.kappa_nu_min),
                            (cp.l_nu_eq_max, cp.nu_max, cp.kappa_nu_max)))
            )
            
            l_nu_eq  = popt[0]
            nu       = popt[1]
            kappa_nu = popt[2]
            
            l_nu_eq_init  = l_nu_eq
            nu_init       = nu
            kappa_nu_init = kappa_nu
            
            nu_b = l_nu_eq/l_b_eq
            
            print('l_nu_eq = {}'.format(l_nu_eq))
            print('nu = {}'.format(nu))
            print('kappa_nu = {}'.format(kappa_nu))
            
            f_c_fit = (
                f_c_func_l_nu_eq_nu_kappa_nu_fit(
                    r_nu_fit, l_nu_eq, nu, kappa_nu)
            )
            r_nu_fit_intgr_nu = np.copy(r_nu_fit)
            
            nu_b_floor = np.floor(nu_b)
            nu_b_ceil  = np.ceil(nu_b)
            
            l_nu_eq_floor = nu_b_floor*l_b_eq
            l_nu_eq_ceil  = nu_b_ceil*l_b_eq
            
            l_nu_eq    = l_nu_eq_floor
            popt, pcov = (
                optimize.curve_fit(
                    f_c_func_nu_kappa_nu_fit, r_nu, f_c,
                    bounds=((cp.nu_min, cp.kappa_nu_min),
                            (cp.nu_max, cp.kappa_nu_max)))
            )

            nu_floor       = popt[0]
            kappa_nu_floor = popt[1]
            residual_floor = (
                np.linalg.norm(f_c-f_c_func_l_nu_eq_nu_kappa_nu_fit(
                    r_nu, l_nu_eq_floor, nu_floor, kappa_nu_floor))
            )
            
            l_nu_eq    = l_nu_eq_ceil
            popt, pcov = (
                optimize.curve_fit(
                    f_c_func_nu_kappa_nu_fit, r_nu, f_c,
                    bounds=((cp.nu_min, cp.kappa_nu_min),
                            (cp.nu_max, cp.kappa_nu_max)))
            )

            nu_ceil       = popt[0]
            kappa_nu_ceil = popt[1]
            residual_ceil = (
                np.linalg.norm(f_c-f_c_func_l_nu_eq_nu_kappa_nu_fit(
                    r_nu, l_nu_eq_ceil, nu_ceil, kappa_nu_ceil))
            )
            
            if residual_floor <= residual_ceil:
                l_nu_eq = l_nu_eq_floor
                nu      = nu_floor
            else:
                l_nu_eq = l_nu_eq_ceil
                nu      = nu_ceil
            
            print('\nl_nu_eq = {}'.format(l_nu_eq))
            print('nu = {}'.format(nu))
            
            nu_floor = np.floor(nu)
            nu_ceil  = np.ceil(nu)
            
            nu         = nu_floor
            popt, pcov = (
                optimize.curve_fit(
                    f_c_func_kappa_nu_fit, r_nu, f_c,
                    bounds=(cp.kappa_nu_min,
                            cp.kappa_nu_max))
            )

            kappa_nu_floor = popt[0]
            residual_floor = (
                np.linalg.norm(f_c-f_c_func_l_nu_eq_nu_kappa_nu_fit(
                    r_nu, l_nu_eq, nu_floor, kappa_nu_floor))
            )
            
            nu         = nu_ceil
            popt, pcov = (
                optimize.curve_fit(
                    f_c_func_kappa_nu_fit, r_nu, f_c,
                    bounds=(cp.kappa_nu_min,
                            cp.kappa_nu_max))
            )

            kappa_nu_ceil = popt[0]
            residual_ceil = (
                np.linalg.norm(f_c-f_c_func_l_nu_eq_nu_kappa_nu_fit(
                    r_nu, l_nu_eq, nu_ceil, kappa_nu_ceil))
            )
            
            if residual_floor <= residual_ceil:
                nu       = nu_floor
                kappa_nu = kappa_nu_floor
            else:
                nu       = nu_ceil
                kappa_nu = kappa_nu_ceil
            
            print('\nnu = {}'.format(nu))
            print('kappa_nu = {}'.format(kappa_nu))
            
            l_nu_eq_intgr_nu  = l_nu_eq
            intgr_nu          = nu
            kappa_nu_intgr_nu = kappa_nu
            
            f_c_fit_intgr_nu = (
                f_c_func_l_nu_eq_nu_kappa_nu_fit(
                    r_nu_fit_intgr_nu, l_nu_eq_intgr_nu,
                    intgr_nu, kappa_nu_intgr_nu)
            )
            
            l_nu_eq  = l_nu_eq_init
            nu       = nu_init
            kappa_nu = kappa_nu_init
            
            return (
                r_nu_fit, r_nu_fit_intgr_nu, f_c_fit, f_c_fit_intgr_nu, l_nu_eq,
                l_nu_eq_intgr_nu, nu, intgr_nu, kappa_nu, kappa_nu_intgr_nu
            )

        k_B     = constants.value(u'Boltzmann constant') # J/K
        h       = constants.value(u'Planck constant') # J/Hz
        hbar    = h / (2*np.pi) # J*sec
        beta    = 1. / (k_B*self.T) # 1/J
        omega_0 = 1. / (beta*hbar) # J/(J*sec) = 1/sec

        beta = beta / (1e9*1e9) # 1/J = 1/(N*m) -> 1/(nN*m) -> 1/(nN*nm)

        cp = self.parameters.characterizer

        polymer_type = cp.paper_authors2polymer_type_dict[self.paper_authors]
        chain_backbone_bond_type = (
            cp.polymer_type_label2chain_backbone_bond_type_dict[polymer_type]
        )
        self.data_file_prefix = (
            self.paper_authors + '-' + polymer_type + '-'
            + chain_backbone_bond_type + '-' + self.chain
        )

        f_c_max = (
            cp.chain_backbone_bond_type2f_c_max_dict[chain_backbone_bond_type]
        ) # nN
        zeta_b_char = (
            cp.chain_backbone_bond_type2zeta_b_char_dict[chain_backbone_bond_type]
        )
        l_b_eq = (
            cp.chain_backbone_bond_type2l_b_eq_dict[chain_backbone_bond_type]
        ) # nm

        # Curve fit single chain AFM tensile test results;
        # nm, nN respectively
        r_nu, f_c = (
            chain_data_file_load_func(
                self.data_file_prefix, cp.chain_data_directory)
        )

        # nm, nN, nm, unitless, unitless, respectively
        (r_nu_fit, r_nu_fit_intgr_nu,
        f_c_fit, f_c_fit_intgr_nu,
        l_nu_eq, l_nu_eq_intgr_nu, 
        nu, intgr_nu, 
        kappa_nu, kappa_nu_intgr_nu) = l_nu_eq_nu_kappa_nu_fit_func(r_nu, f_c)

        lmbda_c_eq          = r_nu / (nu*l_nu_eq) # nm/nm
        lmbda_c_eq_fit      = r_nu_fit/ ( nu*l_nu_eq) # nm/nm
        lmbda_c_eq_intgr_nu = r_nu / (intgr_nu*l_nu_eq_intgr_nu) # nm/nm
        lmbda_c_eq_fit_intgr_nu = (
            r_nu_fit_intgr_nu / (intgr_nu*l_nu_eq_intgr_nu)
        ) # nm/nm

        xi_c          = f_c * beta * l_nu_eq # nN*nm/(nN*nm)
        xi_c_fit      = f_c_fit * beta * l_nu_eq # nN*nm/(nN*nm)
        xi_c_intgr_nu = f_c * beta * l_nu_eq_intgr_nu # nN*nm/(nN*nm)
        xi_c_fit_intgr_nu = (
            f_c_fit_intgr_nu * beta * l_nu_eq_intgr_nu
        ) # nN*nm/(nN*nm)

        nu_b          = l_nu_eq / l_b_eq
        nu_b_intgr_nu = l_nu_eq_intgr_nu / l_b_eq

        zeta_nu_char          = nu_b * zeta_b_char
        zeta_nu_char_intgr_nu = nu_b_intgr_nu * zeta_b_char

        kappa_b          = kappa_nu / nu_b
        kappa_b_intgr_nu = kappa_nu_intgr_nu / nu_b_intgr_nu

        xi_c_max = f_c_max * beta * l_nu_eq_intgr_nu

        self.r_nu              = r_nu
        self.r_nu_fit          = r_nu_fit
        self.r_nu_fit_intgr_nu = r_nu_fit_intgr_nu

        self.lmbda_c_eq              = lmbda_c_eq
        self.lmbda_c_eq_fit          = lmbda_c_eq_fit
        self.lmbda_c_eq_intgr_nu     = lmbda_c_eq_intgr_nu
        self.lmbda_c_eq_fit_intgr_nu = lmbda_c_eq_fit_intgr_nu

        self.f_c              = f_c
        self.f_c_fit          = f_c_fit
        self.f_c_fit_intgr_nu = f_c_fit_intgr_nu

        self.xi_c              = xi_c
        self.xi_c_fit          = xi_c_fit
        self.xi_c_intgr_nu     = xi_c_intgr_nu
        self.xi_c_fit_intgr_nu = xi_c_fit_intgr_nu
        
        self.l_b_eq           = l_b_eq
        self.l_nu_eq          = l_nu_eq
        self.l_nu_eq_intgr_nu = l_nu_eq_intgr_nu

        self.nu_b          = nu_b
        self.nu_b_intgr_nu = nu_b_intgr_nu

        self.nu       = nu
        self.intgr_nu = intgr_nu
        
        self.zeta_b_char           = zeta_b_char
        self.zeta_nu_char          = zeta_nu_char
        self.zeta_nu_char_intgr_nu = zeta_nu_char_intgr_nu

        self.kappa_b          = kappa_b
        self.kappa_b_intgr_nu = kappa_b_intgr_nu

        self.kappa_nu          = kappa_nu
        self.kappa_nu_intgr_nu = kappa_nu_intgr_nu

        self.xi_c_max = xi_c_max

        # Initialize a rate-dependent CompositeuFJC which matches the
        # chain used in the AFM tensile test where the bond and segment
        # numbers in the chain are strictly integer values
        rate_dependent_single_chain = (
            RateDependentScissionCompositeuFJC(
                nu=intgr_nu, zeta_nu_char=zeta_nu_char_intgr_nu,
                kappa_nu=kappa_nu_intgr_nu, omega_0=omega_0)
        )

        f_c_crit = (
            rate_dependent_single_chain.xi_c_crit / (beta*l_nu_eq_intgr_nu)
        ) # (nN*nm)/nm = nN

        # Define the applied chain force values to calculate over
        f_c_steps = np.linspace(0, f_c_crit, cp.f_c_num_steps) # nN

        t_steps___f_c_dot_chunk = [
            0. for f_c_dot_indx in range(len(cp.f_c_dot_list))
        ]
        xi_c___f_c_dot_chunk = [
            0. for f_c_dot_indx in range(len(cp.f_c_dot_list))
        ]
        lmbda_nu___f_c_dot_chunk = [
            0. for f_c_dot_indx in range(len(cp.f_c_dot_list))
        ]
        lmbda_c_eq___f_c_dot_chunk = [
            0. for f_c_dot_indx in range(len(cp.f_c_dot_list))
        ]
        gamma_c___f_c_dot_chunk = [
            0. for f_c_dot_indx in range(len(cp.f_c_dot_list))
        ]
        overline_epsilon_cnu_diss_hat___f_c_dot_chunk = [
            0. for f_c_dot_indx in range(len(cp.f_c_dot_list))
        ]

        for f_c_dot_indx in range(len(cp.f_c_dot_list)):
            # Define the timestamp values to calculate over
            f_c_dot = cp.f_c_dot_list[f_c_dot_indx] # nN/sec
            t_steps = f_c_steps / f_c_dot # nN/(nN/sec) = sec

            # Make arrays to allocate results
            xi_c                      = []
            lmbda_nu                  = []
            lmbda_c_eq                = []
            p_nu_sci_hat              = []
            p_nu_sci_hat_cmltv_intgrl = []
            gamma_c                   = []
            epsilon_cnu_diss_hat      = []

            # initialization
            p_nu_sci_hat_cmltv_intgrl_val = 0.
            gamma_c_val                   = 0.
            epsilon_cnu_diss_hat_val      = 0.

            # Calculate results through applied chain force values
            for f_c_indx in range(cp.f_c_num_steps):
                t_val = t_steps[f_c_indx]
                xi_c_val = (
                    f_c_steps[f_c_indx] * beta * l_nu_eq_intgr_nu
                ) # nN*nm/(nN*nm)
                lmbda_nu_val = (
                    rate_dependent_single_chain.lmbda_nu_xi_c_hat_func(xi_c_val)
                )
                lmbda_c_eq_val = (
                    rate_dependent_single_chain.lmbda_c_eq_func(lmbda_nu_val)
                )
                p_nu_sci_hat_val = (
                    rate_dependent_single_chain.p_nu_sci_hat_func(lmbda_nu_val)
                )
                epsilon_cnu_sci_hat_val = (
                    rate_dependent_single_chain.epsilon_cnu_sci_hat_func(
                        lmbda_nu_val)
                )

                if f_c_indx == 0:
                    pass
                else:
                    p_nu_sci_hat_cmltv_intgrl_val = (
                        rate_dependent_single_chain.p_nu_sci_hat_cmltv_intgrl_func(
                            p_nu_sci_hat_val, t_val, p_nu_sci_hat[f_c_indx-1],
                            t_steps[f_c_indx-1],
                            p_nu_sci_hat_cmltv_intgrl[f_c_indx-1])
                    )
                    gamma_c_val = (
                        rate_dependent_single_chain.gamma_c_func(
                            p_nu_sci_hat_cmltv_intgrl_val)
                    )
                    epsilon_cnu_diss_hat_val = (
                        rate_dependent_single_chain.epsilon_cnu_diss_hat_func(
                            p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
                            epsilon_cnu_sci_hat_val, t_val, t_steps[f_c_indx-1],
                            epsilon_cnu_diss_hat[f_c_indx-1])
                    )

                xi_c.append(xi_c_val)
                lmbda_nu.append(lmbda_nu_val)
                lmbda_c_eq.append(lmbda_c_eq_val)
                p_nu_sci_hat.append(p_nu_sci_hat_val)
                p_nu_sci_hat_cmltv_intgrl.append(p_nu_sci_hat_cmltv_intgrl_val)
                gamma_c.append(gamma_c_val)
                epsilon_cnu_diss_hat.append(epsilon_cnu_diss_hat_val)
            
            overline_epsilon_cnu_diss_hat = [
                epsilon_cnu_diss_hat_val/rate_dependent_single_chain.zeta_nu_char
                for epsilon_cnu_diss_hat_val in epsilon_cnu_diss_hat
            ]

            t_steps___f_c_dot_chunk[f_c_dot_indx] = t_steps
            xi_c___f_c_dot_chunk[f_c_dot_indx] = xi_c
            lmbda_nu___f_c_dot_chunk[f_c_dot_indx] = lmbda_nu
            lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx] = lmbda_c_eq
            gamma_c___f_c_dot_chunk[f_c_dot_indx] = gamma_c
            overline_epsilon_cnu_diss_hat___f_c_dot_chunk[f_c_dot_indx] = (
                overline_epsilon_cnu_diss_hat
            )
        
        self.rate_dependent_t_steps___f_c_dot_chunk = t_steps___f_c_dot_chunk
        self.rate_dependent_xi_c___f_c_dot_chunk = xi_c___f_c_dot_chunk
        self.rate_dependent_lmbda_nu___f_c_dot_chunk = lmbda_nu___f_c_dot_chunk
        self.rate_dependent_lmbda_c_eq___f_c_dot_chunk = (
            lmbda_c_eq___f_c_dot_chunk
        )
        self.rate_dependent_gamma_c___f_c_dot_chunk = gamma_c___f_c_dot_chunk
        self.rate_dependent_overline_epsilon_cnu_diss_hat___f_c_dot_chunk = (
            overline_epsilon_cnu_diss_hat___f_c_dot_chunk
        )

        # Initialize a rate-independent CompositeuFJC which matches the
        # chain used in the AFM tensile test where the bond and segment
        # numbers in the chain are strictly integer values
        rate_independent_single_chain = (
            RateIndependentScissionCompositeuFJC(
                nu=intgr_nu, zeta_nu_char=zeta_nu_char_intgr_nu,
                kappa_nu=kappa_nu_intgr_nu)
        )

        f_c_crit = (
            rate_independent_single_chain.xi_c_crit / (beta*l_nu_eq_intgr_nu)
        ) # (nN*nm)/nm = nN

        # Define the applied chain force values to calculate over
        f_c_steps = np.linspace(0, f_c_crit, cp.f_c_num_steps) # nN

        xi_c                 = []
        lmbda_nu             = []
        lmbda_nu_max         = []
        lmbda_c_eq           = []
        p_c_sci_hat          = []
        epsilon_cnu_sci_hat  = []
        epsilon_cnu_diss_hat = []

        # initialization
        lmbda_nu_max_val         = 0.
        epsilon_cnu_diss_hat_val = 0.

        # Calculate results through applied chain force values
        for f_c_indx in range(cp.f_c_num_steps):
            xi_c_val = (
                f_c_steps[f_c_indx] * beta * l_nu_eq_intgr_nu
            ) # nN*nm/(nN*nm)
            lmbda_nu_val = (
                rate_independent_single_chain.lmbda_nu_xi_c_hat_func(xi_c_val)
            )
            lmbda_nu_max_val = max([lmbda_nu_max_val, lmbda_nu_val])
            lmbda_c_eq_val = (
                rate_independent_single_chain.lmbda_c_eq_func(lmbda_nu_val)
            )
            p_c_sci_hat_val = (
                rate_independent_single_chain.p_c_sci_hat_func(lmbda_nu_val)
            )
            epsilon_cnu_sci_hat_val = (
                rate_independent_single_chain.epsilon_cnu_sci_hat_func(
                    lmbda_nu_val)
            )

            if f_c_indx == 0:
                pass
            else:
                epsilon_cnu_diss_hat_val = (
                    rate_independent_single_chain.epsilon_cnu_diss_hat_func(
                        lmbda_nu_max_val, lmbda_nu_max[f_c_indx-1],
                        lmbda_nu_val, lmbda_nu[f_c_indx-1],
                        epsilon_cnu_diss_hat[f_c_indx-1])
                )
            
            xi_c.append(xi_c_val)
            lmbda_nu.append(lmbda_nu_val)
            lmbda_nu_max.append(lmbda_nu_max_val)
            lmbda_c_eq.append(lmbda_c_eq_val)
            p_c_sci_hat.append(p_c_sci_hat_val)
            epsilon_cnu_sci_hat.append(epsilon_cnu_sci_hat_val)
            epsilon_cnu_diss_hat.append(epsilon_cnu_diss_hat_val)

        overline_epsilon_cnu_sci_hat  = [
            epsilon_cnu_sci_hat_val/rate_independent_single_chain.zeta_nu_char
            for epsilon_cnu_sci_hat_val in epsilon_cnu_sci_hat
        ]
        overline_epsilon_cnu_diss_hat = [
            epsilon_cnu_diss_hat_val/rate_independent_single_chain.zeta_nu_char
            for epsilon_cnu_diss_hat_val in epsilon_cnu_diss_hat
        ]

        self.rate_independent_xi_c = xi_c
        self.rate_independent_lmbda_nu = lmbda_nu
        self.rate_independent_lmbda_c_eq = lmbda_c_eq
        self.rate_independent_p_c_sci_hat = p_c_sci_hat
        self.rate_independent_overline_epsilon_cnu_sci_hat  = (
            overline_epsilon_cnu_sci_hat
        )
        self.rate_independent_overline_epsilon_cnu_diss_hat = (
            overline_epsilon_cnu_diss_hat
        )

    def finalization(self):
        """Define finalization analysis"""
        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing

        # print and save numerical results in text files
        print("\nl_b_eq = {}".format(self.l_b_eq))
        print("l_nu_eq = {}".format(self.l_nu_eq))
        print("l_nu_eq_intgr_nu = {}".format(self.l_nu_eq_intgr_nu))
        print("nu_b = {}".format(self.nu_b))
        print("nu_b_intgr_nu = {}".format(self.nu_b_intgr_nu))
        print("nu = {}".format(self.nu))
        print("intgr_nu = {}".format(self.intgr_nu))
        print("zeta_b_char = {}".format(self.zeta_b_char))
        print("zeta_nu_char = {}".format(self.zeta_nu_char))
        print("zeta_nu_char_intgr_nu = {}".format(self.zeta_nu_char_intgr_nu))
        print("kappa_b = {}".format(self.kappa_b))
        print("kappa_b_intgr_nu = {}".format(self.kappa_b_intgr_nu))
        print("kappa_nu = {}".format(self.kappa_nu))
        print("kappa_nu_intgr_nu = {}".format(self. kappa_nu_intgr_nu))

        np.savetxt(
            self.savedir+self.data_file_prefix+'-composite-uFJC-curve-fit-l_b_eq'+'.txt',
            [self.l_b_eq])
        np.savetxt(
            self.savedir+self.data_file_prefix+'-composite-uFJC-curve-fit-l_nu_eq'+'.txt',
            [self.l_nu_eq])
        np.savetxt(
            self.savedir+self.data_file_prefix+'-composite-uFJC-curve-fit-l_nu_eq_intgr_nu'+'.txt',
            [self.l_nu_eq_intgr_nu])
        np.savetxt(
            self.savedir+self.data_file_prefix+'-composite-uFJC-curve-fit-nu_b'+'.txt',
            [self.nu_b])
        np.savetxt(
            self.savedir+self.data_file_prefix+'-composite-uFJC-curve-fit-nu_b_intgr_nu'+'.txt',
            [self.nu_b_intgr_nu])
        np.savetxt(
            self.savedir+self.data_file_prefix+'-composite-uFJC-curve-fit-nu'+'.txt',
            [self.nu])
        np.savetxt(
            self.savedir+self.data_file_prefix+'-composite-uFJC-curve-fit-intgr_nu'+'.txt',
            [self.intgr_nu])
        np.savetxt(
            self.savedir+self.data_file_prefix+'-composite-uFJC-curve-fit-zeta_b_char'+'.txt',
            [self.zeta_b_char])
        np.savetxt(
            self.savedir+self.data_file_prefix+'-composite-uFJC-curve-fit-zeta_nu_char'+'.txt',
            [self.zeta_nu_char])
        np.savetxt(
            self.savedir+self.data_file_prefix+'-composite-uFJC-curve-fit-zeta_nu_char_intgr_nu'+'.txt',
            [self.zeta_nu_char_intgr_nu])
        np.savetxt(
            self.savedir+self.data_file_prefix+'-composite-uFJC-curve-fit-kappa_b'+'.txt',
            [self.kappa_b])
        np.savetxt(
            self.savedir+self.data_file_prefix+'-composite-uFJC-curve-fit-kappa_b_intgr_nu'+'.txt',
            [self.kappa_b_intgr_nu])
        np.savetxt(
            self.savedir+self.data_file_prefix+'-composite-uFJC-curve-fit-kappa_nu'+'.txt',
            [self.kappa_nu])
        np.savetxt(
            self.savedir+self.data_file_prefix+'-composite-uFJC-curve-fit-kappa_nu_intgr_nu'+'.txt',
            [self.kappa_nu_intgr_nu])

        # plot results
        latex_formatting_figure(ppp)

        # plot curve fit results
        fig = plt.figure()
        plt.scatter(
            self.r_nu, self.f_c, color='blue', marker='o', alpha=1,
            linewidth=2.5,
            label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        plt.plot(
            self.r_nu_fit, self.f_c_fit, linestyle='--', color='red', alpha=1,
            linewidth=2.5, label=r'$u\textrm{FJC model fit}$')
        plt.legend(loc='best', fontsize=18)
        plt.grid(True, alpha=0.25)
        plt.xlim(
            cp.paper_authors_chain2xlim_chain_mechanical_response_plot[self.paper_authors][self.chain])
        plt.xticks(fontsize=20)
        plt.ylim(
            cp.paper_authors_chain2ylim_chain_mechanical_response_plot[self.paper_authors][self.chain])
        plt.yticks(fontsize=20)
        save_current_figure(
            self.savedir, r'$r_{\nu}~(nm)$', 30, r'$f_c~(nN)$', 30,
            self.data_file_prefix+"-f_c-vs-r_nu-composite-uFJC-curve-fit")

        fig = plt.figure()
        plt.scatter(
            self.r_nu, self.f_c, color='blue', marker='o', alpha=1,
            linewidth=2.5,
            label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        plt.plot(
            self.r_nu_fit_intgr_nu, self.f_c_fit_intgr_nu, linestyle='--',
            color='red', alpha=1, linewidth=2.5,
            label=r'$u\textrm{FJC model fit}$')
        plt.legend(loc='best', fontsize=18)
        plt.grid(True, alpha=0.25)
        plt.xlim(
            cp.paper_authors_chain2xlim_chain_mechanical_response_plot[self.paper_authors][self.chain])
        plt.xticks(fontsize=20)
        plt.ylim(
            cp.paper_authors_chain2ylim_chain_mechanical_response_plot[self.paper_authors][self.chain])
        plt.yticks(fontsize=20)
        save_current_figure(
            self.savedir, r'$r_{\nu}~(nm)$', 30, r'$f_c~(nN)$', 30,
            self.data_file_prefix+"-intgr_nu-f_c-vs-r_nu-composite-uFJC-curve-fit")
        
        fig = plt.figure()
        plt.scatter(
            self.lmbda_c_eq, self.xi_c, color='blue', marker='o', alpha=1,
            linewidth=2.5,
            label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        plt.plot(
            self.lmbda_c_eq_fit, self.xi_c_fit, linestyle='--', color='red',
            alpha=1, linewidth=2.5, label=r'$u\textrm{FJC model fit}$')
        plt.legend(loc='best', fontsize=18)
        plt.grid(True, alpha=0.25)
        plt.xlim([0, 1.15])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        save_current_figure(
            self.savedir, r'$\lambda_c^{eq}$', 30, r'$\xi_{c}$', 30,
            self.data_file_prefix+"-xi_c-vs-lmbda_c_eq-composite-uFJC-curve-fit")
        
        fig = plt.figure()
        plt.scatter(
            self.lmbda_c_eq_intgr_nu, self.xi_c_intgr_nu, color='blue',
            marker='o', alpha=1, linewidth=2.5,
            label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        plt.plot(
            self.lmbda_c_eq_fit_intgr_nu, self.xi_c_fit_intgr_nu,
            linestyle='--', color='red', alpha=1, linewidth=2.5,
            label=r'$u\textrm{FJC model fit}$')
        plt.legend(loc='best', fontsize=18)
        plt.grid(True, alpha=0.25)
        plt.xlim([0, 1.15])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        save_current_figure(
            self.savedir, r'$\lambda_c^{eq}$', 30, r'$\xi_{c}$', 30,
            self.data_file_prefix+"-intgr_nu-xi_c-vs-lmbda_c_eq-gen-uFJC-curve-fit")

        # plot rate-dependent chain results
        t_max = 0
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            4, 1, gridspec_kw={'height_ratios': [2, 1, 1, 1]}, sharex=True)
        for f_c_dot_indx in range(len(cp.f_c_dot_list)):
            t_max = max(
                [t_max, self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx][-1]])
            ax1.semilogx(
                self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx],
                self.rate_dependent_xi_c___f_c_dot_chunk[f_c_dot_indx],
                linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                alpha=1, linewidth=2.5,
                label=cp.f_c_dot_label_list[f_c_dot_indx])
            ax2.semilogx(
                self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx],
                self.rate_dependent_lmbda_nu___f_c_dot_chunk[f_c_dot_indx],
                linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                alpha=1, linewidth=2.5)
            ax3.semilogx(
                self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx],
                self.rate_dependent_gamma_c___f_c_dot_chunk[f_c_dot_indx],
                linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                alpha=1, linewidth=2.5)
            ax4.semilogx(
                self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx],
                self.rate_dependent_overline_epsilon_cnu_diss_hat___f_c_dot_chunk[f_c_dot_indx],
                linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                alpha=1, linewidth=2.5)
        ax1.hlines(
            y=self.xi_c_max, xmin=0, xmax=t_max, linestyle='--', color='black',
            alpha=1, linewidth=1)
        ax1.legend(loc='best', fontsize=12)
        ax1.tick_params(axis='y', labelsize=16)
        ax1.set_ylabel(r'$\xi_c$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        ax3.tick_params(axis='y', labelsize=16)
        ax3.set_ylabel(r'$\gamma_c$', fontsize=20)
        ax3.grid(True, alpha=0.25)
        ax4.tick_params(axis='y', labelsize=16)
        ax4.set_yticks([0.0, 0.25, 0.5])
        ax4.set_ylabel(
            r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', fontsize=20)
        ax4.grid(True, alpha=0.25)
        plt.xticks(fontsize=16)
        plt.xlabel(r'$t~(sec)$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir,
            self.data_file_prefix+"-rate-dependent-xi_c-lmbda_nu-gamma_c-overline_epsilon_cnu_diss_hat-vs-time")
        
        # plot rate-independent and rate-dependent chain results
        # together
        # plot rate-dependent chain results
        lmbda_c_eq_max = 0
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            4, 1, gridspec_kw={'height_ratios': [1.25, 0.75, 2, 1.5]},
            sharex=True)
        ax1.scatter(
            self.lmbda_c_eq_intgr_nu, self.xi_c_intgr_nu, color='red',
            marker='o', alpha=1, linewidth=1,
            label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        for f_c_dot_indx in range(len(cp.f_c_dot_list)):
            lmbda_c_eq_max = max(
                [lmbda_c_eq_max, self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx][-1]])
            ax1.plot(
                self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
                self.rate_dependent_xi_c___f_c_dot_chunk[f_c_dot_indx],
                linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                alpha=1, linewidth=2.5)
            ax2.plot(
                self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
                self.rate_dependent_lmbda_nu___f_c_dot_chunk[f_c_dot_indx],
                linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                alpha=1, linewidth=2.5)
            ax3.plot(
                self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
                self.rate_dependent_gamma_c___f_c_dot_chunk[f_c_dot_indx],
                linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                alpha=1, linewidth=2.5, label=cp.f_c_dot_label_list[f_c_dot_indx])
            ax4.plot(
                self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
                self.rate_dependent_overline_epsilon_cnu_diss_hat___f_c_dot_chunk[f_c_dot_indx],
                linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                alpha=1, linewidth=2.5)
        ax1.hlines(
            y=self.xi_c_max, xmin=-0.05, xmax=lmbda_c_eq_max + 0.05,
            linestyle='--', color='black', alpha=1, linewidth=1)
        # plot rate-independent chain results
        ax1.plot(
            self.rate_independent_lmbda_c_eq,
            self.rate_independent_xi_c, linestyle='-', color='blue',
            alpha=1, linewidth=2.5, label=r'$\textrm{rate-independent chain}$')
        ax2.plot(
            self.rate_independent_lmbda_c_eq,
            self.rate_independent_lmbda_nu, linestyle='-', color='blue',
            alpha=1, linewidth=2.5)
        ax3.plot(
            self.rate_independent_lmbda_c_eq,
            self.rate_independent_p_c_sci_hat, linestyle='-', color='blue',
            alpha=1, linewidth=2.5)
        ax4.plot(
            self.rate_independent_lmbda_c_eq,
            self.rate_independent_overline_epsilon_cnu_sci_hat, linestyle='-',
            color='black', alpha=1, linewidth=2.5)
        ax4.plot(
            self.rate_independent_lmbda_c_eq,
            self.rate_independent_overline_epsilon_cnu_diss_hat, linestyle='-',
            color='blue', alpha=1, linewidth=2.5)

        ax1.legend(loc='best', fontsize=12)
        ax1.tick_params(axis='y', labelsize=16)
        ax1.set_ylabel(r'$\xi_c$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        ax3.legend(loc='best', fontsize=12)
        ax3.tick_params(axis='y', labelsize=16)
        ax3.set_ylabel(r'$\gamma_c,~\hat{p}_c^{sci}$', fontsize=20)
        ax3.grid(True, alpha=0.25)
        ax4.set_yticks([0.0, 0.25, 0.5])
        ax4.tick_params(axis='y', labelsize=16)
        ax4.set_ylabel(
            r'$\overline{\hat{\varepsilon}_{c\nu}^{sci}},~\overline{\hat{\varepsilon}_{c\nu}^{diss}}$',
            fontsize=20)
        ax4.grid(True, alpha=0.25)
        plt.xlim([-0.05, lmbda_c_eq_max + 0.05])
        plt.xticks(fontsize=16)
        plt.xlabel(r'$\lambda_c^{eq}$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir,
            self.data_file_prefix+"-rate-independent-and-rate-dependent-chains-vs-lmbda_c_eq")

        # plot rate-independent and rate-dependent chain results
        # together while omitting chain scission energy
        # plot rate-dependent chain results
        lmbda_c_eq_max = 0
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            4, 1, gridspec_kw={'height_ratios': [1.25, 0.75, 2, 1.5]},
            sharex=True)
        ax1.scatter(
            self.lmbda_c_eq_intgr_nu, self.xi_c_intgr_nu, color='red',
            marker='o', alpha=1, linewidth=1,
            label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        for f_c_dot_indx in range(len(cp.f_c_dot_list)):
            lmbda_c_eq_max = max([lmbda_c_eq_max, self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx][-1]])
            ax1.plot(
                self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
                self.rate_dependent_xi_c___f_c_dot_chunk[f_c_dot_indx],
                linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                alpha=1, linewidth=2.5)
            ax2.plot(
                self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
                self.rate_dependent_lmbda_nu___f_c_dot_chunk[f_c_dot_indx],
                linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                alpha=1, linewidth=2.5)
            ax3.plot(
                self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
                self.rate_dependent_gamma_c___f_c_dot_chunk[f_c_dot_indx],
                linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                alpha=1, linewidth=2.5,
                label=cp.f_c_dot_label_list[f_c_dot_indx])
            ax4.plot(
                self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
                self.rate_dependent_overline_epsilon_cnu_diss_hat___f_c_dot_chunk[f_c_dot_indx],
                linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
                alpha=1, linewidth=2.5)
        ax1.hlines(
            y=self.xi_c_max, xmin=-0.05, xmax=lmbda_c_eq_max + 0.05,
            linestyle='--', color='black', alpha=1, linewidth=1)
        # plot rate-independent chain results
        ax1.plot(
            self.rate_independent_lmbda_c_eq,
            self.rate_independent_xi_c, linestyle='-', color='blue',
            alpha=1, linewidth=2.5,
            label=r'$\textrm{rate-independent chain}$')
        ax2.plot(
            self.rate_independent_lmbda_c_eq,
            self.rate_independent_lmbda_nu, linestyle='-', color='blue',
            alpha=1, linewidth=2.5)
        ax3.plot(
            self.rate_independent_lmbda_c_eq,
            self.rate_independent_p_c_sci_hat, linestyle='-', color='blue',
            alpha=1, linewidth=2.5)
        ax4.plot(
            self.rate_independent_lmbda_c_eq,
            self.rate_independent_overline_epsilon_cnu_diss_hat, linestyle='-',
            color='blue', alpha=1, linewidth=2.5)

        ax1.legend(loc='best', fontsize=12)
        ax1.tick_params(axis='y', labelsize=16)
        ax1.set_ylabel(r'$\xi_c$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        ax3.tick_params(axis='y', labelsize=16)
        ax3.legend(loc='best', fontsize=12)
        ax3.set_ylabel(r'$\gamma_c,~\hat{p}_c^{sci}$', fontsize=20)
        ax3.grid(True, alpha=0.25)
        ax4.set_yticks([0.0, 0.25, 0.5])
        ax4.tick_params(axis='y', labelsize=16)
        ax4.set_ylabel(
            r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$',
            fontsize=20)
        ax4.grid(True, alpha=0.25)
        plt.xlim([-0.05, lmbda_c_eq_max + 0.05])
        plt.xticks(fontsize=16)
        plt.xlabel(r'$\lambda_c^{eq}$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir,
            self.data_file_prefix+"-rate-independent-and-rate-dependent-chains-vs-lmbda_c_eq-no-epsilon_cnu_sci")


if __name__ == '__main__':

    T = 298 # absolute room temperature, K

    AFM_chain_tensile_tests_dict = {
        "al-maawali-et-al": [
            "chain-a"
        ],
        "ortiz-and-hadziioannou": [
            "chain-a", "chain-b", "chain-c", "chain-d", "chain-e", "chain-f",
            "chain-g", "chain-h"
        ],
        "hugel-et-al": [
            "chain-a", "chain-b"]
        ,
        "sakai-et-al": [
            "chain-a"
        ],
        "yamamoto-et-al": [
            "chain-a", "chain-b", "chain-c", "chain-d", "chain-e"
        ],
        "zhang-et-al": [
            "chain-a"
        ],
        "wang-et-al": [
            "chain-a", "chain-b", "chain-c"
        ]
    }

    AFM_chain_tensile_test_list = [
        (paper_authors, chain) for paper_authors, chain_list
        in AFM_chain_tensile_tests_dict.items() for chain in chain_list
    ]

    AFM_chain_tensile_tests_characterizer_list = [
        AFMChainTensileTestCurveFitCharacterizer(
            paper_authors=AFM_chain_tensile_test[0],
            chain=AFM_chain_tensile_test[1], T=T)
        for AFM_chain_tensile_test in AFM_chain_tensile_test_list
    ]

    for AFM_chain_tensile_test_indx \
        in range(len(AFM_chain_tensile_tests_characterizer_list)):
        AFM_chain_tensile_tests_characterizer_list[AFM_chain_tensile_test_indx].characterization()
        AFM_chain_tensile_tests_characterizer_list[AFM_chain_tensile_test_indx].finalization()
    
    # characterizer = AFMChainTensileTestCurveFitCharacterizer(
    #     paper_authors=AFM_chain_tensile_test_list[0][0],
    #     chain=AFM_chain_tensile_test_list[0][1], T=T)
    # characterizer.characterization()
    # characterizer.finalization()

    # characterizer = AFMChainTensileTestCurveFitCharacterizer(
    #     paper_authors=AFM_chain_tensile_test_list[9][0],
    #     chain=AFM_chain_tensile_test_list[9][1], T=T)
    # characterizer.characterization()
    # characterizer.finalization()