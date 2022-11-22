"""The equilibrium chain stretch characterization module for composite
uFJCs
"""

# import external modules
from __future__ import division
from composite_ufjc_scission import (
    CompositeuFJCScissionCharacterizer,
    RateIndependentScissionCompositeuFJC,
    latex_formatting_figure,
    save_current_figure,
    save_current_figure_no_labels
)
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


class EquilibriumChainStretchFunctionCharacterizer(
        CompositeuFJCScissionCharacterizer):
    """The characterization class assessing the equilibrium chain
    stretch for composite uFJCs. It inherits all attributes and methods
    from the ``CompositeuFJCScissionCharacterizer`` class.
    """
    def __init__(self):
        """Initializes the
        ``EquilibriumChainStretchFunctionCharacterizer`` class by
        initializing and inheriting all attributes and methods from the
        ``CompositeuFJCScissionCharacterizer`` class.
        """
        CompositeuFJCScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        p = self.parameters

        lmbda_nu_inc = 0.0001
        lmbda_nu_min = 1.+lmbda_nu_inc
        lmbda_nu_max = 1.5

        p.characterizer.lmbda_nu_inc = lmbda_nu_inc
        p.characterizer.lmbda_nu_min = lmbda_nu_min
        p.characterizer.lmbda_nu_max = lmbda_nu_max
    
    def prefix(self):
        """Set characterization prefix"""
        return "equilibrium_chain_stretch_function"
    
    def characterization(self):
        """Define characterization routine"""
        def subcrit_governing_lmbda_c_eq_func(lmbda_c_eq, lmbda_nu, kappa_nu):
            """Sub-critical chain state governing equation for the
            equilibrium chain stretch
            
            This function is used to compute the equilibrium chain
            stretch for the sub-critical chain state via Brent's method
            with hyperbolic extrapolation, where the segment stretch,
            nondimensional segment stiffness, and an equilibrium chain
            stretch bracketing interval are provided.
            """
            langevin_arg = kappa_nu * (lmbda_nu-1.)
            langevin = 1. / np.tanh(langevin_arg) - 1. / langevin_arg
            return langevin - (lmbda_c_eq-lmbda_nu+1.)
        
        def supercrit_governing_lmbda_c_eq_func(
                lmbda_c_eq, lmbda_nu, kappa_nu, zeta_nu_char):
            """Super-critical chain state governing equation for the
            equilibrium chain stretch
            
            This function is used to compute the equilibrium chain
            stretch for the super-critical chain state via Brent's
            method with hyperbolic extrapolation, where the segment
            stretch, nondimensional characteristic segment potential
            energy scale, nondimensional segment stiffness, and an
            equilibrium chain stretch bracketing interval are provided.
            """
            langevin_arg = zeta_nu_char**2 / (kappa_nu*(lmbda_nu-1.)**3)
            langevin = 1. / np.tanh(langevin_arg) - 1. / langevin_arg
            return langevin - (lmbda_c_eq-lmbda_nu+1.)
        
        def subcrit_lmbda_c_eq_berg_approx_func(kappa_nu, lmbda_nu):
            """Sub-critical chain state equilibrium chain stretch as
            derived via the Bergstrom approximant for the inverse
            Langevin function

            This function computes the sub-critical chain state
            equilibrium chain stretch (as derived via the Bergstrom
            approximant for the inverse Langevin function) as a function
            of the segment stretch and nondimensional segment stiffness.
            """
            return lmbda_nu - 1. / (kappa_nu*(lmbda_nu-1.))
        
        def subcrit_lmbda_c_eq_pade_approx_func(kappa_nu, lmbda_nu):
            """Sub-critical chain state equilibrium chain stretch as
            derived via the Pade approximant for the inverse Langevin
            function

            This function computes the sub-critical chain state
            equilibrium chain stretch (as derived via the Pade
            approximant for the inverse Langevin function) as a function
            of the segment stretch and nondimensional segment stiffness.
            """
            # analytical solution
            if lmbda_nu == 1.:
                return 0.
            
            else:
                alpha_tilde = 1.
            
            trm_i  = kappa_nu + 3.
            trm_ii = 1.
            beta_tilde = trm_i * (trm_ii-lmbda_nu)

            trm_i   = 2. * kappa_nu + 3.
            trm_ii  = 2.
            trm_iii = 2. * kappa_nu
            gamma_tilde = trm_i * (lmbda_nu**2-trm_ii*lmbda_nu) + trm_iii
            
            trm_i   = kappa_nu + 1.
            trm_ii  = 3.
            trm_iii = 2.
            trm_iv  = kappa_nu
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
        
        cp = self.parameters.characterizer

        # nu=125, zeta_nu_char=100, and kappa_nu=1000
        single_chain = RateIndependentScissionCompositeuFJC(
            nu=cp.nu_single_chain_list[1],
            zeta_nu_char=cp.zeta_nu_char_single_chain_list[2],
            kappa_nu=cp.kappa_nu_single_chain_list[2])

        # Define the segment stretch values to calculate over
        lmbda_nu_num_steps = (int(
            np.around(
                (cp.lmbda_nu_max-cp.lmbda_nu_min)/cp.lmbda_nu_inc))
            + 1)
        lmbda_nu_steps = np.linspace(
            cp.lmbda_nu_min, cp.lmbda_nu_max, lmbda_nu_num_steps)
        
        # Make arrays to allocate results
        lmbda_nu                = []
        lmbda_c_eq              = []
        lmbda_comp_nu           = []
        lmbda_c_eq_exact        = []
        lmbda_c_eq_err          = []
        lmbda_c_eq_bergapprx    = []
        lmbda_c_eq_bergapprxerr = []
        lmbda_c_eq_padeapprx    = []
        lmbda_c_eq_padeapprxerr = []
        
        # Calculate results through specified segment stretch values
        for lmbda_nu_indx in range(lmbda_nu_num_steps):
            lmbda_nu_val      = lmbda_nu_steps[lmbda_nu_indx]
            lmbda_c_eq_val    = single_chain.lmbda_c_eq_func(lmbda_nu_val)
            lmbda_comp_nu_val = lmbda_c_eq_val - lmbda_nu_val + 1.
            
            if lmbda_nu_val < single_chain.lmbda_nu_crit:
                lmbda_c_eq_exact_val = optimize.brenth(
                    subcrit_governing_lmbda_c_eq_func, -0.1, 2.0,
                    args=(lmbda_nu_val, single_chain.kappa_nu))
            
            else:
                lmbda_c_eq_exact_val = optimize.brenth(
                    supercrit_governing_lmbda_c_eq_func, -0.1, 2.0,
                    args=(lmbda_nu_val, single_chain.kappa_nu,
                            single_chain.zeta_nu_char))
            
            lmbda_c_eq_err_val = (np.abs(
                (lmbda_c_eq_val-lmbda_c_eq_exact_val)/lmbda_c_eq_exact_val)
                * 100)
            lmbda_c_eq_bergapprx_val = (
                subcrit_lmbda_c_eq_berg_approx_func(
                    single_chain.kappa_nu, lmbda_nu_val)
            )
            lmbda_c_eq_bergapprxerr_val = (np.abs(
                (lmbda_c_eq_bergapprx_val-lmbda_c_eq_exact_val)/lmbda_c_eq_exact_val)
                * 100)
            lmbda_c_eq_padeapprx_val = (
                subcrit_lmbda_c_eq_pade_approx_func(
                    single_chain.kappa_nu, lmbda_nu_val)
            )
            lmbda_c_eq_padeapprxerr_val = (np.abs(
                (lmbda_c_eq_padeapprx_val-lmbda_c_eq_exact_val)/lmbda_c_eq_exact_val)
                * 100)
            
            lmbda_nu.append(lmbda_nu_val)
            lmbda_c_eq.append(lmbda_c_eq_val)
            lmbda_comp_nu.append(lmbda_comp_nu_val)
            lmbda_c_eq_exact.append(lmbda_c_eq_exact_val)
            lmbda_c_eq_err.append(lmbda_c_eq_err_val)
            lmbda_c_eq_bergapprx.append(lmbda_c_eq_bergapprx_val)
            lmbda_c_eq_bergapprxerr.append(lmbda_c_eq_bergapprxerr_val)
            lmbda_c_eq_padeapprx.append(lmbda_c_eq_padeapprx_val)
            lmbda_c_eq_padeapprxerr.append(lmbda_c_eq_padeapprxerr_val)
        
        self.single_chain = single_chain

        self.lmbda_nu                = lmbda_nu
        self.lmbda_c_eq              = lmbda_c_eq
        self.lmbda_comp_nu           = lmbda_comp_nu
        self.lmbda_c_eq_exact        = lmbda_c_eq_exact
        self.lmbda_c_eq_err          = lmbda_c_eq_err
        self.lmbda_c_eq_bergapprx    = lmbda_c_eq_bergapprx
        self.lmbda_c_eq_bergapprxerr = lmbda_c_eq_bergapprxerr
        self.lmbda_c_eq_padeapprx    = lmbda_c_eq_padeapprx
        self.lmbda_c_eq_padeapprxerr = lmbda_c_eq_padeapprxerr

    def finalization(self):
        """Define finalization analysis"""
        cp = self.parameters.characterizer
        pp = self.parameters.post_processing

        lmbda_comp_nu_array        = np.asarray(self.lmbda_comp_nu)
        lmbda_c_eq_bergapprx_array = np.asarray(self.lmbda_c_eq_bergapprx)
        lmbda_c_eq_padeapprx_array = np.asarray(self.lmbda_c_eq_padeapprx)
        
        bergapprx_cutoff_indx = np.argmin(
            np.abs(lmbda_comp_nu_array-cp.bergapprx_lmbda_nu_cutoff))
        pade2berg_crit_indx = np.argmin(
            np.abs(lmbda_c_eq_padeapprx_array[0:bergapprx_cutoff_indx]-lmbda_c_eq_bergapprx_array[0:bergapprx_cutoff_indx]))

        # plot results
        latex_formatting_figure(pp)
        
        fig = plt.figure()
        plt.hlines(
            y=self.single_chain.lmbda_c_eq_crit, xmin=self.lmbda_nu[0]-0.025,
            xmax=self.single_chain.lmbda_nu_crit, linestyle=':',
            color='black', alpha=1, linewidth=1)
        plt.vlines(
            x=self.single_chain.lmbda_nu_crit, ymin=self.lmbda_c_eq[0]-0.05,
            ymax=self.single_chain.lmbda_c_eq_crit, linestyle=':',
            color='black', alpha=1, linewidth=1)
        plt.plot(
            self.lmbda_nu, self.lmbda_c_eq_exact, linestyle='-', color='red',
            alpha=1, linewidth=2.5,
            label=r'$\textrm{Highly accurate numerical solution}$')
        plt.plot(
            self.lmbda_nu, self.lmbda_c_eq, linestyle='--', color='blue',
            alpha=1, linewidth=2.5, label=r'$\textrm{Approximated solution}$')
        plt.legend(loc='best', fontsize=12)
        plt.ylim([self.lmbda_c_eq[0]-0.05, self.lmbda_c_eq[-1]+0.05])
        plt.yticks(fontsize=14)
        plt.xlim([self.lmbda_nu[0]-0.025, self.lmbda_nu[-1]+0.025])
        plt.xticks(fontsize=14)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_{\nu}$', 20, r'$\lambda_c^{eq}$', 20,
            "lmbda_c_eq-vs-lmbda_nu-exact-and-approximated-solutions")
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        
        ax1.hlines(
            y=self.single_chain.lmbda_c_eq_crit, xmin=self.lmbda_nu[0]-0.025,
            xmax=self.single_chain.lmbda_nu_crit, linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax1.vlines(
            x=self.single_chain.lmbda_nu_crit, ymin=self.lmbda_c_eq[0]-0.05,
            ymax=self.single_chain.lmbda_c_eq_crit, linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax1.plot(
            self.lmbda_nu, self.lmbda_c_eq_exact, linestyle='-', color='red',
            alpha=1, linewidth=2.5,
            label=r'$\textrm{Highly accurate numerical solution}$')
        ax1.plot(
            self.lmbda_nu, self.lmbda_c_eq, linestyle='--', color='blue',
            alpha=1, linewidth=2.5, label=r'$\textrm{Approximated solution}$')
        ax1.legend(loc='best', fontsize=12)
        ax1.set_ylim([self.lmbda_c_eq[0]-0.05, self.lmbda_c_eq[-1]+0.05])
        ax1.tick_params(axis='y', labelsize=14)
        ax1.set_ylabel(r'$\lambda_c^{eq}$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        
        ax2.semilogy(
            self.lmbda_nu, self.lmbda_c_eq_err, linestyle='-', color='blue',
            alpha=1, linewidth=2.5)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        
        plt.xlim([self.lmbda_nu[0]-0.025, self.lmbda_nu[-1]+0.025])
        plt.xticks(fontsize=14)
        plt.xlabel(r'$\lambda_{\nu}$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir,
            "lmbda_c_eq-vs-lmbda_nu-exact-and-approximated-solutions-percent-error")
        
        fig = plt.figure()
        plt.plot(
            self.lmbda_nu, self.lmbda_comp_nu, linestyle='-', color='blue',
            alpha=1, linewidth=2.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_{\nu}$', 20,
            r'$\lambda_c^{eq} - \lambda_{\nu} + 1$', 20,
            "lmbda_comp_nu-vs-lmbda_nu")
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        
        ax1.axvline(
            x=self.lmbda_nu[bergapprx_cutoff_indx], linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax1.axvline(
            x=self.lmbda_nu[pade2berg_crit_indx], linestyle='--',
            color='black', alpha=1, linewidth=1)
        ax1.plot(
            self.lmbda_nu, self.lmbda_c_eq_padeapprx, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\textrm{Pad\'e approximate}$')
        ax1.plot(
            self.lmbda_nu, self.lmbda_c_eq_bergapprx, linestyle=':',
            color='red', alpha=1, linewidth=2.5,
            label=r'$\textrm{Bergstr\"{o}m approximate}$')
        ax1.legend(loc='best', fontsize=12)
        ax1.set_xlim([0.99, self.single_chain.lmbda_nu_crit-cp.lmbda_nu_inc])
        ax1.tick_params(axis='x', labelsize=14)
        ax1.set_ylim([-0.01, self.single_chain.lmbda_c_eq_crit])
        ax1.tick_params(axis='y', labelsize=14)
        ax1.set_ylabel(r'$\lambda_c^{eq}$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        
        ax2.axvline(
            x=self.lmbda_nu[bergapprx_cutoff_indx], linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax2.axvline(
            x=self.lmbda_nu[pade2berg_crit_indx], linestyle='--',
            color='black', alpha=1, linewidth=1)
        ax2.semilogy(
            self.lmbda_nu, self.lmbda_c_eq_padeapprxerr, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\textrm{Pad\'e approximate}$')
        ax2.semilogy(
            self.lmbda_nu, self.lmbda_c_eq_bergapprxerr, linestyle='-',
            color='red', alpha=1, linewidth=2.5,
            label=r'$\textrm{Bergstr\"{o}m approximate}$')
        ax2.legend(loc='best', fontsize=12)
        ax2.set_xlim([0.99, self.single_chain.lmbda_nu_crit-cp.lmbda_nu_inc])
        ax2.tick_params(axis='x', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        
        plt.xlabel(r'$\lambda_{\nu}$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir, "lmbda_c_eq-vs-lmbda_nu-approximation-comparison")

if __name__ == '__main__':

    characterizer = EquilibriumChainStretchFunctionCharacterizer()
    characterizer.characterization()
    characterizer.finalization()