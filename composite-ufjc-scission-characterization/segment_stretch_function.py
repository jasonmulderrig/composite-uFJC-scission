"""The segment stretch characterization module for composite uFJCs"""

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


class SegmentStretchFunctionCharacterizer(CompositeuFJCScissionCharacterizer):
    """The characterization class assessing the segment stretch for
    composite uFJCs. It inherits all attributes and methods from the
    ``CompositeuFJCScissionCharacterizer`` class.
    """
    def __init__(self):
        """Initializes the
        ``SegmentStretchFunctionCharacterizer`` class by initializing
        and inheriting all attributes and methods from the
        ``CompositeuFJCScissionCharacterizer`` class.
        """
        CompositeuFJCScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        p = self.parameters

        lmbda_c_eq_inc = 0.0001
        lmbda_c_eq_min = lmbda_c_eq_inc
        lmbda_c_eq_max = 1.5

        p.characterizer.lmbda_c_eq_inc = lmbda_c_eq_inc
        p.characterizer.lmbda_c_eq_min = lmbda_c_eq_min
        p.characterizer.lmbda_c_eq_max = lmbda_c_eq_max
    
    def prefix(self):
        """Set characterization prefix"""
        return "segment_stretch_function"
    
    def characterization(self):
        """Define characterization routine"""
        def subcrit_governing_lmbda_nu_func(lmbda_nu, lmbda_c_eq, kappa_nu):
            """Sub-critical chain state governing equation for the
            segment stretch
            
            This function is used to compute the segment stretch for the
            sub-critical chain state via Brent's method with hyperbolic
            extrapolation, where the equilibrium chain stretch,
            nondimensional segment stiffness, and a segment stretch
            bracketing interval are provided.
            """
            langevin_arg = kappa_nu * (lmbda_nu-1.)
            langevin = 1. / np.tanh(langevin_arg) - 1. / langevin_arg
            return langevin - (lmbda_c_eq-lmbda_nu+1.)
        
        def supercrit_governing_lmbda_nu_func(
                lmbda_nu, lmbda_c_eq, zeta_nu_char, kappa_nu):
            """Super-critical chain state governing equation for the
            segment stretch
            
            This function is used to compute the segment stretch for the
            super-critical chain state via Brent's method with
            hyperbolic extrapolation, where the equilibrium chain
            stretch, nondimensional characteristic segment potential
            energy scale, nondimensional segment stiffness, and a
            segment stretch bracketing interval are provided.
            """
            langevin_arg = zeta_nu_char**2 / (kappa_nu*(lmbda_nu-1.)**3)
            langevin = 1. / np.tanh(langevin_arg) - 1. / langevin_arg
            return langevin - (lmbda_c_eq-lmbda_nu+1.)

        cp = self.parameters.characterizer

        # nu=125, zeta_nu_char=100, and kappa_nu=1000
        single_chain = (
            RateIndependentScissionCompositeuFJC(
                nu=cp.nu_single_chain_list[1],
                zeta_nu_char=cp.zeta_nu_char_single_chain_list[2],
                kappa_nu=cp.kappa_nu_single_chain_list[2])
        )

        # Define the values of the equilibrium chain stretch to
        # calculate over
        lmbda_c_eq_num_steps = (
            int(np.around(
                (cp.lmbda_c_eq_max-cp.lmbda_c_eq_min)/cp.lmbda_c_eq_inc))
            + 1
        )
        lmbda_c_eq_steps = (
            np.linspace(
                cp.lmbda_c_eq_min, cp.lmbda_c_eq_max, lmbda_c_eq_num_steps)
        )
        
        # Make arrays to allocate results
        lmbda_c_eq            = []
        lmbda_nu              = []
        lmbda_comp_nu         = []
        lmbda_nu_exact        = []
        lmbda_nu_err          = []
        lmbda_nu_bergapprx    = []
        lmbda_nu_bergapprxerr = []
        lmbda_nu_padeapprx    = []
        lmbda_nu_padeapprxerr = []
        
        # Calculate results through specified equilibrium chain stretch
        # values
        for lmbda_c_eq_indx in range(lmbda_c_eq_num_steps):
            lmbda_c_eq_val    = lmbda_c_eq_steps[lmbda_c_eq_indx]
            lmbda_nu_val      = single_chain.lmbda_nu_func(lmbda_c_eq_val)
            lmbda_comp_nu_val = lmbda_c_eq_val - lmbda_nu_val + 1.
            
            if lmbda_c_eq_val < single_chain.lmbda_c_eq_crit:
                lmbda_nu_exact_val = (
                    optimize.brenth(
                        subcrit_governing_lmbda_nu_func, 0.8, 2.0,
                        args=(lmbda_c_eq_val, single_chain.kappa_nu))
                )
            
            else:
                lmbda_nu_exact_val = (
                    optimize.brenth(
                        supercrit_governing_lmbda_nu_func, 0.8, 2.0,
                        args=(lmbda_c_eq_val, single_chain.zeta_nu_char,
                                single_chain.kappa_nu))
                )
            
            lmbda_nu_err_val = (
                np.abs((lmbda_nu_val-lmbda_nu_exact_val)/lmbda_nu_exact_val)
                * 100
            )
            lmbda_nu_bergapprx_val = (
                single_chain.subcrit_lmbda_nu_berg_approx_func(lmbda_c_eq_val)
            )
            lmbda_nu_bergapprxerr_val = (
                np.abs(
                    (lmbda_nu_bergapprx_val-lmbda_nu_exact_val)/lmbda_nu_exact_val)
                * 100
            )
            lmbda_nu_padeapprx_val = (
                single_chain.subcrit_lmbda_nu_pade_approx_func(lmbda_c_eq_val)
            )
            lmbda_nu_padeapprxerr_val = (
                np.abs(
                    (lmbda_nu_padeapprx_val-lmbda_nu_exact_val)/lmbda_nu_exact_val)
                * 100
            )
            
            lmbda_c_eq.append(lmbda_c_eq_val)
            lmbda_nu.append(lmbda_nu_val)
            lmbda_comp_nu.append(lmbda_comp_nu_val)
            lmbda_nu_exact.append(lmbda_nu_exact_val)
            lmbda_nu_err.append(lmbda_nu_err_val)
            lmbda_nu_bergapprx.append(lmbda_nu_bergapprx_val)
            lmbda_nu_bergapprxerr.append(lmbda_nu_bergapprxerr_val)
            lmbda_nu_padeapprx.append(lmbda_nu_padeapprx_val)
            lmbda_nu_padeapprxerr.append(lmbda_nu_padeapprxerr_val)
        
        lmbda_nu_pade2berg_crit, lmbda_c_eq_pade2berg_crit = (
            single_chain.pade2berg_crit_func()
        )

        self.single_chain = single_chain

        self.lmbda_c_eq                = lmbda_c_eq
        self.lmbda_nu                  = lmbda_nu
        self.lmbda_comp_nu             = lmbda_comp_nu
        self.lmbda_nu_exact            = lmbda_nu_exact
        self.lmbda_nu_err              = lmbda_nu_err
        self.lmbda_nu_bergapprx        = lmbda_nu_bergapprx
        self.lmbda_nu_bergapprxerr     = lmbda_nu_bergapprxerr
        self.lmbda_nu_padeapprx        = lmbda_nu_padeapprx
        self.lmbda_nu_padeapprxerr     = lmbda_nu_padeapprxerr
        self.lmbda_nu_pade2berg_crit   = lmbda_nu_pade2berg_crit
        self.lmbda_c_eq_pade2berg_crit = lmbda_c_eq_pade2berg_crit

    def finalization(self):
        """Define finalization analysis"""
        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing

        lmbda_comp_nu_array   = np.asarray(self.lmbda_comp_nu)
        bergapprx_cutoff_indx = (
            np.argmin(np.abs(lmbda_comp_nu_array-cp.bergapprx_lmbda_nu_cutoff))
        )

        # plot results
        latex_formatting_figure(ppp)
        
        fig = plt.figure()
        plt.vlines(
            x=self.single_chain.lmbda_c_eq_crit, ymin=self.lmbda_nu[0]-0.025,
            ymax=self.single_chain.lmbda_nu_crit, linestyle=':', 
            color='black', alpha=1, linewidth=1)
        plt.hlines(
            y=self.single_chain.lmbda_nu_crit, xmin=self.lmbda_c_eq[0]-0.05,
            xmax=self.single_chain.lmbda_c_eq_crit, linestyle=':', 
            color='black', alpha=1, linewidth=1)
        plt.plot(
            self.lmbda_c_eq, self.lmbda_nu_exact, linestyle='-', color='red',
            alpha=1, linewidth=2.5,
            label=r'$\textrm{Highly accurate numerical solution}$')
        plt.plot(
            self.lmbda_c_eq, self.lmbda_nu, linestyle='--', color='blue',
            alpha=1, linewidth=2.5, label=r'$\textrm{Approximated solution}$')
        plt.legend(loc='best', fontsize=12)
        plt.xlim([self.lmbda_c_eq[0]-0.05, self.lmbda_c_eq[-1]+0.05])
        plt.xticks(fontsize=14)
        plt.ylim([self.lmbda_nu[0]-0.025, self.lmbda_nu[-1]+0.025])
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{eq}$', 20, r'$\lambda_{\nu}$', 20,
            "lmbda_nu-vs-lmbda_c_eq-exact-and-approximated-solutions")
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        
        ax1.vlines(
            x=self.single_chain.lmbda_c_eq_crit, ymin=self.lmbda_nu[0]-0.025,
            ymax=self.single_chain.lmbda_nu_crit, linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax1.hlines(
            y=self.single_chain.lmbda_nu_crit, xmin=self.lmbda_c_eq[0]-0.05,
            xmax=self.single_chain.lmbda_c_eq_crit, linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax1.plot(
            self.lmbda_c_eq, self.lmbda_nu_exact, linestyle='-', color='red',
            alpha=1, linewidth=2.5,
            label=r'$\textrm{Highly accurate numerical solution}$')
        ax1.plot(
            self.lmbda_c_eq, self.lmbda_nu, linestyle='--', color='blue',
            alpha=1, linewidth=2.5, label=r'$\textrm{Approximated solution}$')
        ax1.legend(loc='best', fontsize=12)
        ax1.set_ylim([self.lmbda_nu[0]-0.025, self.lmbda_nu[-1]+0.025])
        ax1.tick_params(axis='y', labelsize=14)
        ax1.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        
        ax2.semilogy(
            self.lmbda_c_eq, self.lmbda_nu_err, linestyle='-', color='blue',
            alpha=1, linewidth=2.5)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        
        plt.xlim([self.lmbda_c_eq[0]-0.05, self.lmbda_c_eq[-1]+0.05])
        plt.xticks(fontsize=14)
        plt.xlabel(r'$\lambda_c^{eq}$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir,
            "lmbda_nu-vs-lmbda_c_eq-exact-and-approximated-solutions-percent-error")
        
        fig = plt.figure()
        plt.plot(
            self.lmbda_c_eq, self.lmbda_comp_nu, linestyle='-', color='blue',
            alpha=1, linewidth=2.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{eq}$', 20,
            r'$\lambda_c^{eq} - \lambda_{\nu} + 1$', 20,
            "lmbda_comp_nu-vs-lmbda_c_eq")
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        
        ax1.axvline(
            x=self.lmbda_c_eq[bergapprx_cutoff_indx], linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax1.axvline(
            x=self.lmbda_c_eq_pade2berg_crit, linestyle='--',
            color='black', alpha=1, linewidth=1)
        ax1.plot(
            self.lmbda_c_eq, self.lmbda_nu_padeapprx, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\textrm{Pad\'e approximant}$')
        ax1.plot(
            self.lmbda_c_eq, self.lmbda_nu_bergapprx, linestyle=':',
            color='red', alpha=1, linewidth=2.5,
            label=r'$\textrm{Bergstr\"{o}m approximant}$')
        ax1.legend(loc='best', fontsize=12)
        ax1.set_xlim(
            [-0.01, self.single_chain.lmbda_c_eq_crit-cp.lmbda_c_eq_inc])
        ax1.tick_params(axis='x', labelsize=14)
        ax1.set_ylim([0.99, self.single_chain.lmbda_nu_crit])
        ax1.tick_params(axis='y', labelsize=14)
        ax1.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        
        ax2.axvline(
            x=self.lmbda_c_eq[bergapprx_cutoff_indx], linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax2.axvline(
            x=self.lmbda_c_eq_pade2berg_crit, linestyle='--',
            color='black', alpha=1, linewidth=1)
        ax2.semilogy(
            self.lmbda_c_eq, self.lmbda_nu_padeapprxerr, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\textrm{Pad\'e approximant}$')
        ax2.semilogy(
            self.lmbda_c_eq, self.lmbda_nu_bergapprxerr, linestyle='-',
            color='red', alpha=1, linewidth=2.5,
            label=r'$\textrm{Bergstr\"{o}m approximant}$')
        ax2.legend(loc='best', fontsize=12)
        ax2.set_xlim(
            [-0.01, self.single_chain.lmbda_c_eq_crit-cp.lmbda_c_eq_inc])
        ax2.tick_params(axis='x', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        
        plt.xlabel(r'$\lambda_c^{eq}$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir, "lmbda_nu-vs-lmbda_c_eq-approximation-comparison")

if __name__ == '__main__':

    characterizer = SegmentStretchFunctionCharacterizer()
    characterizer.characterization()
    characterizer.finalization()