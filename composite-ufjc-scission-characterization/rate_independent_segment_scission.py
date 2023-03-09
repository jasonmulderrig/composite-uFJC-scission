"""The rate-independent segment scission characterization module for
composite uFJCs
"""

# import external modules
from __future__ import division
from composite_ufjc_scission import (
    CompositeuFJCScissionCharacterizer,
    RateIndependentScissionCompositeuFJC,
    latex_formatting_figure,
    save_current_figure_no_labels,
    save_current_figure
)
import numpy as np
import matplotlib.pyplot as plt


class RateIndependentSegmentScissionCharacterizer(
        CompositeuFJCScissionCharacterizer):
    """The characterization class assessing the rate-independent segment
    scission for composite uFJCs. It inherits all attributes and methods
    from the ``CompositeuFJCScissionCharacterizer`` class.
    """
    def __init__(self):
        """Initializes the
        ``RateIndependentSegmentScissionCharacterizer`` class by
        initializing and inheriting all attributes and methods from the
        ``CompositeuFJCScissionCharacterizer`` class.
        """
        CompositeuFJCScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        pass
    
    def prefix(self):
        """Set characterization prefix"""
        return "rate_independent_segment_scission"
    
    def characterization(self):
        """Define characterization routine"""
        cp = self.parameters.characterizer

        # nu=125, zeta_nu_char=100, and kappa_nu=1000
        single_chain = RateIndependentScissionCompositeuFJC(
            nu=cp.nu_single_chain_list[1],
            zeta_nu_char=cp.zeta_nu_char_single_chain_list[2],
            kappa_nu=cp.kappa_nu_single_chain_list[2])

        # Define the applied segment stretch values to calculate over
        lmbda_nu_hat_num_steps = (
            int(np.around(
                (single_chain.lmbda_nu_crit-single_chain.lmbda_nu_ref)/single_chain.lmbda_nu_hat_inc))
            + 1
        )
        lmbda_nu_hat_steps = (
            np.linspace(
                single_chain.lmbda_nu_ref, single_chain.lmbda_nu_crit,
                lmbda_nu_hat_num_steps)
        )
        
        # Make arrays to allocate results
        lmbda_nu_hat     = []
        lmbda_nu_hat_max = []
        e_nu_sci_hat            = []
        e_nu_sci_hat_analytical = []
        e_nu_sci_hat_err        = []
        e_nu_sci_hat_abs_diff   = []
        epsilon_nu_sci_hat            = []
        epsilon_nu_sci_hat_analytical = []
        epsilon_nu_sci_hat_err        = []
        epsilon_nu_sci_hat_abs_diff   = []
        p_nu_sci_hat            = []
        p_nu_sci_hat_analytical = []
        p_nu_sci_hat_err        = []
        p_nu_sci_hat_abs_diff   = []
        p_nu_sur_hat            = []
        p_nu_sur_hat_analytical = []
        p_nu_sur_hat_err        = []
        p_nu_sur_hat_abs_diff   = []
        epsilon_nu_diss_hat            = []
        epsilon_nu_diss_hat_analytical = []
        epsilon_nu_diss_hat_err        = []
        epsilon_nu_diss_hat_abs_diff   = []
        epsilon_nu_diss_hat_equiv      = []
        epsilon_nu_diss_hat_equiv_err  = []
        epsilon_nu_diss_hat_equiv_abs_diff = []

        # initialization
        lmbda_nu_hat_max_val = 0.
        epsilon_nu_diss_hat_val            = 0.
        epsilon_nu_diss_hat_analytical_val = 0.
        epsilon_nu_diss_hat_err_val        = 0.
        epsilon_nu_diss_hat_abs_diff_val   = 0.
        epsilon_nu_diss_hat_equiv_val      = 0.
        epsilon_nu_diss_hat_equiv_err_val  = 0.
        epsilon_nu_diss_hat_equiv_abs_diff_val = 0.
        
        # Calculate results through specified applied segment
        # stretch values
        for lmbda_nu_hat_indx in range(lmbda_nu_hat_num_steps):
            lmbda_nu_hat_val = lmbda_nu_hat_steps[lmbda_nu_hat_indx]
            lmbda_nu_hat_max_val = max([lmbda_nu_hat_max_val, lmbda_nu_hat_val])
            e_nu_sci_hat_val = single_chain.e_nu_sci_hat_func(lmbda_nu_hat_val)
            e_nu_sci_hat_analytical_val = (
                single_chain.e_nu_sci_hat_analytical_func(lmbda_nu_hat_val)
            )
            e_nu_sci_hat_err_val = (
                np.abs(
                    (e_nu_sci_hat_analytical_val-e_nu_sci_hat_val)/(e_nu_sci_hat_val+single_chain.cond_val))
                * 100
            )
            e_nu_sci_hat_abs_diff_val = (
                np.abs(e_nu_sci_hat_analytical_val-e_nu_sci_hat_val)
            )
            epsilon_nu_sci_hat_val = (
                single_chain.epsilon_nu_sci_hat_func(lmbda_nu_hat_val)
            )
            epsilon_nu_sci_hat_analytical_val = (
                single_chain.epsilon_nu_sci_hat_analytical_func(lmbda_nu_hat_val)
            )
            epsilon_nu_sci_hat_err_val = (
                np.abs(
                    (epsilon_nu_sci_hat_analytical_val-epsilon_nu_sci_hat_val)/(epsilon_nu_sci_hat_val+single_chain.cond_val))
                * 100
            )
            epsilon_nu_sci_hat_abs_diff_val = (
                np.abs(epsilon_nu_sci_hat_analytical_val-epsilon_nu_sci_hat_val)
            )
            p_nu_sci_hat_val = single_chain.p_nu_sci_hat_func(lmbda_nu_hat_val)
            p_nu_sci_hat_analytical_val = (
                single_chain.p_nu_sci_hat_analytical_func(lmbda_nu_hat_val)
            )
            p_nu_sci_hat_err_val = (
                np.abs(
                    (p_nu_sci_hat_analytical_val-p_nu_sci_hat_val)/(p_nu_sci_hat_val+single_chain.cond_val))
                * 100
            )
            p_nu_sci_hat_abs_diff_val = (
                np.abs(p_nu_sci_hat_analytical_val-p_nu_sci_hat_val)
            )
            p_nu_sur_hat_val = single_chain.p_nu_sur_hat_func(lmbda_nu_hat_val)
            p_nu_sur_hat_analytical_val = (
                single_chain.p_nu_sur_hat_analytical_func(lmbda_nu_hat_val)
            )
            p_nu_sur_hat_err_val = (
                np.abs(
                    (p_nu_sur_hat_analytical_val-p_nu_sur_hat_val)/(p_nu_sur_hat_val+single_chain.cond_val))
                * 100
            )
            p_nu_sur_hat_abs_diff_val = (
                np.abs(p_nu_sur_hat_analytical_val-p_nu_sur_hat_val)
            )
            
            if lmbda_nu_hat_indx == 0:
                pass
            else:
                epsilon_nu_diss_hat_val = (
                    single_chain.epsilon_nu_diss_hat_func(
                        lmbda_nu_hat_max_val, 
                        lmbda_nu_hat_max[lmbda_nu_hat_indx-1],
                        lmbda_nu_hat_val,
                        lmbda_nu_hat[lmbda_nu_hat_indx-1],
                        epsilon_nu_diss_hat[lmbda_nu_hat_indx-1])
                )
                epsilon_nu_diss_hat_analytical_val = (
                    single_chain.epsilon_nu_diss_hat_analytical_func(
                        lmbda_nu_hat_max_val, 
                        lmbda_nu_hat_max[lmbda_nu_hat_indx-1],
                        lmbda_nu_hat_val,
                        lmbda_nu_hat[lmbda_nu_hat_indx-1],
                        epsilon_nu_diss_hat_analytical[lmbda_nu_hat_indx-1])
                )
                epsilon_nu_diss_hat_err_val = (
                    np.abs(
                        (epsilon_nu_diss_hat_analytical_val-epsilon_nu_diss_hat_val)/(epsilon_nu_diss_hat_val+single_chain.cond_val))
                    * 100
                )
                epsilon_nu_diss_hat_abs_diff_val = (
                    np.abs(epsilon_nu_diss_hat_analytical_val-epsilon_nu_diss_hat_val)
                )
                epsilon_nu_diss_hat_equiv_val = (
                    single_chain.epsilon_nu_diss_hat_equiv_func(lmbda_nu_hat_val)
                )
                epsilon_nu_diss_hat_equiv_err_val = (
                    np.abs(
                        (epsilon_nu_diss_hat_equiv_val-epsilon_nu_diss_hat_val)/(epsilon_nu_diss_hat_val+single_chain.cond_val))
                    * 100
                )
                epsilon_nu_diss_hat_equiv_abs_diff_val = (
                    np.abs(epsilon_nu_diss_hat_equiv_val-epsilon_nu_diss_hat_val)
                )

            lmbda_nu_hat.append(lmbda_nu_hat_val)
            lmbda_nu_hat_max.append(lmbda_nu_hat_max_val)
            e_nu_sci_hat.append(e_nu_sci_hat_val)
            e_nu_sci_hat_analytical.append(e_nu_sci_hat_analytical_val)
            e_nu_sci_hat_err.append(e_nu_sci_hat_err_val)
            e_nu_sci_hat_abs_diff.append(e_nu_sci_hat_abs_diff_val)
            epsilon_nu_sci_hat.append(epsilon_nu_sci_hat_val)
            epsilon_nu_sci_hat_analytical.append(epsilon_nu_sci_hat_analytical_val)
            epsilon_nu_sci_hat_err.append(epsilon_nu_sci_hat_err_val)
            epsilon_nu_sci_hat_abs_diff.append(epsilon_nu_sci_hat_abs_diff_val)
            p_nu_sci_hat.append(p_nu_sci_hat_val)
            p_nu_sci_hat_analytical.append(p_nu_sci_hat_analytical_val)
            p_nu_sci_hat_err.append(p_nu_sci_hat_err_val)
            p_nu_sci_hat_abs_diff.append(p_nu_sci_hat_abs_diff_val)
            p_nu_sur_hat.append(p_nu_sur_hat_val)
            p_nu_sur_hat_analytical.append(p_nu_sur_hat_analytical_val)
            p_nu_sur_hat_err.append(p_nu_sur_hat_err_val)
            p_nu_sur_hat_abs_diff.append(p_nu_sur_hat_abs_diff_val)
            epsilon_nu_diss_hat.append(epsilon_nu_diss_hat_val)
            epsilon_nu_diss_hat_analytical.append(epsilon_nu_diss_hat_analytical_val)
            epsilon_nu_diss_hat_err.append(epsilon_nu_diss_hat_err_val)
            epsilon_nu_diss_hat_abs_diff.append(epsilon_nu_diss_hat_abs_diff_val)
            epsilon_nu_diss_hat_equiv.append(epsilon_nu_diss_hat_equiv_val)
            epsilon_nu_diss_hat_equiv_err.append(epsilon_nu_diss_hat_equiv_err_val)
            epsilon_nu_diss_hat_equiv_abs_diff.append(epsilon_nu_diss_hat_equiv_abs_diff_val)
        
        overline_e_nu_sci_hat = [
            e_nu_sci_hat_val/single_chain.zeta_nu_char
            for e_nu_sci_hat_val in e_nu_sci_hat
        ]
        overline_e_nu_sci_hat_analytical = [
            e_nu_sci_hat_analytical_val/single_chain.zeta_nu_char
            for e_nu_sci_hat_analytical_val in e_nu_sci_hat_analytical
        ]
        overline_e_nu_sci_hat_abs_diff = [
            e_nu_sci_hat_abs_diff_val/single_chain.zeta_nu_char
            for e_nu_sci_hat_abs_diff_val in e_nu_sci_hat_abs_diff
        ]
        overline_epsilon_nu_sci_hat = [
            epsilon_nu_sci_hat_val/single_chain.zeta_nu_char
            for epsilon_nu_sci_hat_val in epsilon_nu_sci_hat
        ]
        overline_epsilon_nu_sci_hat_analytical = [
            epsilon_nu_sci_hat_analytical_val/single_chain.zeta_nu_char
            for epsilon_nu_sci_hat_analytical_val
            in epsilon_nu_sci_hat_analytical
        ]
        overline_epsilon_nu_sci_hat_abs_diff = [
            epsilon_nu_sci_hat_abs_diff_val/single_chain.zeta_nu_char
            for epsilon_nu_sci_hat_abs_diff_val in epsilon_nu_sci_hat_abs_diff
        ]
        overline_epsilon_nu_diss_hat = [
            epsilon_nu_diss_hat_val/single_chain.zeta_nu_char
            for epsilon_nu_diss_hat_val in epsilon_nu_diss_hat
        ]
        overline_epsilon_nu_diss_hat_analytical = [
            epsilon_nu_diss_hat_analytical_val/single_chain.zeta_nu_char
            for epsilon_nu_diss_hat_analytical_val
            in epsilon_nu_diss_hat_analytical
        ]
        overline_epsilon_nu_diss_hat_abs_diff = [
            epsilon_nu_diss_hat_abs_diff_val/single_chain.zeta_nu_char
            for epsilon_nu_diss_hat_abs_diff_val in epsilon_nu_diss_hat_abs_diff
        ]
        overline_epsilon_nu_diss_hat_equiv = [
            epsilon_nu_diss_hat_equiv_val/single_chain.zeta_nu_char
            for epsilon_nu_diss_hat_equiv_val in epsilon_nu_diss_hat_equiv
        ]
        overline_epsilon_nu_diss_hat_equiv_abs_diff = [
            epsilon_nu_diss_hat_equiv_abs_diff_val/single_chain.zeta_nu_char
            for epsilon_nu_diss_hat_equiv_abs_diff_val
            in epsilon_nu_diss_hat_equiv_abs_diff
        ]
        
        self.single_chain = single_chain
        
        self.lmbda_nu_hat = lmbda_nu_hat
        self.overline_e_nu_sci_hat            = overline_e_nu_sci_hat
        self.overline_e_nu_sci_hat_analytical = overline_e_nu_sci_hat_analytical
        self.e_nu_sci_hat_err                 = e_nu_sci_hat_err
        self.overline_e_nu_sci_hat_abs_diff = overline_e_nu_sci_hat_abs_diff
        self.overline_epsilon_nu_sci_hat = (
            overline_epsilon_nu_sci_hat
        )
        self.overline_epsilon_nu_sci_hat_analytical = (
            overline_epsilon_nu_sci_hat_analytical
        )
        self.epsilon_nu_sci_hat_err = epsilon_nu_sci_hat_err
        self.overline_epsilon_nu_sci_hat_abs_diff = (
            overline_epsilon_nu_sci_hat_abs_diff
        )
        self.p_nu_sci_hat            = p_nu_sci_hat
        self.p_nu_sci_hat_analytical = p_nu_sci_hat_analytical
        self.p_nu_sci_hat_err        = p_nu_sci_hat_err
        self.p_nu_sci_hat_abs_diff   = p_nu_sci_hat_abs_diff
        self.p_nu_sur_hat            = p_nu_sur_hat
        self.p_nu_sur_hat_analytical = p_nu_sur_hat_analytical
        self.p_nu_sur_hat_err        = p_nu_sur_hat_err
        self.p_nu_sur_hat_abs_diff   = p_nu_sur_hat_abs_diff
        self.overline_epsilon_nu_diss_hat            = (
            overline_epsilon_nu_diss_hat
        )
        self.overline_epsilon_nu_diss_hat_analytical = (
            overline_epsilon_nu_diss_hat_analytical
        )
        self.epsilon_nu_diss_hat_err = epsilon_nu_diss_hat_err
        self.overline_epsilon_nu_diss_hat_abs_diff = (
            overline_epsilon_nu_diss_hat_abs_diff
        )
        self.overline_epsilon_nu_diss_hat_equiv = (
            overline_epsilon_nu_diss_hat_equiv
        )
        self.epsilon_nu_diss_hat_equiv_err = epsilon_nu_diss_hat_equiv_err
        self.overline_epsilon_nu_diss_hat_equiv_abs_diff = (
            overline_epsilon_nu_diss_hat_equiv_abs_diff
        )

    def finalization(self):
        """Define finalization analysis"""
        ppp = self.parameters.post_processing

        # plot results
        latex_formatting_figure(ppp)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        ax1.plot(
            self.lmbda_nu_hat, self.overline_e_nu_sci_hat, linestyle=':',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{e}_{\nu}^{sci}}$')
        ax1.plot(
            self.lmbda_nu_hat, self.overline_epsilon_nu_sci_hat, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{sci}}$')
        ax1.plot(
            self.lmbda_nu_hat, self.overline_epsilon_nu_diss_hat,
            linestyle=(0, (3, 1, 1, 1)), color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{diss}}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sci_hat, linestyle='-', color='blue',
            alpha=1, linewidth=2.5, label=r'$\hat{p}_{\nu}^{sci}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sur_hat, linestyle=':', color='blue',
            alpha=1, linewidth=2.5, label=r'$\hat{p}_{\nu}^{sur}$')

        ax1.legend(loc='best', fontsize=14, handlelength=3)
        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(axis='y', labelsize=16)
        ax1.set_ylabel(
            r'$\overline{\hat{e}_{\nu}^{sci}},~\overline{\hat{\varepsilon}_{\nu}^{sci}},~\overline{\hat{\varepsilon}_{\nu}^{diss}}$',
            fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.legend(loc='best', fontsize=14)
        ax2.set_ylim([-0.05, 1.05])
        ax2.tick_params(axis='y', labelsize=16)
        ax2.set_ylabel(
            r'$\hat{p}_{\nu}^{sci},~\hat{p}_{\nu}^{sur}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        
        plt.xlim([self.lmbda_nu_hat[0], self.lmbda_nu_hat[-1]])
        plt.xticks(fontsize=16)
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir,
            "rate-independent-segment-scission-indicators-vs-lmbda_nu_hat")
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        ax1.plot(
            self.lmbda_nu_hat, self.overline_e_nu_sci_hat_analytical, linestyle=':',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{e}_{\nu}^{sci}}$')
        ax1.plot(
            self.lmbda_nu_hat, self.overline_epsilon_nu_sci_hat_analytical, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{sci}}$')
        ax1.plot(
            self.lmbda_nu_hat, self.overline_epsilon_nu_diss_hat_analytical,
            linestyle=(0, (3, 1, 1, 1)), color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{diss}}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sci_hat_analytical, linestyle='-', color='blue',
            alpha=1, linewidth=2.5, label=r'$\hat{p}_{\nu}^{sci}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sur_hat_analytical, linestyle=':', color='blue',
            alpha=1, linewidth=2.5, label=r'$\hat{p}_{\nu}^{sur}$')

        ax1.legend(loc='best', fontsize=14, handlelength=3)
        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(axis='y', labelsize=16)
        ax1.set_ylabel(
            r'$\overline{\hat{e}_{\nu}^{sci}},~\overline{\hat{\varepsilon}_{\nu}^{sci}},~\overline{\hat{\varepsilon}_{\nu}^{diss}}$',
            fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.legend(loc='best', fontsize=14)
        ax2.set_ylim([-0.05, 1.05])
        ax2.tick_params(axis='y', labelsize=16)
        ax2.set_ylabel(
            r'$\hat{p}_{\nu}^{sci},~\hat{p}_{\nu}^{sur}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        
        plt.xlim([self.lmbda_nu_hat[0], self.lmbda_nu_hat[-1]])
        plt.xticks(fontsize=16)
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir,
            "analytical-rate-independent-segment-scission-indicators-vs-lmbda_nu_hat")
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
        ax1.plot(
            self.lmbda_nu_hat, self.e_nu_sci_hat_err, linestyle=':',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{e}_{\nu}^{sci}}$')
        ax1.plot(
            self.lmbda_nu_hat, self.epsilon_nu_sci_hat_err, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{sci}}$')
        ax1.plot(
            self.lmbda_nu_hat, self.epsilon_nu_diss_hat_err,
            linestyle=(0, (3, 1, 1, 1)), color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{diss}}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sci_hat_err, linestyle='-', color='blue',
            alpha=1, linewidth=2.5, label=r'$\hat{p}_{\nu}^{sci}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sur_hat_err, linestyle=':', color='blue',
            alpha=1, linewidth=2.5, label=r'$\hat{p}_{\nu}^{sur}$')

        ax1.legend(loc='best', fontsize=14, handlelength=3)
        ax1.tick_params(axis='y', labelsize=16)
        ax1.set_ylabel(r'$\%~\textrm{error}$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.legend(loc='best', fontsize=14)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        
        plt.xlim([self.lmbda_nu_hat[0], self.lmbda_nu_hat[-1]])
        plt.xticks(fontsize=16)
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir,
            "rate-independent-segment-scission-indicators-percent-error-vs-lmbda_nu_hat")
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
        ax1.plot(
            self.lmbda_nu_hat, self.overline_e_nu_sci_hat_abs_diff,
            linestyle=':', color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{e}_{\nu}^{sci}}$')
        ax1.plot(
            self.lmbda_nu_hat, self.overline_epsilon_nu_sci_hat_abs_diff, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{sci}}$')
        ax1.plot(
            self.lmbda_nu_hat, self.overline_epsilon_nu_diss_hat_abs_diff,
            linestyle=(0, (3, 1, 1, 1)), color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{diss}}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sci_hat_abs_diff, linestyle='-', color='blue',
            alpha=1, linewidth=2.5, label=r'$\hat{p}_{\nu}^{sci}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sur_hat_abs_diff, linestyle=':', color='blue',
            alpha=1, linewidth=2.5, label=r'$\hat{p}_{\nu}^{sur}$')

        ax1.legend(loc='best', fontsize=14, handlelength=3)
        ax1.tick_params(axis='y', labelsize=16)
        ax1.set_ylabel(r'$|\textrm{diff}|$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.legend(loc='best', fontsize=14)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.set_ylabel(r'$|\textrm{diff}|$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        
        plt.xlim([self.lmbda_nu_hat[0], self.lmbda_nu_hat[-1]])
        plt.xticks(fontsize=16)
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir,
            "rate-independent-segment-scission-indicators-absolute-difference-vs-lmbda_nu_hat")
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        ax1.plot(
            self.lmbda_nu_hat, self.overline_e_nu_sci_hat, linestyle=':',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{e}_{\nu}^{sci}}$')
        ax1.plot(
            self.lmbda_nu_hat, self.overline_epsilon_nu_sci_hat, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{sci}}$')
        ax1.plot(
            self.lmbda_nu_hat, self.overline_epsilon_nu_diss_hat_equiv,
            linestyle=(0, (3, 1, 1, 1)), color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{diss}}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sci_hat, linestyle='-', color='blue',
            alpha=1, linewidth=2.5, label=r'$\hat{p}_{\nu}^{sci}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sur_hat, linestyle=':', color='blue',
            alpha=1, linewidth=2.5, label=r'$\hat{p}_{\nu}^{sur}$')

        ax1.legend(loc='best', fontsize=14, handlelength=3)
        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(axis='y', labelsize=16)
        ax1.set_ylabel(
            r'$\overline{\hat{e}_{\nu}^{sci}},~\overline{\hat{\varepsilon}_{\nu}^{sci}},~\overline{\hat{\varepsilon}_{\nu}^{diss}}$',
            fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.legend(loc='best', fontsize=14)
        ax2.set_ylim([-0.05, 1.05])
        ax2.tick_params(axis='y', labelsize=16)
        ax2.set_ylabel(
            r'$\hat{p}_{\nu}^{sci},~\hat{p}_{\nu}^{sur}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        
        plt.xlim([self.lmbda_nu_hat[0], self.lmbda_nu_hat[-1]])
        plt.xticks(fontsize=16)
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir,
            "equivalent-rate-independent-dissipated-segment-scission-vs-lmbda_nu_hat")
        
        fig = plt.figure()
        plt.plot(self.lmbda_nu_hat, self.epsilon_nu_diss_hat_equiv_err,
                 linestyle='-', color='blue', alpha=1, linewidth=2.5,
                 label=r'$\overline{\hat{\varepsilon}_{\nu}^{diss}}$')
        plt.legend(loc='best', fontsize=14, handlelength=3)
        plt.xlim([self.lmbda_nu_hat[0], self.lmbda_nu_hat[-1]])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\hat{\lambda}_{\nu}$', 30, r'$\%~\textrm{error}$', 30,
            "equivalent-rate-independent-dissipated-segment-scission-percent-error-vs-lmbda_nu_hat")
        
        fig = plt.figure()
        plt.plot(self.lmbda_nu_hat, self.overline_epsilon_nu_diss_hat_equiv_abs_diff,
                 linestyle='-', color='blue', alpha=1, linewidth=2.5,
                 label=r'$\overline{\hat{\varepsilon}_{\nu}^{diss}}$')
        plt.legend(loc='best', fontsize=14, handlelength=3)
        plt.xlim([self.lmbda_nu_hat[0], self.lmbda_nu_hat[-1]])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\hat{\lambda}_{\nu}$', 30, r'$|\textrm{diff}|$', 30,
            "equivalent-rate-independent-dissipated-segment-scission-absolute-difference-vs-lmbda_nu_hat")

if __name__ == '__main__':

    characterizer = RateIndependentSegmentScissionCharacterizer()
    characterizer.characterization()
    characterizer.finalization()