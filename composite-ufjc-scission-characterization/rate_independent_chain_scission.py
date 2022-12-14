"""The rate-independent chain scission characterization module for
composite uFJCs
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
import matplotlib.pyplot as plt


class RateIndependentChainScissionCharacterizer(
        CompositeuFJCScissionCharacterizer):
    """The characterization class assessing the rate-independent chain
    scission for composite uFJCs. It inherits all attributes and methods
    from the ``CompositeuFJCScissionCharacterizer`` class.
    """
    def __init__(self):
        """Initializes the
        ``RateIndependentChainScissionCharacterizer`` class by
        initializing and inheriting all attributes and methods from the
        ``CompositeuFJCScissionCharacterizer`` class.
        """
        CompositeuFJCScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        p = self.parameters

        p.characterizer.lmbda_nu_inc     = 0.0001
        p.characterizer.lmbda_nu_min     = 0.50
        p.characterizer.lmbda_nu_max     = 2.50
        p.characterizer.lmbda_nu_hat_num = 4

        p.characterizer.color_list = [
            'orange', 'blue', 'green', 'red', 'purple'
        ]
    
    def prefix(self):
        """Set characterization prefix"""
        return "rate_independent_chain_scission"
    
    def characterization(self):
        """Define characterization routine"""
        cp = self.parameters.characterizer

        # zeta_nu_char=100 and kappa_nu=1000
        single_chain_list = [
            RateIndependentScissionCompositeuFJC(
                nu=cp.nu_single_chain_list[single_chain_indx],
                zeta_nu_char=cp.zeta_nu_char_single_chain_list[2],
                kappa_nu=cp.kappa_nu_single_chain_list[2])
            for single_chain_indx in range(len(cp.nu_single_chain_list))
        ]
        
        lmbda_nu_hat___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        p_c_sur_hat___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        p_c_sur_hat_analytical___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        p_c_sur_hat_err___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        p_c_sci_hat___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        p_c_sci_hat_analytical___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        p_c_sci_hat_err___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        epsilon_cnu_sci_hat___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        epsilon_cnu_sci_hat_analytical___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        epsilon_cnu_sci_hat_err___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        epsilon_cnu_diss_hat___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        epsilon_cnu_diss_hat_analytical___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        epsilon_cnu_diss_hat_err___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        overline_epsilon_cnu_sci_hat___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        overline_epsilon_cnu_sci_hat_analytical___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        overline_epsilon_cnu_diss_hat___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        overline_epsilon_cnu_diss_hat_analytical___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        nu___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]

        for single_chain_indx in range(len(single_chain_list)):
            single_chain = single_chain_list[single_chain_indx]
            nu           = single_chain.nu
            
            # Define the applied segment stretch values to
            # calculate over
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
            p_c_sur_hat            = []
            p_c_sur_hat_analytical = []
            p_c_sur_hat_err        = []
            p_c_sci_hat            = []
            p_c_sci_hat_analytical = []
            p_c_sci_hat_err        = []
            epsilon_cnu_sci_hat            = []
            epsilon_cnu_sci_hat_analytical = []
            epsilon_cnu_sci_hat_err        = []
            epsilon_cnu_diss_hat            = []
            epsilon_cnu_diss_hat_analytical = []
            epsilon_cnu_diss_hat_err        = []

            # initialization
            lmbda_nu_hat_max_val = 0.
            epsilon_cnu_diss_hat_val            = 0.
            epsilon_cnu_diss_hat_analytical_val = 0.
            epsilon_cnu_diss_hat_err_val        = 0.
            
            # Calculate results through specified applied segment
            # stretch values
            for lmbda_nu_hat_indx in range(lmbda_nu_hat_num_steps):
                lmbda_nu_hat_val = lmbda_nu_hat_steps[lmbda_nu_hat_indx]
                lmbda_nu_hat_max_val = (
                    max([lmbda_nu_hat_max_val, lmbda_nu_hat_val])
                )
                p_c_sur_hat_val = (
                    single_chain.p_c_sur_hat_func(lmbda_nu_hat_val)
                )
                p_c_sur_hat_analytical_val = (
                single_chain.p_c_sur_hat_analytical_func(lmbda_nu_hat_val)
                )
                p_c_sur_hat_err_val = (
                    np.abs(
                        (p_c_sur_hat_analytical_val-p_c_sur_hat_val)/(p_c_sur_hat_val+single_chain.cond_val))
                    * 100
                )
                p_c_sci_hat_val = (
                    single_chain.p_c_sci_hat_func(lmbda_nu_hat_val)
                )
                p_c_sci_hat_analytical_val = (
                single_chain.p_c_sci_hat_analytical_func(lmbda_nu_hat_val)
                )
                p_c_sci_hat_err_val = (
                    np.abs(
                        (p_c_sci_hat_analytical_val-p_c_sci_hat_val)/(p_c_sci_hat_val+single_chain.cond_val))
                    * 100
                )
                epsilon_cnu_sci_hat_val = (
                    single_chain.epsilon_cnu_sci_hat_func(lmbda_nu_hat_val)
                )
                epsilon_cnu_sci_hat_analytical_val = (
                single_chain.epsilon_cnu_sci_hat_analytical_func(lmbda_nu_hat_val)
                )
                epsilon_cnu_sci_hat_err_val = (
                    np.abs(
                        (epsilon_cnu_sci_hat_analytical_val-epsilon_cnu_sci_hat_val)/(epsilon_cnu_sci_hat_val+single_chain.cond_val))
                    * 100
                )

                # initialization
                if lmbda_nu_hat_indx == 0:
                    pass
                else:
                    epsilon_cnu_diss_hat_val = (
                        single_chain.epsilon_cnu_diss_hat_func(
                            lmbda_nu_hat_max_val, 
                            lmbda_nu_hat_max[lmbda_nu_hat_indx-1],
                            lmbda_nu_hat_val,
                            lmbda_nu_hat[lmbda_nu_hat_indx-1],
                            epsilon_cnu_diss_hat[lmbda_nu_hat_indx-1])
                    )
                    epsilon_cnu_diss_hat_analytical_val = (
                    single_chain.epsilon_cnu_diss_hat_analytical_func(
                        lmbda_nu_hat_max_val, 
                        lmbda_nu_hat_max[lmbda_nu_hat_indx-1],
                        lmbda_nu_hat_val,
                        lmbda_nu_hat[lmbda_nu_hat_indx-1],
                        epsilon_cnu_diss_hat_analytical[lmbda_nu_hat_indx-1])
                    )
                    epsilon_cnu_diss_hat_err_val = (
                        np.abs(
                            (epsilon_cnu_diss_hat_analytical_val-epsilon_cnu_diss_hat_val)/(epsilon_cnu_diss_hat_val+single_chain.cond_val))
                        * 100
                    )
                
                lmbda_nu_hat.append(lmbda_nu_hat_val)
                lmbda_nu_hat_max.append(lmbda_nu_hat_max_val)
                p_c_sur_hat.append(p_c_sur_hat_val)
                p_c_sur_hat_analytical.append(p_c_sur_hat_analytical_val)
                p_c_sur_hat_err.append(p_c_sur_hat_err_val)
                p_c_sci_hat.append(p_c_sci_hat_val)
                p_c_sci_hat_analytical.append(p_c_sci_hat_analytical_val)
                p_c_sci_hat_err.append(p_c_sci_hat_err_val)
                epsilon_cnu_sci_hat.append(epsilon_cnu_sci_hat_val)
                epsilon_cnu_sci_hat_analytical.append(epsilon_cnu_sci_hat_analytical_val)
                epsilon_cnu_sci_hat_err.append(epsilon_cnu_sci_hat_err_val)
                epsilon_cnu_diss_hat.append(epsilon_cnu_diss_hat_val)
                epsilon_cnu_diss_hat_analytical.append(epsilon_cnu_diss_hat_analytical_val)
                epsilon_cnu_diss_hat_err.append(epsilon_cnu_diss_hat_err_val)
            
            overline_epsilon_cnu_sci_hat = [
                epsilon_cnu_sci_hat_val/single_chain.zeta_nu_char
                for epsilon_cnu_sci_hat_val in epsilon_cnu_sci_hat
            ]
            overline_epsilon_cnu_sci_hat_analytical = [
                epsilon_cnu_sci_hat_analytical_val/single_chain.zeta_nu_char
                for epsilon_cnu_sci_hat_analytical_val
                in epsilon_cnu_sci_hat_analytical
            ]
            overline_epsilon_cnu_diss_hat = [
                epsilon_cnu_diss_hat_val/single_chain.zeta_nu_char
                for epsilon_cnu_diss_hat_val in epsilon_cnu_diss_hat
            ]
            overline_epsilon_cnu_diss_hat_analytical = [
                epsilon_cnu_diss_hat_analytical_val/single_chain.zeta_nu_char
                for epsilon_cnu_diss_hat_analytical_val
                in epsilon_cnu_diss_hat_analytical
            ]
            
            lmbda_nu_hat___single_chain_chunk[single_chain_indx] = lmbda_nu_hat
            p_c_sur_hat___single_chain_chunk[single_chain_indx] = p_c_sur_hat
            p_c_sur_hat_analytical___single_chain_chunk[single_chain_indx] = (
                p_c_sur_hat_analytical
            )
            p_c_sur_hat_err___single_chain_chunk[single_chain_indx] = (
                p_c_sur_hat_err
            )
            p_c_sci_hat___single_chain_chunk[single_chain_indx] = p_c_sci_hat
            p_c_sci_hat_analytical___single_chain_chunk[single_chain_indx] = (
                p_c_sci_hat_analytical
            )
            p_c_sci_hat_err___single_chain_chunk[single_chain_indx] = (
                p_c_sci_hat_err
            )
            epsilon_cnu_sci_hat___single_chain_chunk[single_chain_indx] = (
                epsilon_cnu_sci_hat
            )
            epsilon_cnu_sci_hat_analytical___single_chain_chunk[single_chain_indx] = (
                epsilon_cnu_sci_hat_analytical
            )
            epsilon_cnu_sci_hat_err___single_chain_chunk[single_chain_indx] = (
                epsilon_cnu_sci_hat_err
            )
            epsilon_cnu_diss_hat___single_chain_chunk[single_chain_indx] = (
                epsilon_cnu_diss_hat
            )
            epsilon_cnu_diss_hat_analytical___single_chain_chunk[single_chain_indx] = (
                epsilon_cnu_diss_hat_analytical
            )
            epsilon_cnu_diss_hat_err___single_chain_chunk[single_chain_indx] = (
                epsilon_cnu_diss_hat_err
            )
            overline_epsilon_cnu_sci_hat___single_chain_chunk[single_chain_indx] = (
                overline_epsilon_cnu_sci_hat
            )
            overline_epsilon_cnu_sci_hat_analytical___single_chain_chunk[single_chain_indx] = (
                overline_epsilon_cnu_sci_hat_analytical
            )
            overline_epsilon_cnu_diss_hat___single_chain_chunk[single_chain_indx] = (
                overline_epsilon_cnu_diss_hat
            )
            overline_epsilon_cnu_diss_hat_analytical___single_chain_chunk[single_chain_indx] = (
                overline_epsilon_cnu_diss_hat_analytical
            )
            nu___single_chain_chunk[single_chain_indx] = nu
        
        self.single_chain_list = single_chain_list
        
        self.lmbda_nu_hat___single_chain_chunk = (
            lmbda_nu_hat___single_chain_chunk
        )
        self.p_c_sur_hat___single_chain_chunk = (
            p_c_sur_hat___single_chain_chunk
        )
        self.p_c_sur_hat_analytical___single_chain_chunk = (
            p_c_sur_hat_analytical___single_chain_chunk
        )
        self.p_c_sur_hat_err___single_chain_chunk = (
            p_c_sur_hat_err___single_chain_chunk
        )
        self.p_c_sci_hat___single_chain_chunk = (
            p_c_sci_hat___single_chain_chunk
        )
        self.p_c_sci_hat_analytical___single_chain_chunk = (
            p_c_sci_hat_analytical___single_chain_chunk
        )
        self.p_c_sci_hat_err___single_chain_chunk = (
            p_c_sci_hat_err___single_chain_chunk
        )
        self.epsilon_cnu_sci_hat___single_chain_chunk = (
            epsilon_cnu_sci_hat___single_chain_chunk
        )
        self.epsilon_cnu_sci_hat_analytical___single_chain_chunk = (
            epsilon_cnu_sci_hat_analytical___single_chain_chunk
        )
        self.epsilon_cnu_sci_hat_err___single_chain_chunk = (
            epsilon_cnu_sci_hat_err___single_chain_chunk
        )
        self.epsilon_cnu_diss_hat___single_chain_chunk = (
            epsilon_cnu_diss_hat___single_chain_chunk
        )
        self.epsilon_cnu_diss_hat_analytical___single_chain_chunk = (
            epsilon_cnu_diss_hat_analytical___single_chain_chunk
        )
        self.epsilon_cnu_diss_hat_err___single_chain_chunk = (
            epsilon_cnu_diss_hat_err___single_chain_chunk
        )
        self.overline_epsilon_cnu_sci_hat___single_chain_chunk = (
            overline_epsilon_cnu_sci_hat___single_chain_chunk
        )
        self.overline_epsilon_cnu_sci_hat_analytical___single_chain_chunk = (
            overline_epsilon_cnu_sci_hat_analytical___single_chain_chunk
        )
        self.overline_epsilon_cnu_diss_hat___single_chain_chunk = (
            overline_epsilon_cnu_diss_hat___single_chain_chunk
        )
        self.overline_epsilon_cnu_diss_hat_analytical___single_chain_chunk = (
            overline_epsilon_cnu_diss_hat_analytical___single_chain_chunk
        )
        self.nu___single_chain_chunk = nu___single_chain_chunk

    def finalization(self):
        """Define finalization analysis"""
        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing

        # plot results
        latex_formatting_figure(ppp)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_sci_hat = (
                self.overline_epsilon_cnu_sci_hat___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_diss_hat = (
                self.overline_epsilon_cnu_diss_hat___single_chain_chunk[single_chain_indx]
            )
            ax1.plot(
                lmbda_nu_hat, overline_epsilon_cnu_sci_hat, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5,
                label=cp.nu_label_single_chain_list[single_chain_indx])
            ax1.plot(
                lmbda_nu_hat, overline_epsilon_cnu_diss_hat,
                linestyle=(0, (3, 1, 1, 1)),
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
        ax1.plot(
            lmbda_nu_hat, overline_epsilon_cnu_sci_hat, linestyle='-',
            color='black', alpha=1, linewidth=2.5)
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            p_c_sur_hat  = (
                self.p_c_sur_hat___single_chain_chunk[single_chain_indx]
            )
            p_c_sci_hat  = (
                self.p_c_sci_hat___single_chain_chunk[single_chain_indx]
            )
            ax2.plot(
                lmbda_nu_hat, p_c_sci_hat, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
            ax2.plot(
                lmbda_nu_hat, p_c_sur_hat, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
        ax2.plot(
            [], [], linestyle='-', color='black', label=r'$\hat{p}_c^{sci}$')
        ax2.plot(
            [], [], linestyle=':', color='black', label=r'$\hat{p}_c^{sur}$')

        ax1.legend(loc='best', fontsize=18)
        ax1.set_ylim([-0.05, 0.7])
        ax1.tick_params(axis='y', labelsize=20)
        ax1.set_ylabel(
            r'$\overline{\hat{\varepsilon}_{c\nu}^{sci}}~(-),~\overline{\hat{\varepsilon}_{c\nu}^{diss}}~(-~\cdot)$',
            fontsize=21)
        ax1.grid(True, alpha=0.25)
        ax2.legend(loc='best', fontsize=18)
        ax2.set_ylim([-0.05, 1.05])
        ax2.tick_params(axis='y', labelsize=20)
        ax2.set_ylabel(r'$\hat{p}_c^{sur},~\hat{p}_c^{sci}$', fontsize=21)
        ax2.grid(True, alpha=0.25)
        plt.xlim([lmbda_nu_hat[0], lmbda_nu_hat[-1]])
        plt.xticks(fontsize=20)
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=30)
        save_current_figure_no_labels(
            self.savedir,
            "rate-independent-chain-scission-indicators-vs-lmbda_nu_hat")
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_sci_hat_analytical = (
                self.overline_epsilon_cnu_sci_hat_analytical___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_diss_hat_analytical = (
                self.overline_epsilon_cnu_diss_hat_analytical___single_chain_chunk[single_chain_indx]
            )
            ax1.plot(
                lmbda_nu_hat, overline_epsilon_cnu_sci_hat_analytical, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5,
                label=cp.nu_label_single_chain_list[single_chain_indx])
            ax1.plot(
                lmbda_nu_hat, overline_epsilon_cnu_diss_hat_analytical,
                linestyle=(0, (3, 1, 1, 1)),
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
        ax1.plot(
            lmbda_nu_hat, overline_epsilon_cnu_sci_hat_analytical, linestyle='-',
            color='black', alpha=1, linewidth=2.5)
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            p_c_sur_hat_analytical  = (
                self.p_c_sur_hat_analytical___single_chain_chunk[single_chain_indx]
            )
            p_c_sci_hat_analytical  = (
                self.p_c_sci_hat_analytical___single_chain_chunk[single_chain_indx]
            )
            ax2.plot(
                lmbda_nu_hat, p_c_sci_hat_analytical, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
            ax2.plot(
                lmbda_nu_hat, p_c_sur_hat_analytical, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
        ax2.plot(
            [], [], linestyle='-', color='black', label=r'$\hat{p}_c^{sci}$')
        ax2.plot(
            [], [], linestyle=':', color='black', label=r'$\hat{p}_c^{sur}$')

        ax1.legend(loc='best', fontsize=18)
        ax1.set_ylim([-0.05, 0.7])
        ax1.tick_params(axis='y', labelsize=20)
        ax1.set_ylabel(
            r'$\overline{\hat{\varepsilon}_{c\nu}^{sci}}~(-),~\overline{\hat{\varepsilon}_{c\nu}^{diss}}~(-~\cdot)$',
            fontsize=21)
        ax1.grid(True, alpha=0.25)
        ax2.legend(loc='best', fontsize=18)
        ax2.set_ylim([-0.05, 1.05])
        ax2.tick_params(axis='y', labelsize=20)
        ax2.set_ylabel(r'$\hat{p}_c^{sur},~\hat{p}_c^{sci}$', fontsize=21)
        ax2.grid(True, alpha=0.25)
        plt.xlim([lmbda_nu_hat[0], lmbda_nu_hat[-1]])
        plt.xticks(fontsize=20)
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=30)
        save_current_figure_no_labels(
            self.savedir,
            "analytical-rate-independent-chain-scission-indicators-vs-lmbda_nu_hat")
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            epsilon_cnu_sci_hat_err = (
                self.epsilon_cnu_sci_hat_err___single_chain_chunk[single_chain_indx]
            )
            epsilon_cnu_diss_hat_err = (
                self.epsilon_cnu_diss_hat_err___single_chain_chunk[single_chain_indx]
            )
            ax1.plot(
                lmbda_nu_hat, epsilon_cnu_sci_hat_err, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5,
                label=cp.nu_label_single_chain_list[single_chain_indx])
            ax1.plot(
                lmbda_nu_hat, epsilon_cnu_diss_hat_err,
                linestyle=(0, (3, 1, 1, 1)),
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            p_c_sur_hat_err  = (
                self.p_c_sur_hat_err___single_chain_chunk[single_chain_indx]
            )
            p_c_sci_hat_err  = (
                self.p_c_sci_hat_err___single_chain_chunk[single_chain_indx]
            )
            ax2.plot(
                lmbda_nu_hat, p_c_sci_hat_err, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
            ax2.plot(
                lmbda_nu_hat, p_c_sur_hat_err, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
        ax2.plot(
            [], [], linestyle='-', color='black', label=r'$\hat{p}_c^{sci}$')
        ax2.plot(
            [], [], linestyle=':', color='black', label=r'$\hat{p}_c^{sur}$')

        ax1.legend(loc='best', fontsize=18)
        # ax1.set_ylim([-0.05, 0.7])
        ax1.tick_params(axis='y', labelsize=20)
        ax1.set_ylabel(r'$\%~\textrm{error}$', fontsize=21)
        ax1.grid(True, alpha=0.25)
        ax2.legend(loc='best', fontsize=18)
        # ax2.set_ylim([-0.05, 1.05])
        ax2.tick_params(axis='y', labelsize=20)
        ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=21)
        ax2.grid(True, alpha=0.25)
        plt.xlim([lmbda_nu_hat[0], lmbda_nu_hat[-1]])
        plt.xticks(fontsize=20)
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=30)
        save_current_figure_no_labels(
            self.savedir,
            "rate-independent-chain-scission-indicators-percent-error-vs-lmbda_nu_hat")
        
        fig = plt.figure()
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_sci_hat = (
                self.overline_epsilon_cnu_sci_hat___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_diss_hat = (
                self.overline_epsilon_cnu_diss_hat___single_chain_chunk[single_chain_indx]
            )
            nu = self.nu___single_chain_chunk[single_chain_indx]
            overline_epsilon_c_sci_hat = [
                x*nu for x in overline_epsilon_cnu_sci_hat
            ]
            overline_epsilon_c_diss_hat = [
                x*nu for x in overline_epsilon_cnu_diss_hat
            ]
            plt.semilogy(
                lmbda_nu_hat, overline_epsilon_c_sci_hat, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5,
                label=cp.nu_label_single_chain_list[single_chain_indx])
            plt.semilogy(
                lmbda_nu_hat, overline_epsilon_c_diss_hat,
                linestyle=(0, (3, 1, 1, 1)),
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
        plt.legend(loc='best', fontsize=18)
        plt.xlim([lmbda_nu_hat[0], lmbda_nu_hat[-1]])
        plt.xticks(fontsize=20)
        plt.ylim([1e-5, 1e4])
        plt.yticks(fontsize=20)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\hat{\lambda}_{\nu}$', 30,
            r'$\overline{\hat{\varepsilon}_c^{sci}}~(-),~\overline{\hat{\varepsilon}_c^{diss}}~(-~\cdot)$',
            30, "rate-independent-chain-scission-energy-vs-lmbda_nu_hat")

if __name__ == '__main__':

    characterizer = RateIndependentChainScissionCharacterizer()
    characterizer.characterization()
    characterizer.finalization()