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
        p_c_sur_hat_abs_diff___single_chain_chunk = [
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
        p_c_sci_hat_abs_diff___single_chain_chunk = [
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
        epsilon_cnu_diss_hat_equiv___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        epsilon_cnu_diss_hat_equiv_err___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        overline_epsilon_cnu_sci_hat___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        overline_epsilon_cnu_sci_hat_analytical___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        overline_epsilon_cnu_sci_hat_abs_diff___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        overline_epsilon_cnu_diss_hat___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        overline_epsilon_cnu_diss_hat_analytical___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        overline_epsilon_cnu_diss_hat_abs_diff___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        overline_epsilon_cnu_diss_hat_equiv___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        overline_epsilon_cnu_diss_hat_equiv_abs_diff___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        u_nu_hat_p_c_sur_hat___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        u_nu_hat_p_c_sur_hat_max___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        s_cnu_hat_p_c_sur_hat___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        s_cnu_hat_p_c_sur_hat_max___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        psi_cnu_hat_p_c_sur_hat___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        psi_cnu_hat_p_c_sur_hat_max___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        xi_c_hat_p_c_sur_hat___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        xi_c_hat_p_c_sur_hat_max___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        lmbda_nu_hat_p_c_sci_hat_rms___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        lmbda_nu_hat_p_c_sci_hat_mean___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        p_c_sci_hat_p_c_sci_hat_rms___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        p_c_sci_hat_p_c_sci_hat_mean___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms___single_chain_chunk = [
            0. for single_chain_indx in range(len(single_chain_list))
        ]
        overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean___single_chain_chunk = [
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
            p_c_sur_hat_abs_diff   = []
            p_c_sci_hat            = []
            p_c_sci_hat_analytical = []
            p_c_sci_hat_err        = []
            p_c_sci_hat_abs_diff   = []
            epsilon_cnu_sci_hat            = []
            epsilon_cnu_sci_hat_analytical = []
            epsilon_cnu_sci_hat_err        = []
            epsilon_cnu_sci_hat_abs_diff   = []
            epsilon_cnu_diss_hat            = []
            epsilon_cnu_diss_hat_analytical = []
            epsilon_cnu_diss_hat_err        = []
            epsilon_cnu_diss_hat_abs_diff   = []
            epsilon_cnu_diss_hat_equiv      = []
            epsilon_cnu_diss_hat_equiv_err  = []
            epsilon_cnu_diss_hat_equiv_abs_diff = []
            u_nu_hat_p_c_sur_hat    = []
            s_cnu_hat_p_c_sur_hat   = []
            psi_cnu_hat_p_c_sur_hat = []
            xi_c_hat_p_c_sur_hat    = []

            # initialization
            lmbda_nu_hat_max_val = 0.
            epsilon_cnu_diss_hat_val            = 0.
            epsilon_cnu_diss_hat_analytical_val = 0.
            epsilon_cnu_diss_hat_err_val        = 0.
            epsilon_cnu_diss_hat_abs_diff_val   = 0.
            epsilon_cnu_diss_hat_equiv_val      = 0.
            epsilon_cnu_diss_hat_equiv_err_val  = 0.
            epsilon_cnu_diss_hat_equiv_abs_diff_val = 0.
            u_nu_hat_p_c_sur_hat_val    = 0.
            s_cnu_hat_p_c_sur_hat_val   = 0.
            psi_cnu_hat_p_c_sur_hat_val = 0.
            xi_c_hat_p_c_sur_hat_val    = 0.
            
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
                p_c_sur_hat_abs_diff_val = (
                    np.abs(p_c_sur_hat_analytical_val-p_c_sur_hat_val)
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
                p_c_sci_hat_abs_diff_val = (
                    np.abs(p_c_sci_hat_analytical_val-p_c_sci_hat_val)
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
                epsilon_cnu_sci_hat_abs_diff_val = (
                    np.abs(epsilon_cnu_sci_hat_analytical_val-epsilon_cnu_sci_hat_val)
                )
                lmbda_c_eq_hat_val = (
                    single_chain.lmbda_c_eq_func(lmbda_nu_hat_val)
                )
                lmbda_comp_nu_hat_val = (
                    lmbda_c_eq_hat_val - lmbda_nu_hat_val + 1.
                )
                u_nu_hat_val = single_chain.u_nu_func(lmbda_nu_hat_val)
                s_cnu_hat_val = single_chain.s_cnu_func(lmbda_comp_nu_hat_val)
                psi_cnu_hat_val = (
                    single_chain.psi_cnu_func(lmbda_nu_hat_val, lmbda_c_eq_hat_val)
                )
                xi_c_hat_val = (
                    single_chain.xi_c_func(lmbda_nu_hat_val, lmbda_c_eq_hat_val)
                )

                # initialization
                if lmbda_nu_hat_indx == 0:
                    u_nu_hat_init_val = u_nu_hat_val
                    s_cnu_hat_init_val = s_cnu_hat_val
                    psi_cnu_hat_init_val = psi_cnu_hat_val
                    xi_c_hat_init_val = xi_c_hat_val
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
                    epsilon_cnu_diss_hat_abs_diff_val = (
                        np.abs(epsilon_cnu_diss_hat_analytical_val-epsilon_cnu_diss_hat_val)
                    )
                    epsilon_cnu_diss_hat_equiv_val = (
                        single_chain.epsilon_cnu_diss_hat_equiv_func(lmbda_nu_hat_val)
                    )
                    epsilon_cnu_diss_hat_equiv_err_val = (
                        np.abs(
                            (epsilon_cnu_diss_hat_equiv_val-epsilon_cnu_diss_hat_val)/(epsilon_cnu_diss_hat_val+single_chain.cond_val))
                        * 100
                    )
                    epsilon_cnu_diss_hat_equiv_abs_diff_val = (
                        np.abs(epsilon_cnu_diss_hat_equiv_val-epsilon_cnu_diss_hat_val)
                    )
                    u_nu_hat_p_c_sur_hat_val = (
                        p_c_sur_hat_val * (u_nu_hat_val-u_nu_hat_init_val)
                    )
                    s_cnu_hat_p_c_sur_hat_val = (
                        p_c_sur_hat_val * (s_cnu_hat_val-s_cnu_hat_init_val)
                    )
                    psi_cnu_hat_p_c_sur_hat_val = (
                        p_c_sur_hat_val * (psi_cnu_hat_val-psi_cnu_hat_init_val)
                    )
                    xi_c_hat_p_c_sur_hat_val = (
                        p_c_sur_hat_val * (xi_c_hat_val-xi_c_hat_init_val)
                    )
                
                lmbda_nu_hat.append(lmbda_nu_hat_val)
                lmbda_nu_hat_max.append(lmbda_nu_hat_max_val)
                p_c_sur_hat.append(p_c_sur_hat_val)
                p_c_sur_hat_analytical.append(p_c_sur_hat_analytical_val)
                p_c_sur_hat_err.append(p_c_sur_hat_err_val)
                p_c_sur_hat_abs_diff.append(p_c_sur_hat_abs_diff_val)
                p_c_sci_hat.append(p_c_sci_hat_val)
                p_c_sci_hat_analytical.append(p_c_sci_hat_analytical_val)
                p_c_sci_hat_err.append(p_c_sci_hat_err_val)
                p_c_sci_hat_abs_diff.append(p_c_sci_hat_abs_diff_val)
                epsilon_cnu_sci_hat.append(epsilon_cnu_sci_hat_val)
                epsilon_cnu_sci_hat_analytical.append(epsilon_cnu_sci_hat_analytical_val)
                epsilon_cnu_sci_hat_err.append(epsilon_cnu_sci_hat_err_val)
                epsilon_cnu_sci_hat_abs_diff.append(epsilon_cnu_sci_hat_abs_diff_val)
                epsilon_cnu_diss_hat.append(epsilon_cnu_diss_hat_val)
                epsilon_cnu_diss_hat_analytical.append(epsilon_cnu_diss_hat_analytical_val)
                epsilon_cnu_diss_hat_err.append(epsilon_cnu_diss_hat_err_val)
                epsilon_cnu_diss_hat_abs_diff.append(epsilon_cnu_diss_hat_abs_diff_val)
                epsilon_cnu_diss_hat_equiv.append(epsilon_cnu_diss_hat_equiv_val)
                epsilon_cnu_diss_hat_equiv_err.append(epsilon_cnu_diss_hat_equiv_err_val)
                epsilon_cnu_diss_hat_equiv_abs_diff.append(epsilon_cnu_diss_hat_equiv_abs_diff_val)
                u_nu_hat_p_c_sur_hat.append(u_nu_hat_p_c_sur_hat_val)
                s_cnu_hat_p_c_sur_hat.append(s_cnu_hat_p_c_sur_hat_val)
                psi_cnu_hat_p_c_sur_hat.append(psi_cnu_hat_p_c_sur_hat_val)
                xi_c_hat_p_c_sur_hat.append(xi_c_hat_p_c_sur_hat_val)
            
            indx_u_nu_hat_p_c_sur_hat_max = (
                np.argmax(np.asarray(u_nu_hat_p_c_sur_hat))
            )
            u_nu_hat_p_c_sur_hat_max = (
                u_nu_hat_p_c_sur_hat[indx_u_nu_hat_p_c_sur_hat_max]
            )
            lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max = (
                lmbda_nu_hat[indx_u_nu_hat_p_c_sur_hat_max]
            )
            indx_s_cnu_hat_p_c_sur_hat_max = (
                np.argmax(np.asarray(s_cnu_hat_p_c_sur_hat))
            )
            s_cnu_hat_p_c_sur_hat_max = (
                s_cnu_hat_p_c_sur_hat[indx_s_cnu_hat_p_c_sur_hat_max]
            )
            lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max = (
                lmbda_nu_hat[indx_s_cnu_hat_p_c_sur_hat_max]
            )
            indx_psi_cnu_hat_p_c_sur_hat_max = (
                np.argmax(np.asarray(psi_cnu_hat_p_c_sur_hat))
            )
            psi_cnu_hat_p_c_sur_hat_max = (
                psi_cnu_hat_p_c_sur_hat[indx_psi_cnu_hat_p_c_sur_hat_max]
            )
            lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max = (
                lmbda_nu_hat[indx_psi_cnu_hat_p_c_sur_hat_max]
            )
            indx_xi_c_hat_p_c_sur_hat_max = (
                np.argmax(np.asarray(xi_c_hat_p_c_sur_hat))
            )
            xi_c_hat_p_c_sur_hat_max = (
                xi_c_hat_p_c_sur_hat[indx_xi_c_hat_p_c_sur_hat_max]
            )
            lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max = (
                lmbda_nu_hat[indx_xi_c_hat_p_c_sur_hat_max]
            )
            
            lmbda_nu_hat_arr = np.asarray(lmbda_nu_hat)
            p_c_sci_hat_arr = np.asarray(p_c_sci_hat)

            partial_p_c_sci_hat__partial_lmbda_nu_hat_arr = (
                np.gradient(p_c_sci_hat_arr, lmbda_nu_hat_arr, edge_order=2)
            )

            Z = (
                np.trapz(partial_p_c_sci_hat__partial_lmbda_nu_hat_arr, lmbda_nu_hat_arr)
            )

            lmbda_nu_hat_p_c_sci_hat_rms_intrgrnd_arr = (
                partial_p_c_sci_hat__partial_lmbda_nu_hat_arr * lmbda_nu_hat_arr**2
            )
            lmbda_nu_hat_p_c_sci_hat_mean_intrgrnd_arr = (
                partial_p_c_sci_hat__partial_lmbda_nu_hat_arr * lmbda_nu_hat_arr
            )

            lmbda_nu_hat_p_c_sci_hat_rms = (
                np.sqrt(np.trapz(lmbda_nu_hat_p_c_sci_hat_rms_intrgrnd_arr, lmbda_nu_hat_arr)/Z)
            )
            lmbda_nu_hat_p_c_sci_hat_mean = (
                np.trapz(lmbda_nu_hat_p_c_sci_hat_mean_intrgrnd_arr, lmbda_nu_hat_arr)/Z
            )

            p_c_sci_hat_p_c_sci_hat_rms = (
                single_chain.p_c_sci_hat_func(lmbda_nu_hat_p_c_sci_hat_rms)
            )
            p_c_sci_hat_p_c_sci_hat_mean = (
                single_chain.p_c_sci_hat_func(lmbda_nu_hat_p_c_sci_hat_mean)
            )

            indx_left_lmbda_nu_hat_p_c_sci_hat_rms = (
                np.amax(np.flatnonzero(lmbda_nu_hat_arr < lmbda_nu_hat_p_c_sci_hat_rms))
            )
            indx_right_lmbda_nu_hat_p_c_sci_hat_rms = (
                indx_left_lmbda_nu_hat_p_c_sci_hat_rms + 1
            )
            lmbda_nu_hat_left = (
                lmbda_nu_hat[indx_left_lmbda_nu_hat_p_c_sci_hat_rms]
            )
            lmbda_nu_hat_right = (
                lmbda_nu_hat[indx_right_lmbda_nu_hat_p_c_sci_hat_rms]
            )
            epsilon_cnu_diss_hat_left = (
                epsilon_cnu_diss_hat[indx_left_lmbda_nu_hat_p_c_sci_hat_rms]
            )
            epsilon_cnu_diss_hat_right = (
                epsilon_cnu_diss_hat[indx_right_lmbda_nu_hat_p_c_sci_hat_rms]
            )
            epsilon_cnu_diss_hat_p_c_sci_hat_rms = (
                epsilon_cnu_diss_hat_left
                + (lmbda_nu_hat_p_c_sci_hat_rms-lmbda_nu_hat_left)
                * (epsilon_cnu_diss_hat_right-epsilon_cnu_diss_hat_left)
                / (lmbda_nu_hat_right-lmbda_nu_hat_left)
            )

            indx_left_lmbda_nu_hat_p_c_sci_hat_mean = (
                np.amax(np.flatnonzero(lmbda_nu_hat_arr < lmbda_nu_hat_p_c_sci_hat_mean))
            )
            indx_right_lmbda_nu_hat_p_c_sci_hat_mean = (
                indx_left_lmbda_nu_hat_p_c_sci_hat_mean + 1
            )
            lmbda_nu_hat_left = (
                lmbda_nu_hat[indx_left_lmbda_nu_hat_p_c_sci_hat_mean]
            )
            lmbda_nu_hat_right = (
                lmbda_nu_hat[indx_right_lmbda_nu_hat_p_c_sci_hat_mean]
            )
            epsilon_cnu_diss_hat_left = (
                epsilon_cnu_diss_hat[indx_left_lmbda_nu_hat_p_c_sci_hat_mean]
            )
            epsilon_cnu_diss_hat_right = (
                epsilon_cnu_diss_hat[indx_right_lmbda_nu_hat_p_c_sci_hat_mean]
            )
            epsilon_cnu_diss_hat_p_c_sci_hat_mean = (
                epsilon_cnu_diss_hat_left
                + (lmbda_nu_hat_p_c_sci_hat_rms-lmbda_nu_hat_left)
                * (epsilon_cnu_diss_hat_right-epsilon_cnu_diss_hat_left)
                / (lmbda_nu_hat_right-lmbda_nu_hat_left)
            )
            
            overline_epsilon_cnu_sci_hat = [
                epsilon_cnu_sci_hat_val/single_chain.zeta_nu_char
                for epsilon_cnu_sci_hat_val in epsilon_cnu_sci_hat
            ]
            overline_epsilon_cnu_sci_hat_analytical = [
                epsilon_cnu_sci_hat_analytical_val/single_chain.zeta_nu_char
                for epsilon_cnu_sci_hat_analytical_val
                in epsilon_cnu_sci_hat_analytical
            ]
            overline_epsilon_cnu_sci_hat_abs_diff = [
                epsilon_cnu_sci_hat_abs_diff_val/single_chain.zeta_nu_char
                for epsilon_cnu_sci_hat_abs_diff_val
                in epsilon_cnu_sci_hat_abs_diff
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
            overline_epsilon_cnu_diss_hat_abs_diff = [
                epsilon_cnu_diss_hat_abs_diff_val/single_chain.zeta_nu_char
                for epsilon_cnu_diss_hat_abs_diff_val
                in epsilon_cnu_diss_hat_abs_diff
            ]
            overline_epsilon_cnu_diss_hat_equiv = [
                epsilon_cnu_diss_hat_equiv_val/single_chain.zeta_nu_char
                for epsilon_cnu_diss_hat_equiv_val in epsilon_cnu_diss_hat_equiv
            ]
            overline_epsilon_cnu_diss_hat_equiv_abs_diff = [
                epsilon_cnu_diss_hat_equiv_abs_diff_val/single_chain.zeta_nu_char
                for epsilon_cnu_diss_hat_equiv_abs_diff_val
                in epsilon_cnu_diss_hat_equiv_abs_diff
            ]
            overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms = (
                epsilon_cnu_diss_hat_p_c_sci_hat_rms / single_chain.zeta_nu_char
            )
            overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean = (
                epsilon_cnu_diss_hat_p_c_sci_hat_mean / single_chain.zeta_nu_char
            )
            
            lmbda_nu_hat___single_chain_chunk[single_chain_indx] = lmbda_nu_hat
            p_c_sur_hat___single_chain_chunk[single_chain_indx] = p_c_sur_hat
            p_c_sur_hat_analytical___single_chain_chunk[single_chain_indx] = (
                p_c_sur_hat_analytical
            )
            p_c_sur_hat_err___single_chain_chunk[single_chain_indx] = (
                p_c_sur_hat_err
            )
            p_c_sur_hat_abs_diff___single_chain_chunk[single_chain_indx] = (
                p_c_sur_hat_abs_diff
            )
            p_c_sci_hat___single_chain_chunk[single_chain_indx] = p_c_sci_hat
            p_c_sci_hat_analytical___single_chain_chunk[single_chain_indx] = (
                p_c_sci_hat_analytical
            )
            p_c_sci_hat_err___single_chain_chunk[single_chain_indx] = (
                p_c_sci_hat_err
            )
            p_c_sci_hat_abs_diff___single_chain_chunk[single_chain_indx] = (
                p_c_sci_hat_abs_diff
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
            epsilon_cnu_diss_hat_equiv___single_chain_chunk[single_chain_indx] = (
                epsilon_cnu_diss_hat_equiv
            )
            epsilon_cnu_diss_hat_equiv_err___single_chain_chunk[single_chain_indx] = (
                epsilon_cnu_diss_hat_equiv_err
            )
            overline_epsilon_cnu_sci_hat___single_chain_chunk[single_chain_indx] = (
                overline_epsilon_cnu_sci_hat
            )
            overline_epsilon_cnu_sci_hat_analytical___single_chain_chunk[single_chain_indx] = (
                overline_epsilon_cnu_sci_hat_analytical
            )
            overline_epsilon_cnu_sci_hat_abs_diff___single_chain_chunk[single_chain_indx] = (
                overline_epsilon_cnu_sci_hat_abs_diff
            )
            overline_epsilon_cnu_diss_hat___single_chain_chunk[single_chain_indx] = (
                overline_epsilon_cnu_diss_hat
            )
            overline_epsilon_cnu_diss_hat_analytical___single_chain_chunk[single_chain_indx] = (
                overline_epsilon_cnu_diss_hat_analytical
            )
            overline_epsilon_cnu_diss_hat_abs_diff___single_chain_chunk[single_chain_indx] = (
                overline_epsilon_cnu_diss_hat_abs_diff
            )
            overline_epsilon_cnu_diss_hat_equiv___single_chain_chunk[single_chain_indx] = (
                overline_epsilon_cnu_diss_hat_equiv
            )
            overline_epsilon_cnu_diss_hat_equiv_abs_diff___single_chain_chunk[single_chain_indx] = (
                overline_epsilon_cnu_diss_hat_equiv_abs_diff
            )
            u_nu_hat_p_c_sur_hat___single_chain_chunk[single_chain_indx] = (
                u_nu_hat_p_c_sur_hat
            )
            u_nu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx] = (
                u_nu_hat_p_c_sur_hat_max
            )
            lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx] = (
                lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max
            )
            s_cnu_hat_p_c_sur_hat___single_chain_chunk[single_chain_indx] = (
                s_cnu_hat_p_c_sur_hat
            )
            s_cnu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx] = (
                s_cnu_hat_p_c_sur_hat_max
            )
            lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx] = (
                lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max
            )
            psi_cnu_hat_p_c_sur_hat___single_chain_chunk[single_chain_indx] = (
                psi_cnu_hat_p_c_sur_hat
            )
            psi_cnu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx] = (
                psi_cnu_hat_p_c_sur_hat_max
            )
            lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx] = (
                lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max
            )
            xi_c_hat_p_c_sur_hat___single_chain_chunk[single_chain_indx] = (
                xi_c_hat_p_c_sur_hat
            )
            xi_c_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx] = (
                xi_c_hat_p_c_sur_hat_max
            )
            lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx] = (
                lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max
            )
            lmbda_nu_hat_p_c_sci_hat_rms___single_chain_chunk[single_chain_indx] = (
                lmbda_nu_hat_p_c_sci_hat_rms
            )
            lmbda_nu_hat_p_c_sci_hat_mean___single_chain_chunk[single_chain_indx] = (
                lmbda_nu_hat_p_c_sci_hat_mean
            )
            p_c_sci_hat_p_c_sci_hat_rms___single_chain_chunk[single_chain_indx] = (
                p_c_sci_hat_p_c_sci_hat_rms
            )
            p_c_sci_hat_p_c_sci_hat_mean___single_chain_chunk[single_chain_indx] = (
                p_c_sci_hat_p_c_sci_hat_mean
            )
            overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms___single_chain_chunk[single_chain_indx] = (
                overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms
            )
            overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean___single_chain_chunk[single_chain_indx] = (
                overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean
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
        self.p_c_sur_hat_abs_diff___single_chain_chunk = (
            p_c_sur_hat_abs_diff___single_chain_chunk
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
        self.p_c_sci_hat_abs_diff___single_chain_chunk = (
            p_c_sci_hat_abs_diff___single_chain_chunk
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
        self.epsilon_cnu_diss_hat_equiv___single_chain_chunk = (
            epsilon_cnu_diss_hat_equiv___single_chain_chunk
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
        self.epsilon_cnu_diss_hat_equiv_err___single_chain_chunk = (
            epsilon_cnu_diss_hat_equiv_err___single_chain_chunk
        )
        self.overline_epsilon_cnu_sci_hat___single_chain_chunk = (
            overline_epsilon_cnu_sci_hat___single_chain_chunk
        )
        self.overline_epsilon_cnu_sci_hat_analytical___single_chain_chunk = (
            overline_epsilon_cnu_sci_hat_analytical___single_chain_chunk
        )
        self.overline_epsilon_cnu_sci_hat_abs_diff___single_chain_chunk = (
            overline_epsilon_cnu_sci_hat_abs_diff___single_chain_chunk
        )
        self.overline_epsilon_cnu_diss_hat___single_chain_chunk = (
            overline_epsilon_cnu_diss_hat___single_chain_chunk
        )
        self.overline_epsilon_cnu_diss_hat_analytical___single_chain_chunk = (
            overline_epsilon_cnu_diss_hat_analytical___single_chain_chunk
        )
        self.overline_epsilon_cnu_diss_hat_abs_diff___single_chain_chunk = (
            overline_epsilon_cnu_diss_hat_abs_diff___single_chain_chunk
        )
        self.overline_epsilon_cnu_diss_hat_equiv___single_chain_chunk = (
            overline_epsilon_cnu_diss_hat_equiv___single_chain_chunk
        )
        self.overline_epsilon_cnu_diss_hat_equiv_abs_diff___single_chain_chunk = (
            overline_epsilon_cnu_diss_hat_equiv_abs_diff___single_chain_chunk
        )
        self.u_nu_hat_p_c_sur_hat___single_chain_chunk = (
            u_nu_hat_p_c_sur_hat___single_chain_chunk
        )
        self.u_nu_hat_p_c_sur_hat_max___single_chain_chunk = (
            u_nu_hat_p_c_sur_hat_max___single_chain_chunk
        )
        self.lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max___single_chain_chunk = (
            lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max___single_chain_chunk
        )
        self.s_cnu_hat_p_c_sur_hat___single_chain_chunk = (
            s_cnu_hat_p_c_sur_hat___single_chain_chunk
        )
        self.s_cnu_hat_p_c_sur_hat_max___single_chain_chunk = (
            s_cnu_hat_p_c_sur_hat_max___single_chain_chunk
        )
        self.lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max___single_chain_chunk = (
            lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max___single_chain_chunk
        )
        self.psi_cnu_hat_p_c_sur_hat___single_chain_chunk = (
            psi_cnu_hat_p_c_sur_hat___single_chain_chunk
        )
        self.psi_cnu_hat_p_c_sur_hat_max___single_chain_chunk = (
            psi_cnu_hat_p_c_sur_hat_max___single_chain_chunk
        )
        self.lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max___single_chain_chunk = (
            lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max___single_chain_chunk
        )
        self.xi_c_hat_p_c_sur_hat___single_chain_chunk = (
            xi_c_hat_p_c_sur_hat___single_chain_chunk
        )
        self.xi_c_hat_p_c_sur_hat_max___single_chain_chunk = (
            xi_c_hat_p_c_sur_hat_max___single_chain_chunk
        )
        self.lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max___single_chain_chunk = (
            lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max___single_chain_chunk
        )
        self.lmbda_nu_hat_p_c_sci_hat_rms___single_chain_chunk = (
            lmbda_nu_hat_p_c_sci_hat_rms___single_chain_chunk
        )
        self.lmbda_nu_hat_p_c_sci_hat_mean___single_chain_chunk = (
            lmbda_nu_hat_p_c_sci_hat_mean___single_chain_chunk
        )
        self.p_c_sci_hat_p_c_sci_hat_rms___single_chain_chunk = (
            p_c_sci_hat_p_c_sci_hat_rms___single_chain_chunk
        )
        self.p_c_sci_hat_p_c_sci_hat_mean___single_chain_chunk = (
            p_c_sci_hat_p_c_sci_hat_mean___single_chain_chunk
        )
        self.overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms___single_chain_chunk = (
            overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms___single_chain_chunk
        )
        self.overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean___single_chain_chunk = (
            overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean___single_chain_chunk
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
                lmbda_nu_hat, overline_epsilon_cnu_sci_hat_analytical,
                linestyle='-', color=cp.color_list[single_chain_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_label_single_chain_list[single_chain_indx])
            ax1.plot(
                lmbda_nu_hat, overline_epsilon_cnu_diss_hat_analytical,
                linestyle=(0, (3, 1, 1, 1)),
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
        ax1.plot(
            lmbda_nu_hat, overline_epsilon_cnu_sci_hat_analytical,
            linestyle='-', color='black', alpha=1, linewidth=2.5)
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
        ax1.tick_params(axis='y', labelsize=20)
        ax1.set_ylabel(r'$\%~\textrm{error}$', fontsize=21)
        ax1.grid(True, alpha=0.25)
        ax2.legend(loc='best', fontsize=18)
        ax2.tick_params(axis='y', labelsize=20)
        ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=21)
        ax2.grid(True, alpha=0.25)
        plt.xlim([lmbda_nu_hat[0], lmbda_nu_hat[-1]])
        plt.xticks(fontsize=20)
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=30)
        save_current_figure_no_labels(
            self.savedir,
            "rate-independent-chain-scission-indicators-percent-error-vs-lmbda_nu_hat")
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_sci_hat_abs_diff = (
                self.overline_epsilon_cnu_sci_hat_abs_diff___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_diss_hat_abs_diff = (
                self.overline_epsilon_cnu_diss_hat_abs_diff___single_chain_chunk[single_chain_indx]
            )
            ax1.plot(
                lmbda_nu_hat, overline_epsilon_cnu_sci_hat_abs_diff,
                linestyle='-', color=cp.color_list[single_chain_indx],
                alpha=1, linewidth=2.5,
                label=cp.nu_label_single_chain_list[single_chain_indx])
            ax1.plot(
                lmbda_nu_hat, overline_epsilon_cnu_diss_hat_abs_diff,
                linestyle=(0, (3, 1, 1, 1)),
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            p_c_sur_hat_abs_diff  = (
                self.p_c_sur_hat_abs_diff___single_chain_chunk[single_chain_indx]
            )
            p_c_sci_hat_abs_diff  = (
                self.p_c_sci_hat_abs_diff___single_chain_chunk[single_chain_indx]
            )
            ax2.plot(
                lmbda_nu_hat, p_c_sci_hat_abs_diff, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
            ax2.plot(
                lmbda_nu_hat, p_c_sur_hat_abs_diff, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
        ax2.plot(
            [], [], linestyle='-', color='black', label=r'$\hat{p}_c^{sci}$')
        ax2.plot(
            [], [], linestyle=':', color='black', label=r'$\hat{p}_c^{sur}$')

        ax1.legend(loc='best', fontsize=18)
        # ax1.set_ylim([-0.05, 0.7])
        ax1.tick_params(axis='y', labelsize=20)
        ax1.set_ylabel(r'$|\textrm{diff}|$', fontsize=21)
        ax1.grid(True, alpha=0.25)
        ax2.legend(loc='best', fontsize=18)
        # ax2.set_ylim([-0.05, 1.05])
        ax2.tick_params(axis='y', labelsize=20)
        ax2.set_ylabel(r'$|\textrm{diff}|$', fontsize=21)
        ax2.grid(True, alpha=0.25)
        plt.xlim([lmbda_nu_hat[0], lmbda_nu_hat[-1]])
        plt.xticks(fontsize=20)
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=30)
        save_current_figure_no_labels(
            self.savedir,
            "rate-independent-chain-scission-indicators-absolute-difference-vs-lmbda_nu_hat")
        
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
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_sci_hat = (
                self.overline_epsilon_cnu_sci_hat___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_diss_hat_equiv = (
                self.overline_epsilon_cnu_diss_hat_equiv___single_chain_chunk[single_chain_indx]
            )
            ax1.plot(
                lmbda_nu_hat, overline_epsilon_cnu_sci_hat, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5,
                label=cp.nu_label_single_chain_list[single_chain_indx])
            ax1.plot(
                lmbda_nu_hat, overline_epsilon_cnu_diss_hat_equiv,
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
            "rate-independent-chain-scission-indicators-equivalent-dissipated-chain-scission-vs-lmbda_nu_hat")
        
        fig = plt.figure()
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_sci_hat = (
                self.overline_epsilon_cnu_sci_hat___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_diss_hat_equiv = (
                self.overline_epsilon_cnu_diss_hat_equiv___single_chain_chunk[single_chain_indx]
            )
            nu = self.nu___single_chain_chunk[single_chain_indx]
            overline_epsilon_c_sci_hat = [
                x*nu for x in overline_epsilon_cnu_sci_hat
            ]
            overline_epsilon_c_diss_hat_equiv = [
                x*nu for x in overline_epsilon_cnu_diss_hat_equiv
            ]
            plt.semilogy(
                lmbda_nu_hat, overline_epsilon_c_sci_hat, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5,
                label=cp.nu_label_single_chain_list[single_chain_indx])
            plt.semilogy(
                lmbda_nu_hat, overline_epsilon_c_diss_hat_equiv,
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
            30, "rate-independent-chain-scission-energy-equivalent-dissipated-chain-scission-vs-lmbda_nu_hat")
        
        fig = plt.figure()
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            epsilon_cnu_diss_hat_equiv_err = (
                self.epsilon_cnu_diss_hat_equiv_err___single_chain_chunk[single_chain_indx]
            )
            plt.plot(lmbda_nu_hat, epsilon_cnu_diss_hat_equiv_err,
                 linestyle='-', color=cp.color_list[single_chain_indx],
                 alpha=1, linewidth=2.5,
                 label=cp.nu_label_single_chain_list[single_chain_indx])
        plt.legend(loc='best', fontsize=14, handlelength=3)
        plt.xlim([lmbda_nu_hat[0], lmbda_nu_hat[-1]])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\hat{\lambda}_{\nu}$', 30,
            r'$\%~\textrm{error}$', 30,
            "equivalent-rate-independent-dissipated-chain-scission-percent-error-vs-lmbda_nu_hat")

        fig = plt.figure()
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_diss_hat_equiv_abs_diff = (
                self.overline_epsilon_cnu_diss_hat_equiv_abs_diff___single_chain_chunk[single_chain_indx]
            )
            plt.plot(lmbda_nu_hat, overline_epsilon_cnu_diss_hat_equiv_abs_diff,
                 linestyle='-', color=cp.color_list[single_chain_indx],
                 alpha=1, linewidth=2.5,
                 label=cp.nu_label_single_chain_list[single_chain_indx])
        plt.legend(loc='best', fontsize=14, handlelength=3)
        plt.xlim([lmbda_nu_hat[0], lmbda_nu_hat[-1]])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\hat{\lambda}_{\nu}$', 30,
            r'$|\textrm{diff}|$', 30,
            "equivalent-rate-independent-dissipated-chain-scission-absolute-difference-vs-lmbda_nu_hat")
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_diss_hat = (
                self.overline_epsilon_cnu_diss_hat___single_chain_chunk[single_chain_indx]
            )
            lmbda_nu_hat_p_c_sci_hat_rms = (
                self.lmbda_nu_hat_p_c_sci_hat_rms___single_chain_chunk[single_chain_indx]
            )
            lmbda_nu_hat_p_c_sci_hat_mean = (
                self.lmbda_nu_hat_p_c_sci_hat_mean___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms = (
                self.overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean = (
                self.overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean___single_chain_chunk[single_chain_indx]
            )
            p_c_sci_hat = (
                self.p_c_sci_hat___single_chain_chunk[single_chain_indx]
            )
            lmbda_nu_hat_p_c_sci_hat_rms = (
                self.lmbda_nu_hat_p_c_sci_hat_rms___single_chain_chunk[single_chain_indx]
            )
            lmbda_nu_hat_p_c_sci_hat_mean = (
                self.lmbda_nu_hat_p_c_sci_hat_mean___single_chain_chunk[single_chain_indx]
            )
            p_c_sci_hat_p_c_sci_hat_rms = (
                self.p_c_sci_hat_p_c_sci_hat_rms___single_chain_chunk[single_chain_indx]
            )
            p_c_sci_hat_p_c_sci_hat_mean = (
                self.p_c_sci_hat_p_c_sci_hat_mean___single_chain_chunk[single_chain_indx]
            )

            ax1.plot(
                lmbda_nu_hat, overline_epsilon_cnu_diss_hat, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
            ax1.vlines(
                x=lmbda_nu_hat_p_c_sci_hat_rms,
                ymin=overline_epsilon_cnu_diss_hat[0],
                ymax=overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=1)
            ax1.hlines(
                y=overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms,
                xmin=lmbda_nu_hat[0],xmax=lmbda_nu_hat_p_c_sci_hat_rms,
                linestyle=':', color=cp.color_list[single_chain_indx], alpha=1,
                linewidth=1)
            ax1.vlines(
                x=lmbda_nu_hat_p_c_sci_hat_mean,
                ymin=overline_epsilon_cnu_diss_hat[0],
                ymax=overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=1)
            ax1.hlines(
                y=overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean,
                xmin=lmbda_nu_hat[0], xmax=lmbda_nu_hat_p_c_sci_hat_mean,
                linestyle=':', color=cp.color_list[single_chain_indx],
                alpha=1, linewidth=1)
            ax2.plot(
                lmbda_nu_hat, p_c_sci_hat, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
            ax2.vlines(
                x=lmbda_nu_hat_p_c_sci_hat_rms, ymin=p_c_sci_hat[0],
                ymax=p_c_sci_hat_p_c_sci_hat_rms, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=1)
            ax2.hlines(
                y=p_c_sci_hat_p_c_sci_hat_rms, xmin=lmbda_nu_hat[0],
                xmax=lmbda_nu_hat_p_c_sci_hat_rms, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=1)
            ax2.vlines(
                x=lmbda_nu_hat_p_c_sci_hat_mean, ymin=p_c_sci_hat[0],
                ymax=p_c_sci_hat_p_c_sci_hat_mean, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=1)
            ax2.hlines(
                y=p_c_sci_hat_p_c_sci_hat_mean, xmin=lmbda_nu_hat[0],
                xmax=lmbda_nu_hat_p_c_sci_hat_mean, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=1)
        
        ax1.set_ylim([-0.05, 0.7])
        ax1.tick_params(axis='y', labelsize=20)
        ax1.set_ylabel(
            r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', fontsize=21)
        ax1.grid(True, alpha=0.25)
        ax2.set_ylim([-0.05, 1.05])
        ax2.tick_params(axis='y', labelsize=20)
        ax2.set_ylabel(r'$\hat{p}_c^{sci}$', fontsize=21)
        ax2.grid(True, alpha=0.25)
        plt.xlim([lmbda_nu_hat[0], lmbda_nu_hat[-1]])
        plt.xticks(fontsize=20)
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=30)
        save_current_figure_no_labels(
            self.savedir,
            "rate-independent-probability-chain-scission-statistical-averages-vs-lmbda_nu_hat")
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
        for single_chain_indx in range(len(self.single_chain_list)):
            lmbda_nu_hat = (
                self.lmbda_nu_hat___single_chain_chunk[single_chain_indx]
            )
            u_nu_hat_p_c_sur_hat = (
                self.u_nu_hat_p_c_sur_hat___single_chain_chunk[single_chain_indx]
            )
            lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max = (
                self.lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            u_nu_hat_p_c_sur_hat_max = (
                self.u_nu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            s_cnu_hat_p_c_sur_hat = (
                self.s_cnu_hat_p_c_sur_hat___single_chain_chunk[single_chain_indx]
            )
            lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max = (
                self.lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            s_cnu_hat_p_c_sur_hat_max = (
                self.s_cnu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            psi_cnu_hat_p_c_sur_hat = (
                self.psi_cnu_hat_p_c_sur_hat___single_chain_chunk[single_chain_indx]
            )
            lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max = (
                self.lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            psi_cnu_hat_p_c_sur_hat_max = (
                self.psi_cnu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            xi_c_hat_p_c_sur_hat = (
                self.xi_c_hat_p_c_sur_hat___single_chain_chunk[single_chain_indx]
            )
            lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max = (
                self.lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            xi_c_hat_p_c_sur_hat_max = (
                self.xi_c_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )

            ax1.plot(
                lmbda_nu_hat, u_nu_hat_p_c_sur_hat, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
            ax1.vlines(
                x=lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max,
                ymin=u_nu_hat_p_c_sur_hat[0], ymax=u_nu_hat_p_c_sur_hat_max,
                linestyle=':', color=cp.color_list[single_chain_indx], alpha=1,
                linewidth=1)
            ax1.hlines(
                y=u_nu_hat_p_c_sur_hat_max, xmin=lmbda_nu_hat[0],
                xmax=lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=1)
            ax2.plot(
                lmbda_nu_hat, s_cnu_hat_p_c_sur_hat, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
            ax2.vlines(
                x=lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max,
                ymin=s_cnu_hat_p_c_sur_hat[0],
                ymax=s_cnu_hat_p_c_sur_hat_max, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=1)
            ax2.hlines(
                y=s_cnu_hat_p_c_sur_hat_max, xmin=lmbda_nu_hat[0],
                xmax=lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=1)
            ax3.plot(
                lmbda_nu_hat, psi_cnu_hat_p_c_sur_hat, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
            ax3.vlines(
                x=lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max,
                ymin=psi_cnu_hat_p_c_sur_hat[0],
                ymax=psi_cnu_hat_p_c_sur_hat_max, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=1)
            ax3.hlines(
                y=psi_cnu_hat_p_c_sur_hat_max, xmin=lmbda_nu_hat[0],
                xmax=lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=1)
            ax4.plot(
                lmbda_nu_hat, xi_c_hat_p_c_sur_hat, linestyle='-',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
            ax4.vlines(
                x=lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max,
                ymin=xi_c_hat_p_c_sur_hat[0],
                ymax=xi_c_hat_p_c_sur_hat_max, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=1)
            ax4.hlines(
                y=xi_c_hat_p_c_sur_hat_max, xmin=lmbda_nu_hat[0],
                xmax=lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max, linestyle=':',
                color=cp.color_list[single_chain_indx], alpha=1, linewidth=1)

        ax1.tick_params(axis='y', labelsize=16)
        ax1.set_ylabel(r'$\hat{u}_{\nu}\times\hat{p}_c^{sur}$', fontsize=16)
        ax1.grid(True, alpha=0.25)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.set_ylabel(r'$\hat{s}_{c\nu}\times\hat{p}_c^{sur}$', fontsize=16)
        ax2.grid(True, alpha=0.25)
        ax3.tick_params(axis='y', labelsize=16)
        ax3.set_ylabel(r'$\hat{\psi}_{c\nu}\times\hat{p}_c^{sur}$', fontsize=16)
        ax3.grid(True, alpha=0.25)
        ax4.tick_params(axis='y', labelsize=16)
        ax4.set_ylabel(r'$\hat{\xi}_{c}\times\hat{p}_c^{sur}$', fontsize=16)
        ax4.grid(True, alpha=0.25)
        
        plt.xlim([lmbda_nu_hat[0], lmbda_nu_hat[-1]])
        plt.xticks(fontsize=16)
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir,
            "rate-independent-survived-chain-parameters-vs-lmbda_nu_hat")
        
        for single_chain_indx in range(len(self.single_chain_list)):
            nu = self.nu___single_chain_chunk[single_chain_indx]

            overline_epsilon_cnu_diss_hat = (
                self.overline_epsilon_cnu_diss_hat___single_chain_chunk[single_chain_indx]
            )
            
            u_nu_hat_p_c_sur_hat_max = (
                self.u_nu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max = (
                self.lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            s_cnu_hat_p_c_sur_hat_max = (
                self.s_cnu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max = (
                self.lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            psi_cnu_hat_p_c_sur_hat_max = (
                self.psi_cnu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max = (
                self.lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            xi_c_hat_p_c_sur_hat_max = (
                self.xi_c_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max = (
                self.lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max___single_chain_chunk[single_chain_indx]
            )
            
            lmbda_nu_hat_p_c_sci_hat_rms = (
                self.lmbda_nu_hat_p_c_sci_hat_rms___single_chain_chunk[single_chain_indx]
            )
            lmbda_nu_hat_p_c_sci_hat_mean = (
                self.lmbda_nu_hat_p_c_sci_hat_mean___single_chain_chunk[single_chain_indx]
            )
            p_c_sci_hat_p_c_sci_hat_rms = (
                self.p_c_sci_hat_p_c_sci_hat_rms___single_chain_chunk[single_chain_indx]
            )
            p_c_sci_hat_p_c_sci_hat_mean = (
                self.p_c_sci_hat_p_c_sci_hat_mean___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms = (
                self.overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms___single_chain_chunk[single_chain_indx]
            )
            overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean = (
                self.overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean___single_chain_chunk[single_chain_indx]
            )
            print("\n nu = {}".format(nu))

            print("overline_epsilon_cnu_diss_hat_crit = {}".format(overline_epsilon_cnu_diss_hat[-1]))

            print("u_nu_hat_p_c_sur_hat_max = {}".format(u_nu_hat_p_c_sur_hat_max))
            print("lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max = {}".format(lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max))
            print("s_cnu_hat_p_c_sur_hat_max = {}".format(s_cnu_hat_p_c_sur_hat_max))
            print("lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max = {}".format(lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max))
            print("psi_cnu_hat_p_c_sur_hat_max = {}".format(psi_cnu_hat_p_c_sur_hat_max))
            print("lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max = {}".format(lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max))
            print("xi_c_hat_p_c_sur_hat_max = {}".format(xi_c_hat_p_c_sur_hat_max))
            print("lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max = {}".format(lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max))

            print("lmbda_nu_hat_p_c_sci_hat_rms = {}".format(lmbda_nu_hat_p_c_sci_hat_rms))
            print("lmbda_nu_hat_p_c_sci_hat_mean = {}".format(lmbda_nu_hat_p_c_sci_hat_mean))
            print("p_c_sci_hat_p_c_sci_hat_rms = {}".format(p_c_sci_hat_p_c_sci_hat_rms))
            print("p_c_sci_hat_p_c_sci_hat_mean = {}".format(p_c_sci_hat_p_c_sci_hat_mean))
            print("overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms = {}".format(overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms))
            print("overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean = {}".format(overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean))

            single_chain = self.single_chain_list[single_chain_indx]

            assert np.around(overline_epsilon_cnu_diss_hat[-1]*single_chain.zeta_nu_char, 6) == np.around(single_chain.epsilon_cnu_diss_hat_crit, 6)

            assert u_nu_hat_p_c_sur_hat_max == single_chain.u_nu_hat_p_c_sur_hat_max
            assert lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max == single_chain.lmbda_nu_hat_u_nu_hat_p_c_sur_hat_max
            assert s_cnu_hat_p_c_sur_hat_max == single_chain.s_cnu_hat_p_c_sur_hat_max
            assert lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max == single_chain.lmbda_nu_hat_s_cnu_hat_p_c_sur_hat_max
            assert psi_cnu_hat_p_c_sur_hat_max == single_chain.psi_cnu_hat_p_c_sur_hat_max
            assert lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max == single_chain.lmbda_nu_hat_psi_cnu_hat_p_c_sur_hat_max
            assert xi_c_hat_p_c_sur_hat_max == single_chain.xi_c_hat_p_c_sur_hat_max
            assert lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max == single_chain.lmbda_nu_hat_xi_c_hat_p_c_sur_hat_max

            assert lmbda_nu_hat_p_c_sci_hat_rms == single_chain.lmbda_nu_hat_p_c_sci_hat_rms
            assert lmbda_nu_hat_p_c_sci_hat_mean == single_chain.lmbda_nu_hat_p_c_sci_hat_mean
            assert np.around(overline_epsilon_cnu_diss_hat_p_c_sci_hat_rms*single_chain.zeta_nu_char, 6) == np.around(single_chain.epsilon_cnu_diss_hat_p_c_sci_hat_rms, 6)
            assert np.around(overline_epsilon_cnu_diss_hat_p_c_sci_hat_mean*single_chain.zeta_nu_char, 6) == np.around(single_chain.epsilon_cnu_diss_hat_p_c_sci_hat_mean, 6)


if __name__ == '__main__':

    characterizer = RateIndependentChainScissionCharacterizer()
    characterizer.characterization()
    characterizer.finalization()