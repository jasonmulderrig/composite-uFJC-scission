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
        u_nu_hat_p_nu_sur_hat    = []
        s_cnu_hat_p_nu_sur_hat   = []
        psi_cnu_hat_p_nu_sur_hat = []
        xi_c_hat_p_nu_sur_hat    = []

        # initialization
        lmbda_nu_hat_max_val = 0.
        epsilon_nu_diss_hat_val            = 0.
        epsilon_nu_diss_hat_analytical_val = 0.
        epsilon_nu_diss_hat_err_val        = 0.
        epsilon_nu_diss_hat_abs_diff_val   = 0.
        epsilon_nu_diss_hat_equiv_val      = 0.
        epsilon_nu_diss_hat_equiv_err_val  = 0.
        epsilon_nu_diss_hat_equiv_abs_diff_val = 0.
        u_nu_hat_p_nu_sur_hat_val    = 0.
        s_cnu_hat_p_nu_sur_hat_val   = 0.
        psi_cnu_hat_p_nu_sur_hat_val = 0.
        xi_c_hat_p_nu_sur_hat_val    = 0.
        
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
            lmbda_c_eq_hat_val = single_chain.lmbda_c_eq_func(lmbda_nu_hat_val)
            lmbda_comp_nu_hat_val = lmbda_c_eq_hat_val - lmbda_nu_hat_val + 1.
            u_nu_hat_val = single_chain.u_nu_func(lmbda_nu_hat_val)
            s_cnu_hat_val = single_chain.s_cnu_func(lmbda_comp_nu_hat_val)
            psi_cnu_hat_val = (
                single_chain.psi_cnu_func(lmbda_nu_hat_val, lmbda_c_eq_hat_val)
            )
            xi_c_hat_val = (
                single_chain.xi_c_func(lmbda_nu_hat_val, lmbda_c_eq_hat_val)
            )
            
            if lmbda_nu_hat_indx == 0:
                u_nu_hat_init_val = u_nu_hat_val
                s_cnu_hat_init_val = s_cnu_hat_val
                psi_cnu_hat_init_val = psi_cnu_hat_val
                xi_c_hat_init_val = xi_c_hat_val
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
                u_nu_hat_p_nu_sur_hat_val = (
                    p_nu_sur_hat_val * (u_nu_hat_val-u_nu_hat_init_val)
                )
                s_cnu_hat_p_nu_sur_hat_val = (
                    p_nu_sur_hat_val * (s_cnu_hat_val-s_cnu_hat_init_val)
                )
                psi_cnu_hat_p_nu_sur_hat_val = (
                    p_nu_sur_hat_val * (psi_cnu_hat_val-psi_cnu_hat_init_val)
                )
                xi_c_hat_p_nu_sur_hat_val = (
                    p_nu_sur_hat_val * (xi_c_hat_val-xi_c_hat_init_val)
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
            u_nu_hat_p_nu_sur_hat.append(u_nu_hat_p_nu_sur_hat_val)
            s_cnu_hat_p_nu_sur_hat.append(s_cnu_hat_p_nu_sur_hat_val)
            psi_cnu_hat_p_nu_sur_hat.append(psi_cnu_hat_p_nu_sur_hat_val)
            xi_c_hat_p_nu_sur_hat.append(xi_c_hat_p_nu_sur_hat_val)
        
        indx_u_nu_hat_p_nu_sur_hat_max = (
            np.argmax(np.asarray(u_nu_hat_p_nu_sur_hat))
        )
        u_nu_hat_p_nu_sur_hat_max = (
            u_nu_hat_p_nu_sur_hat[indx_u_nu_hat_p_nu_sur_hat_max]
        )
        lmbda_nu_hat_u_nu_hat_p_nu_sur_hat_max = (
            lmbda_nu_hat[indx_u_nu_hat_p_nu_sur_hat_max]
        )
        indx_s_cnu_hat_p_nu_sur_hat_max = (
            np.argmax(np.asarray(s_cnu_hat_p_nu_sur_hat))
        )
        s_cnu_hat_p_nu_sur_hat_max = (
            s_cnu_hat_p_nu_sur_hat[indx_s_cnu_hat_p_nu_sur_hat_max]
        )
        lmbda_nu_hat_s_cnu_hat_p_nu_sur_hat_max = (
            lmbda_nu_hat[indx_s_cnu_hat_p_nu_sur_hat_max]
        )
        indx_psi_cnu_hat_p_nu_sur_hat_max = (
            np.argmax(np.asarray(psi_cnu_hat_p_nu_sur_hat))
        )
        psi_cnu_hat_p_nu_sur_hat_max = (
            psi_cnu_hat_p_nu_sur_hat[indx_psi_cnu_hat_p_nu_sur_hat_max]
        )
        lmbda_nu_hat_psi_cnu_hat_p_nu_sur_hat_max = (
            lmbda_nu_hat[indx_psi_cnu_hat_p_nu_sur_hat_max]
        )
        indx_xi_c_hat_p_nu_sur_hat_max = (
            np.argmax(np.asarray(xi_c_hat_p_nu_sur_hat))
        )
        xi_c_hat_p_nu_sur_hat_max = (
            xi_c_hat_p_nu_sur_hat[indx_xi_c_hat_p_nu_sur_hat_max]
        )
        lmbda_nu_hat_xi_c_hat_p_nu_sur_hat_max = (
            lmbda_nu_hat[indx_xi_c_hat_p_nu_sur_hat_max]
        )
        
        lmbda_nu_hat_arr = np.asarray(lmbda_nu_hat)
        p_nu_sci_hat_arr = np.asarray(p_nu_sci_hat)

        partial_p_nu_sci_hat__partial_lmbda_nu_hat_arr = (
            np.gradient(p_nu_sci_hat_arr, lmbda_nu_hat_arr, edge_order=2)
        )

        Z = (
            np.trapz(partial_p_nu_sci_hat__partial_lmbda_nu_hat_arr, lmbda_nu_hat_arr)
        )

        lmbda_nu_hat_p_nu_sci_hat_rms_intrgrnd_arr = (
            partial_p_nu_sci_hat__partial_lmbda_nu_hat_arr * lmbda_nu_hat_arr**2
        )
        lmbda_nu_hat_p_nu_sci_hat_mean_intrgrnd_arr = (
            partial_p_nu_sci_hat__partial_lmbda_nu_hat_arr * lmbda_nu_hat_arr
        )

        lmbda_nu_hat_p_nu_sci_hat_rms = (
            np.sqrt(np.trapz(lmbda_nu_hat_p_nu_sci_hat_rms_intrgrnd_arr, lmbda_nu_hat_arr)/Z)
        )
        lmbda_nu_hat_p_nu_sci_hat_mean = (
            np.trapz(lmbda_nu_hat_p_nu_sci_hat_mean_intrgrnd_arr, lmbda_nu_hat_arr)/Z
        )

        p_nu_sci_hat_p_nu_sci_hat_rms = (
            single_chain.p_nu_sci_hat_func(lmbda_nu_hat_p_nu_sci_hat_rms)
        )
        p_nu_sci_hat_p_nu_sci_hat_mean = (
            single_chain.p_nu_sci_hat_func(lmbda_nu_hat_p_nu_sci_hat_mean)
        )

        indx_left_lmbda_nu_hat_p_nu_sci_hat_rms = (
            np.amax(np.flatnonzero(lmbda_nu_hat_arr < lmbda_nu_hat_p_nu_sci_hat_rms))
        )
        indx_right_lmbda_nu_hat_p_nu_sci_hat_rms = (
            indx_left_lmbda_nu_hat_p_nu_sci_hat_rms + 1
        )
        lmbda_nu_hat_left = (
            lmbda_nu_hat[indx_left_lmbda_nu_hat_p_nu_sci_hat_rms]
        )
        lmbda_nu_hat_right = (
            lmbda_nu_hat[indx_right_lmbda_nu_hat_p_nu_sci_hat_rms]
        )
        epsilon_nu_diss_hat_left = (
            epsilon_nu_diss_hat[indx_left_lmbda_nu_hat_p_nu_sci_hat_rms]
        )
        epsilon_nu_diss_hat_right = (
            epsilon_nu_diss_hat[indx_right_lmbda_nu_hat_p_nu_sci_hat_rms]
        )
        epsilon_nu_diss_hat_p_nu_sci_hat_rms = (
            epsilon_nu_diss_hat_left
            + (lmbda_nu_hat_p_nu_sci_hat_rms-lmbda_nu_hat_left)
            * (epsilon_nu_diss_hat_right-epsilon_nu_diss_hat_left)
            / (lmbda_nu_hat_right-lmbda_nu_hat_left)
        )

        indx_left_lmbda_nu_hat_p_nu_sci_hat_mean = (
            np.amax(np.flatnonzero(lmbda_nu_hat_arr < lmbda_nu_hat_p_nu_sci_hat_mean))
        )
        indx_right_lmbda_nu_hat_p_nu_sci_hat_mean = (
            indx_left_lmbda_nu_hat_p_nu_sci_hat_mean + 1
        )
        lmbda_nu_hat_left = (
            lmbda_nu_hat[indx_left_lmbda_nu_hat_p_nu_sci_hat_mean]
        )
        lmbda_nu_hat_right = (
            lmbda_nu_hat[indx_right_lmbda_nu_hat_p_nu_sci_hat_mean]
        )
        epsilon_nu_diss_hat_left = (
            epsilon_nu_diss_hat[indx_left_lmbda_nu_hat_p_nu_sci_hat_mean]
        )
        epsilon_nu_diss_hat_right = (
            epsilon_nu_diss_hat[indx_right_lmbda_nu_hat_p_nu_sci_hat_mean]
        )
        epsilon_nu_diss_hat_p_nu_sci_hat_mean = (
            epsilon_nu_diss_hat_left
            + (lmbda_nu_hat_p_nu_sci_hat_rms-lmbda_nu_hat_left)
            * (epsilon_nu_diss_hat_right-epsilon_nu_diss_hat_left)
            / (lmbda_nu_hat_right-lmbda_nu_hat_left)
        )
        
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
        overline_epsilon_nu_diss_hat_p_nu_sci_hat_rms = (
            epsilon_nu_diss_hat_p_nu_sci_hat_rms / single_chain.zeta_nu_char
        )
        overline_epsilon_nu_diss_hat_p_nu_sci_hat_mean = (
            epsilon_nu_diss_hat_p_nu_sci_hat_mean / single_chain.zeta_nu_char
        )
        
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
        self.overline_epsilon_nu_diss_hat = (
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
        self.u_nu_hat_p_nu_sur_hat = u_nu_hat_p_nu_sur_hat
        self.u_nu_hat_p_nu_sur_hat_max = u_nu_hat_p_nu_sur_hat_max
        self.lmbda_nu_hat_u_nu_hat_p_nu_sur_hat_max = (
            lmbda_nu_hat_u_nu_hat_p_nu_sur_hat_max
        )
        self.s_cnu_hat_p_nu_sur_hat = s_cnu_hat_p_nu_sur_hat
        self.s_cnu_hat_p_nu_sur_hat_max = s_cnu_hat_p_nu_sur_hat_max
        self.lmbda_nu_hat_s_cnu_hat_p_nu_sur_hat_max = (
            lmbda_nu_hat_s_cnu_hat_p_nu_sur_hat_max
        )
        self.psi_cnu_hat_p_nu_sur_hat = psi_cnu_hat_p_nu_sur_hat
        self.psi_cnu_hat_p_nu_sur_hat_max = psi_cnu_hat_p_nu_sur_hat_max
        self.lmbda_nu_hat_psi_cnu_hat_p_nu_sur_hat_max = (
            lmbda_nu_hat_psi_cnu_hat_p_nu_sur_hat_max
        )
        self.xi_c_hat_p_nu_sur_hat = xi_c_hat_p_nu_sur_hat
        self.xi_c_hat_p_nu_sur_hat_max = xi_c_hat_p_nu_sur_hat_max
        self.lmbda_nu_hat_xi_c_hat_p_nu_sur_hat_max = (
            lmbda_nu_hat_xi_c_hat_p_nu_sur_hat_max
        )
        
        self.lmbda_nu_hat_p_nu_sci_hat_rms = lmbda_nu_hat_p_nu_sci_hat_rms
        self.lmbda_nu_hat_p_nu_sci_hat_mean = lmbda_nu_hat_p_nu_sci_hat_mean
        self.p_nu_sci_hat_p_nu_sci_hat_rms = p_nu_sci_hat_p_nu_sci_hat_rms
        self.p_nu_sci_hat_p_nu_sci_hat_mean = p_nu_sci_hat_p_nu_sci_hat_mean
        self.overline_epsilon_nu_diss_hat_p_nu_sci_hat_rms = (
            overline_epsilon_nu_diss_hat_p_nu_sci_hat_rms
        )
        self.overline_epsilon_nu_diss_hat_p_nu_sci_hat_mean = (
            overline_epsilon_nu_diss_hat_p_nu_sci_hat_mean
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
            self.lmbda_nu_hat, self.overline_e_nu_sci_hat_analytical,
            linestyle=':', color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{e}_{\nu}^{sci}}$')
        ax1.plot(
            self.lmbda_nu_hat, self.overline_epsilon_nu_sci_hat_analytical,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{sci}}$')
        ax1.plot(
            self.lmbda_nu_hat, self.overline_epsilon_nu_diss_hat_analytical,
            linestyle=(0, (3, 1, 1, 1)), color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{diss}}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sci_hat_analytical, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\hat{p}_{\nu}^{sci}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sur_hat_analytical, linestyle=':',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\hat{p}_{\nu}^{sur}$')

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
            self.lmbda_nu_hat, self.p_nu_sci_hat_err, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\hat{p}_{\nu}^{sci}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sur_hat_err, linestyle=':',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\hat{p}_{\nu}^{sur}$')

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
            self.lmbda_nu_hat, self.overline_epsilon_nu_sci_hat_abs_diff,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{sci}}$')
        ax1.plot(
            self.lmbda_nu_hat, self.overline_epsilon_nu_diss_hat_abs_diff,
            linestyle=(0, (3, 1, 1, 1)), color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{diss}}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sci_hat_abs_diff, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\hat{p}_{\nu}^{sci}$')
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sur_hat_abs_diff, linestyle=':',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\hat{p}_{\nu}^{sur}$')

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
            self.savedir, r'$\hat{\lambda}_{\nu}$', 30,
            r'$\%~\textrm{error}$', 30,
            "equivalent-rate-independent-dissipated-segment-scission-percent-error-vs-lmbda_nu_hat")
        
        fig = plt.figure()
        plt.plot(self.lmbda_nu_hat,
                 self.overline_epsilon_nu_diss_hat_equiv_abs_diff,
                 linestyle='-', color='blue', alpha=1, linewidth=2.5,
                 label=r'$\overline{\hat{\varepsilon}_{\nu}^{diss}}$')
        plt.legend(loc='best', fontsize=14, handlelength=3)
        plt.xlim([self.lmbda_nu_hat[0], self.lmbda_nu_hat[-1]])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\hat{\lambda}_{\nu}$', 30,
            r'$|\textrm{diff}|$', 30,
            "equivalent-rate-independent-dissipated-segment-scission-absolute-difference-vs-lmbda_nu_hat")
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        ax1.plot(
            self.lmbda_nu_hat, self.overline_epsilon_nu_diss_hat,
            linestyle=(0, (3, 1, 1, 1)), color='blue', alpha=1, linewidth=2.5,
            label=r'$\overline{\hat{\varepsilon}_{\nu}^{diss}}$')
        ax1.vlines(
            x=self.lmbda_nu_hat_p_nu_sci_hat_rms,
            ymin=self.overline_epsilon_nu_diss_hat[0],
            ymax=self.overline_epsilon_nu_diss_hat_p_nu_sci_hat_rms,
            linestyle=':', color='black', alpha=1, linewidth=1)
        ax1.hlines(
            y=self.overline_epsilon_nu_diss_hat_p_nu_sci_hat_rms,
            xmin=self.lmbda_nu_hat[0], xmax=self.lmbda_nu_hat_p_nu_sci_hat_rms,
            linestyle=':', color='black', alpha=1, linewidth=1)
        ax1.vlines(
            x=self.lmbda_nu_hat_p_nu_sci_hat_mean,
            ymin=self.overline_epsilon_nu_diss_hat[0],
            ymax=self.overline_epsilon_nu_diss_hat_p_nu_sci_hat_mean,
            linestyle=':', color='green', alpha=1, linewidth=1)
        ax1.hlines(
            y=self.overline_epsilon_nu_diss_hat_p_nu_sci_hat_mean,
            xmin=self.lmbda_nu_hat[0], xmax=self.lmbda_nu_hat_p_nu_sci_hat_mean,
            linestyle=':', color='green', alpha=1, linewidth=1)
        ax2.plot(
            self.lmbda_nu_hat, self.p_nu_sci_hat, linestyle=':', color='blue',
            alpha=1, linewidth=2.5)
        ax2.vlines(
            x=self.lmbda_nu_hat_p_nu_sci_hat_rms, ymin=self.p_nu_sci_hat[0],
            ymax=self.p_nu_sci_hat_p_nu_sci_hat_rms, linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax2.hlines(
            y=self.p_nu_sci_hat_p_nu_sci_hat_rms, xmin=self.lmbda_nu_hat[0],
            xmax=self.lmbda_nu_hat_p_nu_sci_hat_rms, linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax2.vlines(
            x=self.lmbda_nu_hat_p_nu_sci_hat_mean, ymin=self.p_nu_sci_hat[0],
            ymax=self.p_nu_sci_hat_p_nu_sci_hat_mean, linestyle=':',
            color='green', alpha=1, linewidth=1)
        ax2.hlines(
            y=self.p_nu_sci_hat_p_nu_sci_hat_mean, xmin=self.lmbda_nu_hat[0],
            xmax=self.lmbda_nu_hat_p_nu_sci_hat_mean, linestyle=':',
            color='green', alpha=1, linewidth=1)

        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(axis='y', labelsize=16)
        ax1.set_ylabel(r'$\overline{\hat{\varepsilon}_{\nu}^{diss}}$',
            fontsize=20)
        ax1.grid(True, alpha=0.25)
        ax2.set_ylim([-0.05, 1.05])
        ax2.tick_params(axis='y', labelsize=16)
        ax2.set_ylabel(r'$\hat{p}_{\nu}^{sci}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        
        plt.xlim([self.lmbda_nu_hat[0], self.lmbda_nu_hat[-1]])
        plt.xticks(fontsize=16)
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir,
            "rate-independent-probability-segment-scission-statistical-averages-vs-lmbda_nu_hat")

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
        ax1.plot(
            self.lmbda_nu_hat, self.u_nu_hat_p_nu_sur_hat, linestyle='-',
            color='blue', alpha=1, linewidth=2.5)
        ax1.vlines(
            x=self.lmbda_nu_hat_u_nu_hat_p_nu_sur_hat_max,
            ymin=self.u_nu_hat_p_nu_sur_hat[0],
            ymax=self.u_nu_hat_p_nu_sur_hat_max, linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax1.hlines(
            y=self.u_nu_hat_p_nu_sur_hat_max, xmin=self.lmbda_nu_hat[0],
            xmax=self.lmbda_nu_hat_u_nu_hat_p_nu_sur_hat_max, linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax2.plot(
            self.lmbda_nu_hat, self.s_cnu_hat_p_nu_sur_hat, linestyle='-',
            color='blue', alpha=1, linewidth=2.5)
        ax2.vlines(
            x=self.lmbda_nu_hat_s_cnu_hat_p_nu_sur_hat_max,
            ymin=self.s_cnu_hat_p_nu_sur_hat[0],
            ymax=self.s_cnu_hat_p_nu_sur_hat_max, linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax2.hlines(
            y=self.s_cnu_hat_p_nu_sur_hat_max, xmin=self.lmbda_nu_hat[0],
            xmax=self.lmbda_nu_hat_s_cnu_hat_p_nu_sur_hat_max, linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax3.plot(
            self.lmbda_nu_hat, self.psi_cnu_hat_p_nu_sur_hat, linestyle='-',
            color='blue', alpha=1, linewidth=2.5)
        ax3.vlines(
            x=self.lmbda_nu_hat_psi_cnu_hat_p_nu_sur_hat_max,
            ymin=self.psi_cnu_hat_p_nu_sur_hat[0],
            ymax=self.psi_cnu_hat_p_nu_sur_hat_max, linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax3.hlines(
            y=self.psi_cnu_hat_p_nu_sur_hat_max, xmin=self.lmbda_nu_hat[0],
            xmax=self.lmbda_nu_hat_psi_cnu_hat_p_nu_sur_hat_max, linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax4.plot(
            self.lmbda_nu_hat, self.xi_c_hat_p_nu_sur_hat, linestyle='-',
            color='blue', alpha=1, linewidth=2.5)
        ax4.vlines(
            x=self.lmbda_nu_hat_xi_c_hat_p_nu_sur_hat_max,
            ymin=self.xi_c_hat_p_nu_sur_hat[0],
            ymax=self.xi_c_hat_p_nu_sur_hat_max, linestyle=':',
            color='black', alpha=1, linewidth=1)
        ax4.hlines(
            y=self.xi_c_hat_p_nu_sur_hat_max, xmin=self.lmbda_nu_hat[0],
            xmax=self.lmbda_nu_hat_xi_c_hat_p_nu_sur_hat_max, linestyle=':',
            color='black', alpha=1, linewidth=1)

        ax1.tick_params(axis='y', labelsize=16)
        ax1.set_ylabel(r'$\hat{u}_{\nu}\times\hat{p}_{\nu}^{sur}$', fontsize=16)
        ax1.grid(True, alpha=0.25)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.set_ylabel(r'$\hat{s}_{c\nu}\times\hat{p}_{\nu}^{sur}$', fontsize=16)
        ax2.grid(True, alpha=0.25)
        ax3.tick_params(axis='y', labelsize=16)
        ax3.set_ylabel(r'$\hat{\psi}_{c\nu}\times\hat{p}_{\nu}^{sur}$', fontsize=16)
        ax3.grid(True, alpha=0.25)
        ax4.tick_params(axis='y', labelsize=16)
        ax4.set_ylabel(r'$\hat{\xi}_{c}\times\hat{p}_{\nu}^{sur}$', fontsize=16)
        ax4.grid(True, alpha=0.25)
        
        plt.xlim([self.lmbda_nu_hat[0], self.lmbda_nu_hat[-1]])
        plt.xticks(fontsize=16)
        plt.xlabel(r'$\hat{\lambda}_{\nu}$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir,
            "rate-independent-survived-segment-parameters-vs-lmbda_nu_hat")
        
        print("\n nu = 1")

        print("overline_epsilon_nu_diss_hat_crit = {}".format(self.overline_epsilon_nu_diss_hat[-1]))
        
        print("u_nu_hat_p_nu_sur_hat_max = {}".format(self.u_nu_hat_p_nu_sur_hat_max))
        print("lmbda_nu_hat_u_nu_hat_p_nu_sur_hat_max = {}".format(self.lmbda_nu_hat_u_nu_hat_p_nu_sur_hat_max))
        print("s_cnu_hat_p_nu_sur_hat_max = {}".format(self.s_cnu_hat_p_nu_sur_hat_max))
        print("lmbda_nu_hat_s_cnu_hat_p_nu_sur_hat_max = {}".format(self.lmbda_nu_hat_s_cnu_hat_p_nu_sur_hat_max))
        print("psi_cnu_hat_p_nu_sur_hat_max = {}".format(self.psi_cnu_hat_p_nu_sur_hat_max))
        print("lmbda_nu_hat_psi_cnu_hat_p_nu_sur_hat_max = {}".format(self.lmbda_nu_hat_psi_cnu_hat_p_nu_sur_hat_max))
        print("xi_c_hat_p_nu_sur_hat_max = {}".format(self.xi_c_hat_p_nu_sur_hat_max))
        print("lmbda_nu_hat_xi_c_hat_p_nu_sur_hat_max = {}".format(self.lmbda_nu_hat_xi_c_hat_p_nu_sur_hat_max))

        print("lmbda_nu_hat_p_nu_sci_hat_rms = {}".format(self.lmbda_nu_hat_p_nu_sci_hat_rms))
        print("lmbda_nu_hat_p_nu_sci_hat_mean = {}".format(self.lmbda_nu_hat_p_nu_sci_hat_mean))
        print("p_nu_sci_hat_p_nu_sci_hat_rms = {}".format(self.p_nu_sci_hat_p_nu_sci_hat_rms))
        print("p_nu_sci_hat_p_nu_sci_hat_mean = {}".format(self.p_nu_sci_hat_p_nu_sci_hat_mean))
        print("overline_epsilon_nu_diss_hat_p_nu_sci_hat_rms = {}".format(self.overline_epsilon_nu_diss_hat_p_nu_sci_hat_rms))
        print("overline_epsilon_nu_diss_hat_p_nu_sci_hat_mean = {}".format(self.overline_epsilon_nu_diss_hat_p_nu_sci_hat_mean))

        assert np.around(self.overline_epsilon_nu_diss_hat[-1]*self.single_chain.zeta_nu_char, 6) == np.around(self.single_chain.epsilon_nu_diss_hat_crit, 6)

        assert self.u_nu_hat_p_nu_sur_hat_max == self.single_chain.u_nu_hat_p_nu_sur_hat_max
        assert self.lmbda_nu_hat_u_nu_hat_p_nu_sur_hat_max == self.single_chain.lmbda_nu_hat_u_nu_hat_p_nu_sur_hat_max
        assert self.s_cnu_hat_p_nu_sur_hat_max == self.single_chain.s_cnu_hat_p_nu_sur_hat_max
        assert self.lmbda_nu_hat_s_cnu_hat_p_nu_sur_hat_max == self.single_chain.lmbda_nu_hat_s_cnu_hat_p_nu_sur_hat_max
        assert self.psi_cnu_hat_p_nu_sur_hat_max == self.single_chain.psi_cnu_hat_p_nu_sur_hat_max
        assert self.lmbda_nu_hat_psi_cnu_hat_p_nu_sur_hat_max == self.single_chain.lmbda_nu_hat_psi_cnu_hat_p_nu_sur_hat_max
        assert self.xi_c_hat_p_nu_sur_hat_max == self.single_chain.xi_c_hat_p_nu_sur_hat_max
        assert self.lmbda_nu_hat_xi_c_hat_p_nu_sur_hat_max == self.single_chain.lmbda_nu_hat_xi_c_hat_p_nu_sur_hat_max

        assert self.lmbda_nu_hat_p_nu_sci_hat_rms == self.single_chain.lmbda_nu_hat_p_nu_sci_hat_rms
        assert self.lmbda_nu_hat_p_nu_sci_hat_mean == self.single_chain.lmbda_nu_hat_p_nu_sci_hat_mean
        assert self.overline_epsilon_nu_diss_hat_p_nu_sci_hat_rms*self.single_chain.zeta_nu_char == self.single_chain.epsilon_nu_diss_hat_p_nu_sci_hat_rms
        assert self.overline_epsilon_nu_diss_hat_p_nu_sci_hat_mean*self.single_chain.zeta_nu_char == self.single_chain.epsilon_nu_diss_hat_p_nu_sci_hat_mean


if __name__ == '__main__':

    characterizer = RateIndependentSegmentScissionCharacterizer()
    characterizer.characterization()
    characterizer.finalization()