"""The intact chain probability density function evolution
characterization module for composite uFJCs
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
import quadpy as qp


class IntactChainProbabilityDensityFunctionEvolutionCharacterizer(
        CompositeuFJCScissionCharacterizer):
    """The characterization class assessing the intact chain probability
    density function evolution. It inherits all attributes and methods
    from the ``CompositeuFJCScissionCharacterizer`` class.
    """
    def __init__(self):
        """Initializes the
        ``IntactChainProbabilityDensityFunctionEvolutionCharacterizer``
        class by initializing and inheriting all attributes and methods
        from the ``CompositeuFJCScissionCharacterizer`` class.
        """
        CompositeuFJCScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        p = self.parameters

        nu           = 25
        zeta_nu_char = 100
        kappa_nu     = 1000

        num_points = 501

        p.characterizer.nu = nu
        p.characterizer.zeta_nu_char = zeta_nu_char
        p.characterizer.kappa_nu = kappa_nu

        p.characterizer.num_points = num_points
    
    def prefix(self):
        """Set characterization prefix"""
        return "intact_chain_probability_density_function_evolution"
    
    def characterization(self):
        """Define characterization routine"""
        def J_func(lmbda_c_eq_ref, lmbda_c_eq_crit):
            """Jacobian for the master space-equilibrium chain
            configuration space transformation
            
            This function computes the Jacobian for the master space
            equilibrium chain configuration space transformation
            """
            return (lmbda_c_eq_crit-lmbda_c_eq_ref)/2.
        def lmbda_c_eq_point_func(single_chain, point):
            """Equilibrium chain stretch as a function of master space
            coordinate point

            This function computes the equilibrium chain stretch as a
            function of master space coordinate point
            """
            J = J_func(single_chain.lmbda_c_eq_ref, single_chain.lmbda_c_eq_crit)
            return J*(1+point) + single_chain.lmbda_c_eq_ref
        def Z_intact_func(single_chain, lmbda_c_eq):
            """Integrand involved in the intact equilibrium chain
            configuration partition function integration
            
            This function computes the integrand involved in the intact 
            equilibrium chain configuration partition function integration
            as a function of the equilibrium chain stretch, integer n, and
            segment number nu
            """
            lmbda_nu = single_chain.lmbda_nu_func(lmbda_c_eq)
            psi_cnu  = single_chain.psi_cnu_func(lmbda_nu, lmbda_c_eq)
            
            return np.exp(-single_chain.nu*(psi_cnu+single_chain.zeta_nu_char))
        
        cp = self.parameters.characterizer

        # composite uFJC single chain
        single_chain = RateIndependentScissionCompositeuFJC(nu=cp.nu,
                zeta_nu_char=cp.zeta_nu_char, kappa_nu=cp.kappa_nu)
        
        # numerical quadrature scheme for integration in the master
        # space, which corresponds to the initial intact equilibrium
        # chain configuration
        scheme = qp.c1.gauss_legendre(cp.num_points)
        
        # sort points in ascending order
        indx_ascd_order = np.argsort(scheme.points)
        points = scheme.points[indx_ascd_order]
        weights = scheme.weights[indx_ascd_order]
        
        # Jacobian for the master space-equilibrium chain configuration
        # space transformation
        J = J_func(single_chain.lmbda_c_eq_ref, single_chain.lmbda_c_eq_crit)
        
        # Equilibrium chain stretches corresponding to the master space
        # points for the initial intact chain configuration
        lmbda_c_eq_0_points = lmbda_c_eq_point_func(single_chain, points)

        # Zeroth moment of the initial intact chain configuration
        # equilibrium probability density distribution without
        # normalization
        I_2 = single_chain.I_func(2, single_chain.nu) # remove nu as argument here
        
        # Total configuration equilibrium partition function
        Z_eq_tot = ((1.+single_chain.nu
                        *np.exp(-single_chain.epsilon_nu_diss_hat_crit))*I_2)
        
        # Intact chain configuration equilibrium probability density
        # distribution
        P_eq_intact_0 = np.asarray(
            [Z_intact_func(single_chain, lmbda_c_eq_0_point)/Z_eq_tot
            for lmbda_c_eq_0_point in lmbda_c_eq_0_points]
        )

        # indx = np.nonzero(P_eq_intact <= 0)
        # P_eq_intact = np.delete(P_eq_intact, indx)
        # lmbda_c_eq_points = np.delete(lmbda_c_eq_points, indx)
        # weights = np.delete(weights, indx)
        
        # Integrand for the zeroth moment of the initial intact chain
        # configuration equilibrium probability density distribution
        P_eq_intact_0_I2_intrgrnd = np.asarray(
            [P_eq_intact_0_val*lmbda_c_eq_0_point**2
            for P_eq_intact_0_val,lmbda_c_eq_0_point
            in zip(P_eq_intact_0, lmbda_c_eq_0_points)]
        )

        # Zeroth moment of the initial intact chain configuration
        # equilibrium probability density distribution as evaluated via
        # numerical quadrature
        P_eq_intact_0_I2_intrgl = (
            np.sum(np.multiply(weights, P_eq_intact_0_I2_intrgrnd))*J
        )

        # Analytical solution to the zeroth moment of the initial intact
        # chain configuration equilibrium probability density
        # distribution
        P_eq_intact_0_I2_intrgl_analytical = 1./Z_eq_tot*I_2

        # Percent error between numerical quadrature solution and
        # analytical solution for the zeroth moment of the initial
        # intact chain configuration equilibrium probability density
        # distribution
        percent_error_P_eq_intact_0_I2_intrgl = (
            np.abs(P_eq_intact_0_I2_intrgl-P_eq_intact_0_I2_intrgl_analytical)
            /P_eq_intact_0_I2_intrgl_analytical*100
        )

        # Copy of equilibrium chain stretches for evolution
        lmbda_c_eq_points = lmbda_c_eq_0_points.copy()

        # Copy of intact chain configuration probability density
        # distribution for evolution in the initial configuration
        P_intact_0 = P_eq_intact_0.copy()

        # Copy of intact chain configuration probability density
        # distribution for evolution
        P_intact = P_eq_intact_0.copy()

        # Ruptured chain configuration probability density distribution
        # for evolution
        P_sci = np.zeros(cp.num_points)

        # Define maximum fictitious timestep number
        tstep_fict_max_num = 10

        # Define fictitious timestep
        dt_fict = 0.1

        self.single_chain = single_chain
        self.lmbda_c_eq_0_points = lmbda_c_eq_0_points
        self.P_eq_intact_0 = P_eq_intact_0
        self.percent_error_P_eq_intact_0_I2_intrgl = percent_error_P_eq_intact_0_I2_intrgl

    def finalization(self):
        """Define finalization analysis"""
        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing

        # plot results
        latex_formatting_figure(ppp)

        # print(self.scheme.points)
        # print(self.scheme.weights)
        # self.scheme.show()

        # # Evaluate zeta_nu_char
        # lmbda_c_eq_max = 0
        # lmbda_nu_max   = 0

        print(self.percent_error_P_eq_intact_0_I2_intrgl)

        fig = plt.figure()
        plt.plot(
            self.lmbda_c_eq_0_points, self.P_eq_intact_0, linestyle='-',
            marker = ".", color='black', alpha=1, linewidth=2.5)
        plt.xlim([self.single_chain.lmbda_c_eq_ref, self.single_chain.lmbda_c_eq_crit])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{eq}$', 20, r'$\mathcal{P}_{eq}^{intact}\times 4\pi[\nu l_{\nu}^{eq}]^3$', 20,
            "P_eq_intact-vs-lmbda_c_eq")

        # fig = plt.figure()
        # for single_chain_indx \
        #     in range(len(self.psi_minimization_zeta_nu_char_single_chain_list)):
        #     lmbda_c_eq = (
        #         self.psi_minimization_zeta_nu_char_lmbda_c_eq___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu = (
        #         self.psi_minimization_zeta_nu_char_lmbda_nu___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu_crit = (
        #         self.psi_minimization_zeta_nu_char_single_chain_list[single_chain_indx].lmbda_nu_crit
        #     )
        #     lmbda_c_eq_crit = (
        #         self.psi_minimization_zeta_nu_char_single_chain_list[single_chain_indx].lmbda_c_eq_crit
        #     )
        #     lmbda_c_eq_max = max([lmbda_c_eq_max, lmbda_c_eq[-1]])
        #     lmbda_nu_max = max([lmbda_nu_max, lmbda_nu[-1]])
            
        #     plt.vlines(
        #         x=lmbda_c_eq_crit, ymin=lmbda_nu[0]-0.05, ymax=lmbda_nu_crit,
        #         linestyle=':', color=cp.color_list[single_chain_indx], alpha=1,
        #         linewidth=1)
        #     plt.hlines(
        #         y=lmbda_nu_crit, xmin=lmbda_c_eq[0]-0.05, xmax=lmbda_c_eq_crit,
        #         linestyle=':', color=cp.color_list[single_chain_indx], alpha=1,
        #         linewidth=1)
        
        # for single_chain_indx \
        #     in range(len(self.psi_minimization_zeta_nu_char_single_chain_list)):
        #     lmbda_c_eq = (
        #         self.psi_minimization_zeta_nu_char_lmbda_c_eq___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu = (
        #         self.psi_minimization_zeta_nu_char_lmbda_nu___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu_psimin = (
        #         self.psi_minimization_zeta_nu_char_lmbda_nu_psimin___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu_mthderr = (
        #         self.psi_minimization_zeta_nu_char_lmbda_nu_mthderr___single_chain_chunk[single_chain_indx]
        #     )
        #     plt.plot(
        #         lmbda_c_eq, lmbda_nu, linestyle='-',
        #         color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5,
        #         label=cp.psi_minimization_zeta_nu_char_label_single_chain_list[single_chain_indx])
        #     plt.plot(
        #         lmbda_c_eq, lmbda_nu_psimin, linestyle='-.',
        #         color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)

        # plt.legend(loc='best', fontsize=15)
        # plt.xlim([-0.05, lmbda_c_eq_max + 0.1])
        # plt.xticks(fontsize=20)
        # plt.ylim([0.95, lmbda_nu_max + 0.1])
        # plt.yticks(fontsize=20)
        # plt.grid(True, alpha=0.25)
        # save_current_figure(
        #     self.savedir, r'$\lambda_c^{eq}$', 30, r'$\lambda_{\nu}$', 30,
        #     "zeta_nu_char-lmbda_nu-vs-lmbda_c_eq-method-comparison")
        
        # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        # for single_chain_indx \
        #     in range(len(self.psi_minimization_zeta_nu_char_single_chain_list)):
        #     lmbda_c_eq = (
        #         self.psi_minimization_zeta_nu_char_lmbda_c_eq___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu = (
        #         self.psi_minimization_zeta_nu_char_lmbda_nu___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu_psimin = (
        #         self.psi_minimization_zeta_nu_char_lmbda_nu_psimin___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu_mthderr = (
        #         self.psi_minimization_zeta_nu_char_lmbda_nu_mthderr___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_c_eq_crit = (
        #         self.psi_minimization_zeta_nu_char_single_chain_list[single_chain_indx].lmbda_c_eq_crit
        #     )

        #     lmbda_c_eq__lmbda_c_eq_crit = [
        #         x/lmbda_c_eq_crit for x in lmbda_c_eq
        #     ]

        #     ax1.plot(
        #         lmbda_c_eq__lmbda_c_eq_crit, lmbda_nu, linestyle='-',
        #         color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5,
        #         label=cp.psi_minimization_zeta_nu_char_label_single_chain_list[single_chain_indx])
        #     ax1.plot(
        #         lmbda_c_eq__lmbda_c_eq_crit, lmbda_nu_psimin, linestyle='-.',
        #         color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
            
        #     ax2.semilogy(
        #         lmbda_c_eq__lmbda_c_eq_crit, lmbda_nu_mthderr, linestyle='-',
        #         color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5,
        #         label=cp.psi_minimization_zeta_nu_char_label_single_chain_list[single_chain_indx])
        
        # ax1.legend(loc='best', fontsize=15)
        # ax1.set_ylim([0.95, lmbda_nu_max + 0.1])
        # ax1.tick_params(axis='y', labelsize=20)
        # ax1.set_ylabel(r'$\lambda_{\nu}$', fontsize=30)
        # ax1.grid(True, alpha=0.25)
        # ax2.tick_params(axis='y', labelsize=20)
        # ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=30)
        # ax2.grid(True, alpha=0.25)
        # plt.xlim([-0.05, cp.zeta_nu_char_lmbda_c_eq_crit_factor + 0.05])
        # plt.xticks(fontsize=20)
        # plt.xlabel(r'$\lambda_c^{eq}/(\lambda_c^{eq})^{crit}$', fontsize=30)
        # save_current_figure_no_labels(
        #     self.savedir,
        #     "zeta_nu_char-lmbda_nu-vs-lmbda_c_eq__lmbda_c_eq_crit-method-comparison")

        
        # # Evaluate zeta_nu_char
        # lmbda_c_eq_max = 0
        # lmbda_nu_max   = 0

        # fig = plt.figure()
        # for single_chain_indx \
        #     in range(len(self.psi_minimization_kappa_nu_single_chain_list)):
        #     lmbda_c_eq = (
        #         self.psi_minimization_kappa_nu_lmbda_c_eq___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu = (
        #         self.psi_minimization_kappa_nu_lmbda_nu___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu_crit = (
        #         self.psi_minimization_kappa_nu_single_chain_list[single_chain_indx].lmbda_nu_crit
        #     )
        #     lmbda_c_eq_crit = (
        #         self.psi_minimization_kappa_nu_single_chain_list[single_chain_indx].lmbda_c_eq_crit
        #     )
        #     lmbda_c_eq_max = max([lmbda_c_eq_max, lmbda_c_eq[-1]])
        #     lmbda_nu_max = max([lmbda_nu_max, lmbda_nu[-1]])
            
        #     plt.vlines(
        #         x=lmbda_c_eq_crit, ymin=lmbda_nu[0]-0.05, ymax=lmbda_nu_crit,
        #         linestyle=':', color=cp.color_list[single_chain_indx], alpha=1,
        #         linewidth=1)
        #     plt.hlines(
        #         y=lmbda_nu_crit, xmin=lmbda_c_eq[0]-0.05, xmax=lmbda_c_eq_crit,
        #         linestyle=':', color=cp.color_list[single_chain_indx], alpha=1,
        #         linewidth=1)
        
        # for single_chain_indx \
        #     in range(len(self.psi_minimization_kappa_nu_single_chain_list)):
        #     lmbda_c_eq = (
        #         self.psi_minimization_kappa_nu_lmbda_c_eq___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu = (
        #         self.psi_minimization_kappa_nu_lmbda_nu___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu_psimin = (
        #         self.psi_minimization_kappa_nu_lmbda_nu_psimin___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu_mthderr = (
        #         self.psi_minimization_kappa_nu_lmbda_nu_mthderr___single_chain_chunk[single_chain_indx]
        #     )
        #     plt.plot(
        #         lmbda_c_eq, lmbda_nu, linestyle='-',
        #         color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5,
        #         label=cp.psi_minimization_kappa_nu_label_single_chain_list[single_chain_indx])
        #     plt.plot(
        #         lmbda_c_eq, lmbda_nu_psimin, linestyle='-.',
        #         color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)

        # plt.legend(loc='best', fontsize=15)
        # plt.xlim([-0.05, lmbda_c_eq_max + 0.1])
        # plt.xticks(fontsize=20)
        # plt.ylim([0.95, lmbda_nu_max + 0.1])
        # plt.yticks(fontsize=20)
        # plt.grid(True, alpha=0.25)
        # save_current_figure(
        #     self.savedir, r'$\lambda_c^{eq}$', 30, r'$\lambda_{\nu}$', 30,
        #     "kappa_nu-lmbda_nu-vs-lmbda_c_eq-method-comparison")
        
        # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        # for single_chain_indx \
        #     in range(len(self.psi_minimization_kappa_nu_single_chain_list)):
        #     lmbda_c_eq = (
        #         self.psi_minimization_kappa_nu_lmbda_c_eq___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu = (
        #         self.psi_minimization_kappa_nu_lmbda_nu___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu_psimin = (
        #         self.psi_minimization_kappa_nu_lmbda_nu_psimin___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_nu_mthderr = (
        #         self.psi_minimization_kappa_nu_lmbda_nu_mthderr___single_chain_chunk[single_chain_indx]
        #     )
        #     lmbda_c_eq_crit = (
        #         self.psi_minimization_kappa_nu_single_chain_list[single_chain_indx].lmbda_c_eq_crit
        #     )

        #     lmbda_c_eq__lmbda_c_eq_crit = [
        #         x/lmbda_c_eq_crit for x in lmbda_c_eq
        #     ]

        #     ax1.plot(
        #         lmbda_c_eq__lmbda_c_eq_crit, lmbda_nu, linestyle='-',
        #         color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5,
        #         label=cp.psi_minimization_kappa_nu_label_single_chain_list[single_chain_indx])
        #     ax1.plot(
        #         lmbda_c_eq__lmbda_c_eq_crit, lmbda_nu_psimin, linestyle='-.',
        #         color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5)
            
        #     ax2.semilogy(
        #         lmbda_c_eq__lmbda_c_eq_crit, lmbda_nu_mthderr, linestyle='-',
        #         color=cp.color_list[single_chain_indx], alpha=1, linewidth=2.5,
        #         label=cp.psi_minimization_kappa_nu_label_single_chain_list[single_chain_indx])
        
        # ax1.legend(loc='best', fontsize=15)
        # ax1.set_ylim([0.95, lmbda_nu_max + 0.1])
        # ax1.tick_params(axis='y', labelsize=20)
        # ax1.set_ylabel(r'$\lambda_{\nu}$', fontsize=30)
        # ax1.grid(True, alpha=0.25)
        # ax2.tick_params(axis='y', labelsize=20)
        # ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=30)
        # ax2.grid(True, alpha=0.25)
        # plt.xlim([-0.05, cp.kappa_nu_lmbda_c_eq_crit_factor + 0.05])
        # plt.xticks(fontsize=20)
        # plt.xlabel(r'$\lambda_c^{eq}/(\lambda_c^{eq})^{crit}$', fontsize=30)
        # save_current_figure_no_labels(
        #     self.savedir,
        #     "kappa_nu-lmbda_nu-vs-lmbda_c_eq__lmbda_c_eq_crit-method-comparison")

if __name__ == '__main__':

    characterizer = (
        IntactChainProbabilityDensityFunctionEvolutionCharacterizer()
    )
    characterizer.characterization()
    characterizer.finalization()