"""The Pade-to-Bergstrom critical point characterization module for 
composite uFJCs with varying nondimensional segment stiffness
"""

# import external modules
from __future__ import division
from composite_ufjc_scission import (
    CompositeuFJCScissionCharacterizer,
    RateIndependentScissionCompositeuFJC,
    latex_formatting_figure,
    save_current_figure,
    save_pickle_object,
    load_pickle_object
)
from scipy import optimize
import matplotlib.pyplot as plt


class Pade2BergstromCriticalPointCharacterizer(
        CompositeuFJCScissionCharacterizer):
    """The characterization class assessing the Pade-to-Bergstrom 
    critical point for composite uFJCs with varying nondimensional
    segment stiffness. It inherits all attributes and methods from the
    ``CompositeuFJCScissionCharacterizer`` class.
    """
    def __init__(self):
        """Initializes the ``Pade2BergstromCriticalPointCharacterizer``
        class by initializing and inheriting all attributes and methods
        from the ``CompositeuFJCScissionCharacterizer`` class.
        """
        CompositeuFJCScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        p = self.parameters

        # List of physically relevant kappa_nu 
        # kappa_nu from 1 -> 10000 reduced the curve fit quality over
        # the range of physically relevant kappa_nu. Given this,
        # kappa_nu will range from 100 -> 10000
        p.characterizer.kappa_nu_pade2berg_crit_list = [
            i for i in range(100, 10001)
        ]

    def prefix(self):
        """Set characterization prefix"""
        return "pade2bergstrom_critical_point"
    
    def characterization(self):
        """Define characterization routine"""
        def lmbda_c_eq_pade2berg_func_kappa_nu_fit(kappa_nu, n, b):
            """Pade-to-Bergstrom (P2B) critical equilibrium chain
            stretch function for curve fitting
            
            This function represents the Pade-to-Bergstrom (P2B)
            critical equilibrium chain stretch function used in a scipy 
            optimize curve_fit analysis to determine parameters n and b
            """
            return 1. / kappa_nu**n + b

        cp = self.parameters.characterizer

        single_chain_list = [
            RateIndependentScissionCompositeuFJC(
                nu=25, zeta_nu_char=100,
                kappa_nu=cp.kappa_nu_pade2berg_crit_list[kappa_nu_indx])
            for kappa_nu_indx in range(len(cp.kappa_nu_pade2berg_crit_list))
        ]
        lmbda_c_eq_pade2berg_crit = [
            single_chain.pade2berg_crit_func()[1]
            for single_chain in single_chain_list
        ]

        popt, pcov = (
            optimize.curve_fit(
                lmbda_c_eq_pade2berg_func_kappa_nu_fit,
                cp.kappa_nu_pade2berg_crit_list, lmbda_c_eq_pade2berg_crit)
        )
        n = popt[0]
        b = popt[1]
        print("n = {}".format(n))
        print("b = {}".format(b))

        lmbda_c_eq_pade2berg_crit_curve_fit = [
            lmbda_c_eq_pade2berg_func_kappa_nu_fit(
                cp.kappa_nu_pade2berg_crit_list[kappa_nu_indx], n, b)
            for kappa_nu_indx in range(len(cp.kappa_nu_pade2berg_crit_list))
        ]

        save_pickle_object(
            self.savedir, cp.kappa_nu_pade2berg_crit_list,
            "kappa_nu_pade2berg_crit")
        save_pickle_object(
            self.savedir, lmbda_c_eq_pade2berg_crit,
            "lmbda_c_eq_pade2berg_crit")
        save_pickle_object(
            self.savedir, lmbda_c_eq_pade2berg_crit_curve_fit,
            "lmbda_c_eq_pade2berg_crit_curve_fit")

    def finalization(self):
        """Define finalization analysis"""
        ppp = self.parameters.post_processing

        kappa_nu_pade2berg_crit = (
            load_pickle_object(self.savedir, "kappa_nu_pade2berg_crit")
        )
        lmbda_c_eq_pade2berg_crit = (
            load_pickle_object(self.savedir, "lmbda_c_eq_pade2berg_crit")
        )
        lmbda_c_eq_pade2berg_crit_curve_fit = (
            load_pickle_object(
                self.savedir, "lmbda_c_eq_pade2berg_crit_curve_fit")
        )

        # plot results
        latex_formatting_figure(ppp)

        fig = plt.figure()
        plt.plot(
            kappa_nu_pade2berg_crit, lmbda_c_eq_pade2berg_crit,
            linestyle='-', color='blue', alpha=1, linewidth=2.5,
            label=r'$\textrm{Exact calculation}$')
        plt.plot(
            kappa_nu_pade2berg_crit, lmbda_c_eq_pade2berg_crit_curve_fit,
            linestyle='--', color='red', alpha=1, linewidth=2.5,
            label=r'$\textrm{Curve fit}$')
        plt.legend(loc='best', fontsize=12)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\kappa_{\nu}$', 20,
            r'$(\lambda_c^{eq})^{pade2berg}$', 20,
            "pade2berg-critical-point-equilibrium-chain-stretch-vs-kappa_nu")

if __name__ == '__main__':

    characterizer = Pade2BergstromCriticalPointCharacterizer()
    # characterizer.characterization()
    characterizer.finalization()