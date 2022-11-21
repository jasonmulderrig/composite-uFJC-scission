"""The segment Helmholtz free energy characterization module for
composite uFJCs
"""

# import external modules
from __future__ import division
from composite_ufjc_scission import (
    CompositeuFJCScissionCharacterizer,
    CompositeuFJC,
    latex_formatting_figure,
    save_current_figure
)
import numpy as np
import matplotlib.pyplot as plt


class SegmentHelmholtzFreeEnergyFunctionCharacterizer(
        CompositeuFJCScissionCharacterizer):
    """The characterization class assessing the segment Helmholtz free
    energy for composite uFJCs. It inherits all attributes and methods
    from the ``CompositeuFJCScissionCharacterizer`` class.
    """
    def __init__(self):
        """Initializes the
        ``SegmentHelmholtzFreeEnergyFunctionCharacterizer`` class by
        initializing and inheriting all attributes and methods from the
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
        return "segment_helmholtz_free_energy_function"
    
    def characterization(self):
        """Define characterization routine"""
        cp = self.parameters.characterizer

        # nu=125, zeta_nu_char=100, and kappa_nu=1000
        single_chain = CompositeuFJC(
            rate_dependence='rate_independent', scission_model='exact',
            nu=cp.nu_single_chain_list[1],
            zeta_nu_char=cp.zeta_nu_char_single_chain_list[2],
            kappa_nu=cp.kappa_nu_single_chain_list[2])

        # Define the equilibrium stretch values to calculate over
        lmbda_c_eq_num_steps = (int(
            np.around(
                (cp.lmbda_c_eq_max-cp.lmbda_c_eq_min)/cp.lmbda_c_eq_inc))
            + 1)
        lmbda_c_eq_steps = np.linspace(
            cp.lmbda_c_eq_min, cp.lmbda_c_eq_max, lmbda_c_eq_num_steps)
        
        # Make arrays to allocate results
        lmbda_c_eq = []
        u_nu       = []
        s_cnu      = []
        psi_cnu    = []
        
        # Calculate results through specified equilibrium chain stretch
        # values
        for lmbda_c_eq_indx in range(lmbda_c_eq_num_steps):
            lmbda_c_eq_val = lmbda_c_eq_steps[lmbda_c_eq_indx]
            lmbda_nu_val   = single_chain.lmbda_nu_func(lmbda_c_eq_val)
            lmbda_comp_nu  = lmbda_c_eq_val - lmbda_nu_val + 1.
            u_nu_val       = single_chain.u_nu_func(lmbda_nu_val)
            s_cnu_val      = single_chain.s_cnu_func(lmbda_comp_nu)
            psi_cnu_val    = (
                single_chain.psi_cnu_func(lmbda_nu_val, lmbda_c_eq_val)
            )
            
            lmbda_c_eq.append(lmbda_c_eq_val)
            u_nu.append(u_nu_val)
            s_cnu.append(s_cnu_val)
            psi_cnu.append(psi_cnu_val)

        self.single_chain = single_chain

        self.lmbda_c_eq = lmbda_c_eq
        self.u_nu       = u_nu
        self.s_cnu      = s_cnu
        self.psi_cnu    = psi_cnu

    def finalization(self):
        """Define finalization analysis"""
        ppp = self.parameters.post_processing

        # plot results
        latex_formatting_figure(ppp)

        fig = plt.figure()
        plt.axvline(
            x=self.single_chain.lmbda_c_eq_crit, linestyle=':', color='black',
            alpha=1, linewidth=1)
        plt.plot(
            self.lmbda_c_eq, self.u_nu, linestyle='-.', color='blue',
            alpha=1, linewidth=2.5, label=r'$u_{\nu}$')
        plt.plot(
            self.lmbda_c_eq, self.s_cnu, linestyle='-', color='green',
            alpha=1, linewidth=2.5, label=r'$s_{c\nu}$')
        plt.plot(
            self.lmbda_c_eq, self.psi_cnu, linestyle='-', color='red',
            alpha=1, linewidth=2.5, label=r'$\psi_{c\nu}$')
        plt.legend(loc='best', fontsize=12)
        plt.ylim([-110, 10])
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_c^{eq}$', 20,
            r'$u_{\nu},~s_{c\nu},~\psi_{c\nu}$', 20,
            "psi_cnu-vs-lmbda_c_eq")

if __name__ == '__main__':

    characterizer = SegmentHelmholtzFreeEnergyFunctionCharacterizer()
    characterizer.characterization()
    characterizer.finalization()