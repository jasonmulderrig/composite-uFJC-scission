"""The segment potential energy characterization module for composite
uFJCs
"""

# import necessary libraries
from __future__ import division
from composite_ufjc_scission import (
    CompositeuFJCScissionCharacterizer,
    CompositeuFJC,
    latex_formatting_figure,
    save_current_figure
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class SegmentPotentialEnergyFunctionCharacterizer(
        CompositeuFJCScissionCharacterizer):
    """The characterization class assessing the segment potential energy
    for composite uFJCs. It inherits all attributes and methods from
    the ``CompositeuFJCScissionCharacterizer`` class.
    """
    def __init__(self):
        """Initializes the
        ``SegmentPotentialEnergyFunctionCharacterizer`` class by
        initializing and inheriting all attributes and methods from the
        ``CompositeuFJCScissionCharacterizer`` class.
        """
        CompositeuFJCScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        p = self.parameters

        p.characterizer.lmbda_nu_inc = 0.005
        p.characterizer.lmbda_nu_min = 0.70
        p.characterizer.lmbda_nu_max = 3.50
    
    def prefix(self):
        """Set characterization prefix"""
        return "segment_potential_energy_function"
    
    def characterization(self):
        """Define characterization routine"""
        def u_nu_morse_func(morse_zeta_nu_char, morse_kappa_nu, lmbda_nu):
            """Nondimensional Morse segment potential energy
            
            This function computes the nondimensional Morse segment
            potential energy as a function of the segment stretch,
            nondimensional characteristic segment potential energy
            scale, and nondimensional segment stiffness.
            """
            sqrt_arg = morse_kappa_nu / (2.*morse_zeta_nu_char)
            exp_arg = -np.sqrt(sqrt_arg) * (lmbda_nu-1.)
            return morse_zeta_nu_char * ((1.-np.exp(exp_arg))**2-1.)
        
        def u_nu_ln_squared_func(
                ln_squared_zeta_nu_char, ln_squared_kappa_nu, lmbda_nu):
            """Nondimensional squared logaraithm segment potential
            energy
            
            This function computes the nondimensional squared logaraithm
            segment potential energy as a function of the segment
            stretch, nondimensional characteristic segment potential
            energy scale, and nondimensional segment stiffness.
            """
            return (ln_squared_kappa_nu * (np.log(lmbda_nu))**2
                    - ln_squared_zeta_nu_char)
        
        def u_nu_lj_func(lj_zeta_nu_char, lmbda_nu):
            """Nondimensional 12-6 Lennard-Jones segment potential
            energy
            
            This function computes the nondimensional 12-6 Lennard-Jones
            segment potential energy as a function of the segment
            stretch, nondimensional characteristic segment potential
            energy scale, and nondimensional segment stiffness.
            """
            return lj_zeta_nu_char * (lmbda_nu**(-12)-2.*lmbda_nu**(-6))

        cp = self.parameters.characterizer

        # nu=125, zeta_nu_char=100, and kappa_nu=1000
        single_chain = CompositeuFJC(
            rate_dependence='rate_independent',
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
        lmbda_nu        = []
        u_nu_morse      = []
        u_nu_ln_squared = []
        u_nu_lj         = []
        u_nu_har        = []
        u_nu            = []
        u_nu_har_comp   = []
        u_nu_sci_comp   = []
        
        # Calculate results through specified segment stretch values
        for lmbda_nu_indx in range(lmbda_nu_num_steps):
            lmbda_nu_val = lmbda_nu_steps[lmbda_nu_indx]
            u_nu_morse_val = u_nu_morse_func(
                single_chain.zeta_nu_char, single_chain.kappa_nu, lmbda_nu_val)
            u_nu_ln_squared_val = u_nu_ln_squared_func(
                single_chain.zeta_nu_char, single_chain.kappa_nu, lmbda_nu_val)
            u_nu_lj_val = u_nu_lj_func(single_chain.zeta_nu_char, lmbda_nu_val)
            u_nu_har_val = single_chain.u_nu_har_func(lmbda_nu_val)
            u_nu_val = single_chain.u_nu_func(lmbda_nu_val)
            u_nu_har_comp_val = single_chain.u_nu_har_comp_func(lmbda_nu_val)
            u_nu_sci_comp_val = single_chain.u_nu_sci_comp_func(lmbda_nu_val)
            
            lmbda_nu.append(lmbda_nu_val)
            u_nu_morse.append(u_nu_morse_val)
            u_nu_ln_squared.append(u_nu_ln_squared_val)
            u_nu_lj.append(u_nu_lj_val)
            u_nu_har.append(u_nu_har_val)
            u_nu.append(u_nu_val)
            u_nu_har_comp.append(u_nu_har_comp_val)
            u_nu_sci_comp.append(u_nu_sci_comp_val)
        
        overline_u_nu_morse = [
            u_nu_morse_val/single_chain.zeta_nu_char
            for u_nu_morse_val in u_nu_morse
        ]
        overline_u_nu_ln_squared = [
            u_nu_ln_squared_val/single_chain.zeta_nu_char
            for u_nu_ln_squared_val in u_nu_ln_squared
        ]
        overline_u_nu_lj = [
            u_nu_lj_val/single_chain.zeta_nu_char
            for u_nu_lj_val in u_nu_lj
        ]
        overline_u_nu_har = [
            u_nu_har_val/single_chain.zeta_nu_char
            for u_nu_har_val in u_nu_har
        ]
        overline_u_nu = [
            u_nu_val/single_chain.zeta_nu_char
            for u_nu_val in u_nu
        ]
        overline_u_nu_har_comp = [
            u_nu_har_comp_val/single_chain.zeta_nu_char
            for u_nu_har_comp_val in u_nu_har_comp
        ]
        overline_u_nu_sci_comp = [
            u_nu_sci_comp_val/single_chain.zeta_nu_char
            for u_nu_sci_comp_val in u_nu_sci_comp
        ]

        tilde_u_nu_morse = [
            overline_u_nu_morse_val+1.
            for overline_u_nu_morse_val in overline_u_nu_morse
        ]
        tilde_u_nu_ln_squared = [
            overline_u_nu_ln_squared_val+1.
            for overline_u_nu_ln_squared_val in overline_u_nu_ln_squared
        ]
        tilde_u_nu_lj = [
            overline_u_nu_lj_val+1.
            for overline_u_nu_lj_val in overline_u_nu_lj
        ]
        tilde_u_nu_har = [
            overline_u_nu_har_val+1.
            for overline_u_nu_har_val in overline_u_nu_har
        ]
        tilde_u_nu = [
            overline_u_nu_val+1.
            for overline_u_nu_val in overline_u_nu
        ]
        tilde_u_nu_har_comp = [
            overline_u_nu_har_comp_val+1.
            for overline_u_nu_har_comp_val in overline_u_nu_har_comp
        ]
        tilde_u_nu_sci_comp = [
            overline_u_nu_sci_comp_val+1.
            for overline_u_nu_sci_comp_val in overline_u_nu_sci_comp
        ]

        self.single_chain = single_chain

        self.lmbda_nu                 = lmbda_nu
        self.u_nu_morse               = u_nu_morse
        self.u_nu_ln_squared          = u_nu_ln_squared
        self.u_nu_lj                  = u_nu_lj
        self.u_nu_har                 = u_nu_har
        self.u_nu                     = u_nu
        self.u_nu_har_comp            = u_nu_har_comp
        self.u_nu_sci_comp            = u_nu_sci_comp
        self.overline_u_nu_morse      = overline_u_nu_morse
        self.overline_u_nu_ln_squared = overline_u_nu_ln_squared
        self.overline_u_nu_lj         = overline_u_nu_lj
        self.overline_u_nu_har        = overline_u_nu_har
        self.overline_u_nu            = overline_u_nu
        self.overline_u_nu_har_comp   = overline_u_nu_har_comp
        self.overline_u_nu_sci_comp   = overline_u_nu_sci_comp
        self.tilde_u_nu_morse         = tilde_u_nu_morse
        self.tilde_u_nu_ln_squared    = tilde_u_nu_ln_squared
        self.tilde_u_nu_lj            = tilde_u_nu_lj
        self.tilde_u_nu_har           = tilde_u_nu_har
        self.tilde_u_nu               = tilde_u_nu
        self.tilde_u_nu_har_comp      = tilde_u_nu_har_comp
        self.tilde_u_nu_sci_comp      = tilde_u_nu_sci_comp

    def finalization(self):
        """Define finalization analysis"""
        ppp = self.parameters.post_processing

        # plot results
        latex_formatting_figure(ppp)

        fig = plt.figure()
        plt.plot(
            self.lmbda_nu, self.overline_u_nu_har, linestyle='-',
            color='orange', alpha=1, linewidth=2.5,
            label=r'$\textrm{harmonic}$')
        plt.plot(
            self.lmbda_nu, self.overline_u_nu_ln_squared, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\textrm{log-squared}$')
        plt.plot(
            self.lmbda_nu, self.overline_u_nu_lj, linestyle='-',
            color='green', alpha=1, linewidth=2.5,
            label=r'$\textrm{Lennard-Jones}$')
        plt.plot(
            self.lmbda_nu, self.overline_u_nu_morse, linestyle='-',
            color='red', alpha=1, linewidth=2.5,
            label=r'$\textrm{Morse}$')
        plt.legend(loc='best', fontsize=12)
        plt.xlim([self.lmbda_nu[0], self.lmbda_nu[-1]])
        plt.xticks(fontsize=16)
        plt.ylim([-1.05, 0.05])
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_{\nu}$', 20, r'$\overline{u}_{\nu}$', 20,
            "overline_u_nu-vs-lmbda_nu")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            self.lmbda_nu, self.tilde_u_nu_har, linestyle='-',
            color='orange', alpha=1, linewidth=2.5,
            label=r'$\textrm{harmonic}$')
        ax.plot(
            self.lmbda_nu, self.tilde_u_nu_ln_squared, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$\textrm{log-squared}$')
        ax.plot(
            self.lmbda_nu, self.tilde_u_nu_lj, linestyle='-',
            color='green', alpha=1, linewidth=2.5,
            label=r'$\textrm{Lennard-Jones}$')
        ax.plot(
            self.lmbda_nu, self.tilde_u_nu_morse, linestyle='-',
            color='red', alpha=1, linewidth=2.5,
            label=r'$\textrm{Morse}$')
        ax.legend(loc='best', fontsize=12)
        ax.set_xlim([self.lmbda_nu[0], self.lmbda_nu[-1]])
        ax.tick_params(axis='x', labelsize=16)
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(axis='y', labelsize=16)
        ax.add_patch(
            Rectangle(
                (0.825, -0.025), 0.35, 0.175, linestyle='--', edgecolor='black',
                facecolor='none', linewidth=1.125, zorder=2))
        ax.add_patch(
            Rectangle(
                (3.00, 0.965), 0.49, 0.05, linestyle='--', edgecolor='black',
                facecolor='none', linewidth=1.125, zorder=2))
        ax.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_{\nu}$', 20, r'$\tilde{u}_{\nu}$', 20,
            "tilde_u_nu-vs-lmbda_nu")

        fig = plt.figure()
        plt.axvline(
            x=self.single_chain.lmbda_nu_crit, linestyle=':',
            color='black', alpha=1, linewidth=1)
        plt.plot(
            self.lmbda_nu, self.tilde_u_nu_har, linestyle='-',
            color='orange', alpha=1, linewidth=2.5,
            label=r'$\textrm{harmonic}$')
        plt.plot(
            self.lmbda_nu, self.tilde_u_nu_har_comp, linestyle='-',
            color='green', alpha=1, linewidth=2.5,
            label=r'$\textrm{low/intermediate force regime contribution}$')
        plt.plot(
            self.lmbda_nu, self.tilde_u_nu_sci_comp, linestyle=':',
            color='red', alpha=1, linewidth=2.5,
            label=r'$\textrm{high force regime contribution}$')
        plt.plot(
            self.lmbda_nu, self.tilde_u_nu, linestyle='--',
            color='black', alpha=1, linewidth=2.5,
            label=r'$\textrm{composite}$')
        plt.legend(loc='best', fontsize=12)
        plt.xlim([self.lmbda_nu[0], self.lmbda_nu[-1]])
        plt.xticks(fontsize=16)
        plt.ylim([-0.05, 1.05])
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_{\nu}$', 20, r'$\tilde{u}_{\nu}$', 20,
            "composite-u_nu-contributions-vs-lmbda_nu")

        fig = plt.figure()
        plt.axvline(
            x=self.single_chain.lmbda_nu_crit, linestyle=':',
            color='black', alpha=1, linewidth=1)
        plt.plot(
            self.lmbda_nu, self.overline_u_nu_lj, linestyle='-',
            color='green', alpha=1, linewidth=2.5,
            label=r'$\textrm{Lennard-Jones}$')
        plt.plot(
            self.lmbda_nu, self.overline_u_nu_morse, linestyle='-',
            color='red', alpha=1, linewidth=2.5,
            label=r'$\textrm{Morse}$')
        plt.plot(
            self.lmbda_nu, self.overline_u_nu, linestyle='-',
            color='black', alpha=1, linewidth=2.5,
            label=r'$\textrm{composite}$')
        plt.legend(loc='best', fontsize=12)
        plt.xlim([self.lmbda_nu[0], self.lmbda_nu[-1]])
        plt.xticks(fontsize=16)
        plt.ylim([-1.05, 0.05])
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_{\nu}$', 20, r'$\overline{u}_{\nu}$', 20,
            "anharmonic-overline_u_nu-vs-lmbda_nu")

        fig = plt.figure()
        plt.plot(
            self.lmbda_nu, self.u_nu, linestyle='-', color='blue',
            alpha=1, linewidth=2.5)
        plt.xlim([self.lmbda_nu[0], self.lmbda_nu[-1]])
        plt.xticks(fontsize=16)
        plt.ylim([-104, 4])
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\l_{\nu}$', 20, r'$U_{\nu}$', 20, "U_nu-vs-l_nu")

        fig = plt.figure()
        plt.plot(
            self.lmbda_nu, self.u_nu_lj, linestyle='-', color='green', alpha=1,
            linewidth=2.5, label=r'$\textrm{Lennard-Jones}$')
        plt.plot(
            self.lmbda_nu, self.u_nu_morse, linestyle='-', color='red', alpha=1,
            linewidth=2.5, label=r'$\textrm{Morse}$')
        plt.legend(loc='best', fontsize=12)
        plt.xlim([self.lmbda_nu[0], self.lmbda_nu[-1]])
        plt.xticks(fontsize=16)
        plt.ylim([-104, 4])
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_{\nu}$', 20, r'$u_{\nu}$', 20,
            "morse-lj-u_nu-vs-lmbda_nu")

        fig = plt.figure()
        plt.axvline(
            x=self.single_chain.lmbda_nu_crit, linestyle=':',
            color='black', alpha=1, linewidth=1)
        plt.plot(
            self.lmbda_nu, self.u_nu_lj, linestyle='-',
            color='green', alpha=1, linewidth=2.5,
            label=r'$\textrm{Lennard-Jones}$')
        plt.plot(
            self.lmbda_nu, self.u_nu_morse, linestyle='-',
            color='red', alpha=1, linewidth=2.5,
            label=r'$\textrm{Morse}$')
        plt.plot(
            self.lmbda_nu, self.u_nu, linestyle='-',
            color='black', alpha=1, linewidth=2.5,
            label=r'$\textrm{composite}$')
        plt.legend(loc='best', fontsize=12)
        plt.xlim([self.lmbda_nu[0], self.lmbda_nu[-1]])
        plt.xticks(fontsize=16)
        plt.ylim([-104, 4])
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.25)
        save_current_figure(
            self.savedir, r'$\lambda_{\nu}$', 20, r'$u_{\nu}$', 20,
            "anharmonic-u_nu-vs-lmbda_nu")

if __name__ == '__main__':

    characterizer = SegmentPotentialEnergyFunctionCharacterizer()
    characterizer.characterization()
    characterizer.finalization()