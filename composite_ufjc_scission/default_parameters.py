"""
The default parameters module for the composite uFJC scission model.
"""

# import necessary libraries
from types import SimpleNamespace


def default_parameters():
    """Save default parameters.
    
    This core function stores the default parameters from each subset of
    parameters to a main parameters SimpleNamespace object, which is
    then returned.
    """
    parameters = SimpleNamespace()
    subset_list = ["characterizer", "post_processing"]
    for subparset in subset_list:
        subparset_is = eval("default_"+subparset+"_parameters()")
        setattr(parameters, subparset, subparset_is)
    return parameters

def default_characterizer_parameters():
    """Save default characterizer parameters.
    
    This function stores the default characterizer parameters to a
    characterizer SimpleNamespace object, which is then returned.
    """
    characterizer = SimpleNamespace()

    # 5, 125, 3125
    nu_single_chain_list = [int(5**(2*i-1)) for i in range(1, 4)]
    nu_label_single_chain_list = [
        r'$\nu='+str(nu_single_chain_list[i])+'$'
        for i in range(len(nu_single_chain_list))
    ]

    characterizer.nu_single_chain_list       = nu_single_chain_list
    characterizer.nu_label_single_chain_list = nu_label_single_chain_list

    zeta_nu_char_single_chain_list = [10, 50, 100, 500, 1000]
    # Reverse the order of the zeta_nu_char list
    zeta_nu_char_single_chain_list = zeta_nu_char_single_chain_list[::-1]
    zeta_nu_char_label_single_chain_list = [
        r'$\zeta_{\nu}^{char}='+str(zeta_nu_char_single_chain_list[i])+'$'
        for i in range(len(zeta_nu_char_single_chain_list))
    ]

    characterizer.zeta_nu_char_single_chain_list = (
        zeta_nu_char_single_chain_list
    )
    characterizer.zeta_nu_char_label_single_chain_list = (
        zeta_nu_char_label_single_chain_list
    )

    kappa_nu_single_chain_list       = [100, 500, 1000, 5000, 10000]
    kappa_nu_label_single_chain_list = [
        r'$\kappa_{\nu}='+str(kappa_nu_single_chain_list[i])+'$'
        for i in range(len(kappa_nu_single_chain_list))
    ]

    characterizer.kappa_nu_single_chain_list = kappa_nu_single_chain_list
    characterizer.kappa_nu_label_single_chain_list = (
        kappa_nu_label_single_chain_list
    )

    psi_minimization_zeta_nu_char_single_chain_list = (
        zeta_nu_char_single_chain_list[0:4]
    )
    psi_minimization_zeta_nu_char_label_single_chain_list = (
        zeta_nu_char_label_single_chain_list[0:4]
    )

    characterizer.psi_minimization_zeta_nu_char_single_chain_list = (
        psi_minimization_zeta_nu_char_single_chain_list
    )
    characterizer.psi_minimization_zeta_nu_char_label_single_chain_list = (
        psi_minimization_zeta_nu_char_label_single_chain_list
    )

    psi_minimization_kappa_nu_single_chain_list = [
        int(kappa_nu_single_chain_list[i]/10) 
        for i in range(len(kappa_nu_single_chain_list))
    ]
    psi_minimization_kappa_nu_label_single_chain_list = [
        r'$\kappa_{\nu}='+str(psi_minimization_kappa_nu_single_chain_list[i])+'$'
        for i in range(len(psi_minimization_kappa_nu_single_chain_list))
    ]

    characterizer.psi_minimization_kappa_nu_single_chain_list = (
        psi_minimization_kappa_nu_single_chain_list
    )
    characterizer.psi_minimization_kappa_nu_label_single_chain_list = (
        psi_minimization_kappa_nu_label_single_chain_list
    )

    bergapprx_lmbda_nu_cutoff = 0.84136

    characterizer.bergapprx_lmbda_nu_cutoff = bergapprx_lmbda_nu_cutoff

    # nu = 5 -> nu = 3125
    nu_chain_network_list = [i for i in range(5, 5**5+1)]

    characterizer.nu_chain_network_list = nu_chain_network_list

    zeta_nu_char_chain_network_list       = [50, 100, 500]
    zeta_nu_char_label_chain_network_list = [
        r'$\zeta_{\nu}^{char}='+str(zeta_nu_char_chain_network_list[i])+'$'
        for i in range(len(zeta_nu_char_chain_network_list))
    ]

    characterizer.zeta_nu_char_chain_network_list = (
        zeta_nu_char_chain_network_list
    )
    characterizer.zeta_nu_char_label_chain_network_list = (
        zeta_nu_char_label_chain_network_list
    )

    kappa_nu_chain_network_list       = [500, 1000, 5000]
    kappa_nu_label_chain_network_list = [
        r'$\kappa_{\nu}='+str(kappa_nu_chain_network_list[i])+'$'
        for i in range(len(kappa_nu_chain_network_list))
    ]

    characterizer.kappa_nu_chain_network_list = kappa_nu_chain_network_list
    characterizer.kappa_nu_label_chain_network_list = (
        kappa_nu_label_chain_network_list
    )

    return characterizer

def default_post_processing_parameters():
    """Save default post-processing parameters.
    
    This function stores the default post-processing parameters to a
    characterizer SimpleNamespace object, which is then returned.
    """
    
    post_processing = SimpleNamespace()

    axes_linewidth      = 1.0
    font_family         = "sans-serif"
    text_usetex         = True
    ytick_right         = True
    ytick_direction     = "in"
    xtick_top           = True
    xtick_direction     = "in"
    xtick_minor_visible = True

    post_processing.axes_linewidth      = axes_linewidth
    post_processing.font_family         = font_family
    post_processing.text_usetex         = text_usetex
    post_processing.ytick_right         = ytick_right
    post_processing.ytick_direction     = ytick_direction
    post_processing.xtick_top           = xtick_top
    post_processing.xtick_direction     = xtick_direction
    post_processing.xtick_minor_visible = xtick_minor_visible

    return post_processing