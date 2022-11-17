#!/bin/bash

# Ideally, the composite-uFJC-scission virtual environment should be
# activated
# The CompositeuFJCScissionCharacterizer objects in
# chain_network_reference_stretches.py and
# pade2bergstrom_critical_point.py below must be acted upon only by the
# finalization() module, with the pickle objects being loaded in that
# module already created

python3 chain_helmholtz_free_energy_minimization_method_comparison.py
python3 chain_mechanical_response.py
python3 chain_network_reference_stretches.py
python3 equilibrium_chain_stretch_function.py
python3 pade2bergstrom_critical_point.py
python3 rate_independent_chain_scission.py
python3 rate_independent_segment_scission.py
python3 segment_helmholtz_free_energy_function.py
python3 segment_potential_energy_function.py
python3 segment_stretch_function.py
python3 total_distorted_segment_potential_energy_function.py
python3 total_segment_potential_energy_function.py
python3 AFM_chain_tensile_test_curve_fit.py