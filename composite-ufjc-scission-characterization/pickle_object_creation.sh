#!/bin/bash

# Ideally, the composite-uFJC-scission virtual environment should be activated
# The CompositeuFJCScissionCharacterizer objects below must be acted upon only by the characterization() module

python3 pade2bergstrom_critical_point.py
python3 chain_network_reference_stretches.py