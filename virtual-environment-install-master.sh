#!/bin/bash

# For the workstation AnalyticalEngine
VENV_PATH="/home/jason_mulderrig/research/projects/composite-uFJC-scission"

# # For the laptop WSL2 running Ubuntu 20.04
# VENV_PATH="/home/jasonmulderrig/research/projects/temp"

# Set up Python virtual environment and associated Python packages

if [ ! -d ${VENV_PATH} ]
then
  mkdir -p ${VENV_PATH}
  python3 -m venv ${VENV_PATH}
  cd ${VENV_PATH}
else
  cd ${VENV_PATH}
  if [ ! -f pyvenv.cfg ]
  then
    python3 -m venv ${VENV_PATH}
  else
    rm -rf bin include lib share && rm lib64 && rm pyvenv.cfg
    python3 -m venv ${VENV_PATH}
  fi
fi

source bin/activate

pip3 install wheel && pip3 install --upgrade setuptools && pip3 install --upgrade pip
pip3 install numpy scipy mpmath sympy matplotlib orthopy quadpy pynverse

deactivate
